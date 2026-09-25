/**
 * Server wiring for the emulated Claude advisor (src/claude/advisor-loop.ts). The routed model's
 * continuation goes through handleResponses. The consultation does too, so an OpenCodex alias
 * resolves exactly as a client-selected model would, admission scope included, unless the
 * advisor is a Claude model that qualifies for the native Anthropic passthrough: then it goes
 * straight to Anthropic with the caller's own credential, like a Claude main model.
 */
import { runAdvisorLoop } from "../claude/advisor-loop";
import { nativeAdvisorMessagesBody, nativeAdvisorResponse, type AdvisorToolSpec } from "../claude/advisor";
import { stripOneMillionMarker } from "../claude/context-windows";
import { resolveInboundModel } from "../claude/inbound";
import { createTranslatorBudget, finalizeTranslatorBudgetResponse, type TranslatorBudget } from "../lib/translator-budget";
import { routeModel } from "../router";
import { evidenceFromBody } from "../routing/request-evidence";
import type { OcxConfig } from "../types";
import { assertRouteAllowedByScope, type AdmissionModelScope } from "./admission-model-scope";
import { tryClaimNativeMainProfileForTurn } from "../codex/native-main-admission";
import { tryAdmitTurn } from "./lifecycle";
import { addFinalRequestLog, httpStatusForRequestLogTerminal, nextRequestLogId, type RequestLogContext } from "./request-log";
import { sessionLaneIdFromRequest } from "./request-log-conversation";
import { isNativePassthroughSseResponse, markNativePassthroughSseResponse, responseWithDeferredRequestLog } from "./relay";
import { handleResponses } from "./responses";

type Rec = Record<string, unknown>;
type ReplayOptions = NonNullable<Parameters<typeof handleResponses>[3]>;

/**
 * Native Anthropic passthrough for a Claude advisor model. Present only when the advisor model
 * qualifies under the same rule as a Claude main model (claude-messages.ts wantsNativePassthrough).
 */
export interface NativeAdvisorPassthrough {
  /** The Claude model id to send, as the passthrough rule accepted it. */
  model: string;
  /** Send an Anthropic Messages body with the caller's own credential; returns the Anthropic reply. */
  send(body: Rec, signal: AbortSignal, log: { logCtx: RequestLogContext; logIds?: { requestId: string; start: number } }): Promise<Response>;
}

export interface ClaudeAdvisorContext {
  spec: AdvisorToolSpec;
  /** Translated Responses body of the first iteration. */
  body: Rec;
  cc: OcxConfig["claudeCode"];
  config: OcxConfig;
  replayConfig: OcxConfig;
  /** Options shared by every replay of this turn, without per-dispatch budget, signal or log hooks. */
  replayOptions: ReplayOptions;
  headers: Headers;
  scope: AdmissionModelScope | undefined;
  logCtx: RequestLogContext;
  /** Present when the turn is request-logged: each sub-dispatch then gets its own row. */
  logged: boolean;
  signal: AbortSignal;
  translatorBudget: TranslatorBudget;
  native?: NativeAdvisorPassthrough;
}

/**
 * A fresh request-log context for one sub-dispatch. Only the turn's identity carries over: a
 * spread copy would share the live `attempts` array (and the active attempt) with the turn, so
 * every later row would aggregate the usage of every dispatch before it.
 */
function subDispatchLogContext(turn: RequestLogContext): RequestLogContext {
  const fresh: RequestLogContext = { model: "unknown", provider: "unknown" };
  if (turn.requestMetricsRecorder) fresh.requestMetricsRecorder = turn.requestMetricsRecorder;
  if (turn.conversationId !== undefined) fresh.conversationId = turn.conversationId;
  if (turn.surface !== undefined) fresh.surface = turn.surface;
  if (turn.apiKeyId !== undefined) fresh.apiKeyId = turn.apiKeyId;
  if (turn.admissionKind !== undefined) fresh.admissionKind = turn.admissionKind;
  if (turn.inboundProtocol !== undefined) fresh.inboundProtocol = turn.inboundProtocol;
  if (turn.claudeCompatibility !== undefined) fresh.claudeCompatibility = turn.claudeCompatibility;
  return fresh;
}

function resolveAdvisorModel(spec: AdvisorToolSpec, cc: OcxConfig["claudeCode"]): string | null {
  try {
    return resolveInboundModel(stripOneMillionMarker(spec.model), cc);
  } catch {
    return null;
  }
}

/**
 * Wrap the first iteration's response in the advisor loop. Anything other than an ok event
 * stream is returned untouched, so errors keep their existing Anthropic mapping.
 */
export function withClaudeAdvisor(upstream: Response, ctx: ClaudeAdvisorContext): Response {
  const contentType = upstream.headers.get("content-type") ?? "";
  if (!upstream.ok || !upstream.body || !contentType.includes("text/event-stream")) return upstream;

  const dispatch = async (body: Rec, signal: AbortSignal): Promise<Response> => {
    // The first iteration's lease settles when its stream ends, so every later dispatch is
    // admitted as a turn of its own, the same way the HTTP route admits one.
    const lease = tryAdmitTurn(sessionLaneIdFromRequest(ctx.headers));
    if (!lease) {
      return new Response(JSON.stringify({ error: { message: "proxy is not admitting new turns" } }), {
        status: 503,
        headers: { "content-type": "application/json" },
      });
    }
    if (ctx.replayOptions.trustedClaudeMainAuth) tryClaimNativeMainProfileForTurn(lease);
    const budget = createTranslatorBudget();
    try {
      const request = new Request("http://localhost/v1/responses", {
        method: "POST",
        headers: ctx.headers,
        body: JSON.stringify(body),
      });
      const logCtx = subDispatchLogContext(ctx.logCtx);
      const start = Date.now();
      const requestId = nextRequestLogId(start);
      // A native Responses forward is logged by its terminal callbacks, not the stream tap below.
      let nativeLogged = false;
      const finalizeNativeLog = (status: number, meta: Parameters<typeof addFinalRequestLog>[4]) => {
        if (!ctx.logged || nativeLogged) return;
        nativeLogged = true;
        addFinalRequestLog(requestId, start, logCtx, status, meta);
      };
      const response = await handleResponses(request, ctx.replayConfig, logCtx, {
        ...ctx.replayOptions,
        turnAdmissionLease: lease,
        abortSignal: signal,
        translatorBudget: budget,
        onNativePassthroughTerminal: status => finalizeNativeLog(httpStatusForRequestLogTerminal(status, logCtx), { terminalStatus: status, closeReason: "terminal" }),
        onNativePassthroughCancel: () => finalizeNativeLog(499, { closeReason: "client_cancel" }),
      });
      if (!lease.isTransferred()) lease.release();
      const logged = ctx.logged ? responseWithDeferredRequestLog(response, requestId, start, logCtx) : response;
      return finalizeTranslatorBudgetResponse(logged, budget);
    } catch (error) {
      lease.release();
      budget.dispose();
      throw error;
    }
  };

  const native = ctx.native;
  const dispatchNativeAdvisor = async (nativeCtx: NativeAdvisorPassthrough, body: Rec, signal: AbortSignal): Promise<Response> => {
    const logCtx = subDispatchLogContext(ctx.logCtx);
    const start = Date.now();
    const response = await nativeCtx.send(nativeAdvisorMessagesBody(nativeCtx.model, body), signal, {
      logCtx,
      ...(ctx.logged ? { logIds: { requestId: nextRequestLogId(start), start } } : {}),
    });
    return nativeAdvisorResponse(response);
  };

  const advisorModel = native ? native.model : resolveAdvisorModel(ctx.spec, ctx.cc);
  const stitched = new Response(runAdvisorLoop({
    body: ctx.body,
    spec: ctx.spec,
    first: upstream,
    advisorModel,
    dispatchTurn: body => dispatch(body, ctx.signal),
    dispatchAdvisor: async (body, signal) => {
      // A Claude advisor on the caller's own Anthropic credential never enters routing.
      if (native) return dispatchNativeAdvisor(native, body, signal);
      const model = String(body.model);
      // The advisor is a second model choice made by the client; it gets the same scope check.
      assertRouteAllowedByScope(ctx.scope, model, routeModel(ctx.config, model, evidenceFromBody(body)));
      return dispatch(body, signal);
    },
    translatorBudget: ctx.translatorBudget,
    signal: ctx.signal,
  }), { status: upstream.status, headers: upstream.headers });
  // Native passthrough turns are logged by their terminal callbacks, not by the stream tap.
  return isNativePassthroughSseResponse(upstream) ? markNativePassthroughSseResponse(stitched) : stitched;
}
