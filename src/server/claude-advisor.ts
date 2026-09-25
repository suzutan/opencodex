/**
 * Server wiring for the emulated Claude advisor (src/claude/advisor-loop.ts): both the advisor
 * consultation and the routed model's continuation go through handleResponses, so a Claude id
 * and an OpenCodex alias resolve exactly as a client-selected model would, admission scope
 * included.
 */
import { runAdvisorLoop } from "../claude/advisor-loop";
import type { AdvisorToolSpec } from "../claude/advisor";
import { stripOneMillionMarker } from "../claude/context-windows";
import { resolveInboundModel } from "../claude/inbound";
import { createTranslatorBudget, finalizeTranslatorBudgetResponse, type TranslatorBudget } from "../lib/translator-budget";
import { routeModel } from "../router";
import { evidenceFromBody } from "../routing/request-evidence";
import type { OcxConfig } from "../types";
import { assertRouteAllowedByScope, type AdmissionModelScope } from "./admission-model-scope";
import { tryClaimNativeMainProfileForTurn } from "../codex/native-main-admission";
import { tryAdmitTurn } from "./lifecycle";
import { nextRequestLogId, type RequestLogContext } from "./request-log";
import { sessionLaneIdFromRequest } from "./request-log-conversation";
import { isNativePassthroughSseResponse, markNativePassthroughSseResponse, responseWithDeferredRequestLog } from "./relay";
import { handleResponses } from "./responses";

type Rec = Record<string, unknown>;
type ReplayOptions = NonNullable<Parameters<typeof handleResponses>[3]>;

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
      const logCtx: RequestLogContext = { ...ctx.logCtx };
      delete logCtx.routeDecision;
      const start = Date.now();
      const response = await handleResponses(request, ctx.replayConfig, logCtx, {
        ...ctx.replayOptions,
        turnAdmissionLease: lease,
        abortSignal: signal,
        translatorBudget: budget,
      });
      if (!lease.isTransferred()) lease.release();
      const logged = ctx.logged ? responseWithDeferredRequestLog(response, nextRequestLogId(start), start, logCtx) : response;
      return finalizeTranslatorBudgetResponse(logged, budget);
    } catch (error) {
      lease.release();
      budget.dispose();
      throw error;
    }
  };

  const advisorModel = resolveAdvisorModel(ctx.spec, ctx.cc);
  const stitched = new Response(runAdvisorLoop({
    body: ctx.body,
    spec: ctx.spec,
    first: upstream,
    advisorModel,
    dispatchTurn: body => dispatch(body, ctx.signal),
    dispatchAdvisor: async (body, signal) => {
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
