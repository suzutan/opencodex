/**
 * Advisor consultation loop for translated Claude Messages turns (see src/claude/advisor.ts).
 *
 * The loop sits between the internal Responses dispatch and the Anthropic outbound translator.
 * Every frame of the routed model's stream passes through live except the synthetic `advisor`
 * function call. When an iteration ends with that call, the loop consults the advisor model,
 * emits one internal `advisor_call` output item (outbound turns it into `server_tool_use` +
 * `advisor_tool_result`), and continues the routed model with the advice as the call's output.
 * An iteration that also calls a client tool ends the turn instead: the client runs its tools and
 * the advisor pair returns in history, where src/claude/inbound.ts replays it.
 */
import { decodeServerSentEvents } from "../lib/sse-decoder";
import type { TranslatorBudget } from "../lib/translator-budget";
import {
  ADVISOR_CALL_ITEM_TYPE,
  ADVISOR_SYSTEM_PROMPT,
  ADVISOR_TOOL_NAME,
  advisorOutputText,
  advisorResultContent,
  buildAdvisorTranscript,
  neutralizeAdvisorErrorText,
  type AdvisorErrorCode,
  type AdvisorToolSpec,
} from "./advisor";
import { isRec, type Rec } from "./inbound-records";

export const ADVISOR_TIMEOUT_MS = 300_000;
const HEARTBEAT_MS = 15_000;
const TERMINAL_EVENTS = new Set(["response.completed", "response.incomplete", "response.failed"]);
const ITEM_EVENTS = new Set([
  "response.output_item.added",
  "response.output_item.done",
  "response.function_call_arguments.delta",
  "response.function_call_arguments.done",
]);

export type AdvisorOutcome = { text: string } | { errorCode: AdvisorErrorCode };

export interface AdvisorLoopDeps {
  /** Translated Responses body of the first iteration. */
  body: Rec;
  spec: AdvisorToolSpec;
  /** First iteration's response: ok and text/event-stream. */
  first: Response;
  /** Resolved advisor model for the internal body, or null when it cannot be resolved. */
  advisorModel: string | null;
  /** Dispatch a continuation of the routed model through the Responses pipeline. */
  dispatchTurn: (body: Rec) => Promise<Response>;
  /** Dispatch the advisor consultation through the Responses pipeline. */
  dispatchAdvisor: (body: Rec, signal: AbortSignal) => Promise<Response>;
  translatorBudget: TranslatorBudget;
  signal?: AbortSignal;
  advisorTimeoutMs?: number;
  heartbeatMs?: number;
}

function frame(event: string, data: Rec): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function rawFrame(event: string | undefined, data: string): string {
  return `${event !== undefined ? `event: ${event}\n` : ""}data: ${data}\n\n`;
}

function parse(data: string): Rec | null {
  try {
    const value: unknown = JSON.parse(data);
    return isRec(value) ? value : null;
  } catch {
    return null;
  }
}

/** Replayable input form of an iteration's finished output item; other item types are not replayed. */
function replayItem(item: Rec): Rec | null {
  switch (item.type) {
    case "message": {
      const content = Array.isArray(item.content)
        ? item.content.filter((part): part is Rec => isRec(part) && part.type === "output_text" && typeof part.text === "string")
          .map(part => ({ type: "output_text", text: part.text }))
        : [];
      return content.length > 0 ? { type: "message", role: "assistant", content } : null;
    }
    case "reasoning": {
      const out: Rec = { type: "reasoning", summary: Array.isArray(item.summary) ? item.summary : [] };
      if (Array.isArray(item.content)) out.content = item.content;
      // An id without its blob is a reference a store:false destination cannot resolve.
      if (typeof item.encrypted_content === "string") {
        out.encrypted_content = item.encrypted_content;
        if (typeof item.id === "string") out.id = item.id;
      }
      return out;
    }
    case "function_call":
      if (typeof item.call_id !== "string" || typeof item.name !== "string") return null;
      return {
        type: "function_call",
        call_id: item.call_id,
        name: item.name,
        arguments: typeof item.arguments === "string" ? item.arguments : "{}",
        ...(typeof item.namespace === "string" ? { namespace: item.namespace } : {}),
      };
    default:
      return null;
  }
}

function errorCodeForStatus(status: number): AdvisorErrorCode {
  if (status === 429) return "too_many_requests";
  if (status === 413) return "prompt_too_long";
  if (status === 504) return "execution_time_exceeded";
  if (status === 503 || status === 529) return "overloaded";
  return "unavailable";
}

async function responseErrorMessage(response: Response): Promise<string> {
  let message = `upstream error (${response.status})`;
  try {
    const parsed = parse(await response.text());
    const error = parsed && isRec(parsed.error) ? parsed.error : null;
    if (error && typeof error.message === "string" && error.message.length > 0) message = error.message;
  } catch { /* keep fallback */ }
  return neutralizeAdvisorErrorText(message);
}

/** Read the advisor's answer from its Responses stream (or a defensive JSON body). */
async function readAdvisorAnswer(response: Response, budget: TranslatorBudget, signal: AbortSignal): Promise<AdvisorOutcome> {
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    return { errorCode: errorCodeForStatus(response.status) };
  }
  const contentType = response.headers.get("content-type") ?? "";
  let text = "";
  if (contentType.includes("text/event-stream") && response.body) {
    let settled = false;
    let failure: AdvisorOutcome | null = null;
    for await (const event of decodeServerSentEvents(response.body, { translatorBudget: budget, signal })) {
      if (settled) continue; // drain to EOF, as in the main loop
      if (event.event === "response.output_text.delta") {
        const data = parse(event.data);
        if (data && typeof data.delta === "string") text += data.delta;
      } else if (event.event === "response.failed") {
        const data = parse(event.data);
        const error = data && isRec(data.response) && isRec(data.response.error) ? data.response.error : {};
        failure = { errorCode: typeof error.status === "number" ? errorCodeForStatus(error.status) : "unavailable" };
        settled = true;
      } else if (event.event === "response.completed" || event.event === "response.incomplete") {
        settled = true;
      }
    }
    if (failure) return failure;
  } else {
    const json = parse(await response.text());
    const output = json && Array.isArray(json.output) ? json.output : [];
    for (const item of output) {
      if (!isRec(item) || item.type !== "message" || !Array.isArray(item.content)) continue;
      for (const part of item.content) {
        if (isRec(part) && part.type === "output_text" && typeof part.text === "string") text += part.text;
      }
    }
  }
  const trimmed = text.trim();
  return trimmed.length > 0 ? { text: trimmed } : { errorCode: "unavailable" };
}

/**
 * Run one advisor turn. Returns the stitched Responses SSE stream that the Anthropic outbound
 * translator reads in place of the first iteration's stream.
 */
export function runAdvisorLoop(deps: AdvisorLoopDeps): ReadableStream<Uint8Array> {
  const { spec, translatorBudget } = deps;
  const heartbeatMs = deps.heartbeatMs ?? HEARTBEAT_MS;
  const internalAbort = new AbortController();
  const linkAbort = () => internalAbort.abort(deps.signal?.reason);
  if (deps.signal) {
    if (deps.signal.aborted) linkAbort();
    else deps.signal.addEventListener("abort", linkAbort, { once: true });
  }
  const signal = internalAbort.signal;

  const consult = async function* (iterationBody: Rec, turnItems: readonly Rec[]): AsyncGenerator<string, AdvisorOutcome> {
    if (deps.advisorModel === null) return { errorCode: "unavailable" };
    const transcript = buildAdvisorTranscript(iterationBody, turnItems);
    translatorBudget.chargeRetained(Buffer.byteLength(transcript), { kind: "request_copies" });
    const timeout = AbortSignal.timeout(deps.advisorTimeoutMs ?? ADVISOR_TIMEOUT_MS);
    const advisorSignal = AbortSignal.any([signal, timeout]);
    const advisorBody: Rec = {
      model: deps.advisorModel,
      instructions: ADVISOR_SYSTEM_PROMPT,
      input: [{ type: "message", role: "user", content: [{ type: "input_text", text: transcript }] }],
      store: false,
      stream: true,
    };
    const pending = (async (): Promise<AdvisorOutcome> => {
      try {
        const response = await deps.dispatchAdvisor(advisorBody, advisorSignal);
        return await readAdvisorAnswer(response, translatorBudget, advisorSignal);
      } catch {
        return { errorCode: timeout.aborted ? "execution_time_exceeded" : "unavailable" };
      }
    })();
    // Keep the client stream alive while the consultation runs; outbound turns these into pings.
    for (;;) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const tick = new Promise<"tick">(resolve => { timer = setTimeout(() => resolve("tick"), heartbeatMs); });
      const settled = await Promise.race([pending, tick]);
      clearTimeout(timer);
      if (settled !== "tick") return settled;
      yield frame("response.heartbeat", { type: "response.heartbeat" });
    }
  };

  const run = async function* (): AsyncGenerator<string> {
    let current = deps.first;
    let iterationBody = deps.body;
    let advisorCalls = 0;
    let priorOutputTokens = 0;
    const hardCap = spec.maxUses + 2;
    for (let iteration = 1; ; iteration++) {
      const advisorCallIds: string[] = [];
      const suppressedIndexes = new Set<number>();
      const suppressedItemIds = new Set<string>();
      const turnItems: Rec[] = [];
      let clientToolCall = false;
      let terminal: { event: string; data: Rec } | null = null;
      if (!current.body) return;
      for await (const event of decodeServerSentEvents(current.body, { translatorBudget, signal })) {
        const name = event.event ?? "";
        // Drain to EOF after the terminal frame: cancelling the body instead would read as a
        // client cancel to the dispatch it came from.
        if (terminal) continue;
        if (iteration > 1 && (name === "response.created" || name === "response.in_progress")) continue;
        if (TERMINAL_EVENTS.has(name)) {
          terminal = { event: name, data: parse(event.data) ?? {} };
          continue;
        }
        if (ITEM_EVENTS.has(name)) {
          const data = parse(event.data) ?? {};
          const item = isRec(data.item) ? data.item : null;
          const isAdvisorItem = item?.type === "function_call" && item.name === ADVISOR_TOOL_NAME;
          if (isAdvisorItem) {
            if (typeof data.output_index === "number") suppressedIndexes.add(data.output_index);
            if (typeof item.id === "string") suppressedItemIds.add(item.id);
            if (name === "response.output_item.done") {
              const callId = typeof item.call_id === "string" ? item.call_id : `call_${crypto.randomUUID().replace(/-/g, "")}`;
              advisorCallIds.push(callId);
              turnItems.push({ type: "function_call", call_id: callId, name: ADVISOR_TOOL_NAME, arguments: "{}" });
            }
            continue;
          }
          if (!item && ((typeof data.output_index === "number" && suppressedIndexes.has(data.output_index))
            || (typeof data.item_id === "string" && suppressedItemIds.has(data.item_id)))) {
            continue;
          }
          if (item?.type === "function_call" && name === "response.output_item.added") clientToolCall = true;
          if (item && name === "response.output_item.done") {
            const replay = replayItem(item);
            if (replay) turnItems.push(replay);
          }
        }
        yield rawFrame(event.event, event.data);
      }
      if (!terminal) return; // EOF without a terminal: outbound reports the truncated stream.

      const continueTurn = advisorCallIds.length > 0 && terminal.event === "response.completed"
        && !clientToolCall && iteration < hardCap;
      if (advisorCallIds.length > 0 && terminal.event === "response.completed") {
        const outputs: Rec[] = [];
        for (const callId of advisorCallIds) {
          advisorCalls++;
          const outcome: AdvisorOutcome = advisorCalls > spec.maxUses
            ? { errorCode: "max_uses_exceeded" }
            : yield* consult(iterationBody, turnItems);
          const content = advisorResultContent(outcome);
          yield frame("response.output_item.done", {
            type: "response.output_item.done",
            item: {
              type: ADVISOR_CALL_ITEM_TYPE,
              id: `srvtoolu_${crypto.randomUUID().replace(/-/g, "")}`,
              ...("text" in outcome
                ? { status: "completed", text: outcome.text }
                : { status: "failed", error_code: outcome.errorCode }),
            },
          });
          outputs.push({ type: "function_call_output", call_id: callId, output: advisorOutputText(content) });
        }
        turnItems.push(...outputs);
      }

      const response = isRec(terminal.data.response) ? terminal.data.response : null;
      const usage = response && isRec(response.usage) ? response.usage : null;
      if (!continueTurn) {
        // Input/cache counts stay the final iteration's (they describe the context the client
        // accounts); output tokens accumulate across every iteration of the turn.
        if (usage && priorOutputTokens > 0) {
          const output = typeof usage.output_tokens === "number" ? usage.output_tokens : 0;
          usage.output_tokens = output + priorOutputTokens;
          if (typeof usage.total_tokens === "number") usage.total_tokens += priorOutputTokens;
        }
        yield frame(terminal.event, terminal.data);
        return;
      }
      if (usage && typeof usage.output_tokens === "number") priorOutputTokens += usage.output_tokens;

      const input = Array.isArray(iterationBody.input) ? iterationBody.input : [];
      const next: Rec = { ...iterationBody, input: [...input, ...turnItems] };
      if (advisorCalls >= spec.maxUses && Array.isArray(next.tools)) {
        // The budget is spent: the model can no longer ask, so it must continue on its own.
        next.tools = next.tools.filter(tool => !(isRec(tool) && tool.type === "function" && tool.name === ADVISOR_TOOL_NAME));
      }
      if (isRec(next.tool_choice) && next.tool_choice.name === ADVISOR_TOOL_NAME) next.tool_choice = "auto";
      translatorBudget.chargeRetained(Buffer.byteLength(JSON.stringify(turnItems)), { kind: "request_copies" });
      current = await deps.dispatchTurn(next);
      const contentType = current.headers.get("content-type") ?? "";
      if (!current.ok || !contentType.includes("text/event-stream") || !current.body) {
        const message = current.ok ? "advisor continuation returned no stream" : await responseErrorMessage(current);
        yield frame("response.failed", {
          type: "response.failed",
          response: { status: "failed", error: { message, status: current.ok ? 502 : current.status } },
        });
        return;
      }
      iterationBody = next;
    }
  };

  const encoder = new TextEncoder();
  const frames = run();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await frames.next();
        if (done) {
          deps.signal?.removeEventListener("abort", linkAbort);
          controller.close();
        } else {
          controller.enqueue(encoder.encode(value));
        }
      } catch (error) {
        deps.signal?.removeEventListener("abort", linkAbort);
        controller.error(error);
      }
    },
    async cancel(reason) {
      internalAbort.abort(reason);
      deps.signal?.removeEventListener("abort", linkAbort);
      await frames.return(undefined).catch(() => {});
    },
  });
}
