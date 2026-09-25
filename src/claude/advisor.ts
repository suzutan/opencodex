/**
 * Claude Code advisor emulation for translated (routed) Messages turns.
 *
 * Claude Code attaches the server-side `advisor_*` tool (beta `advisor-tool-2026-03-01`) and
 * expects the server to consult a stronger model mid-turn. A routed model has no such server
 * tool, so the translated request exposes a synthetic `advisor` function instead, the proxy
 * runs the consultation itself (src/claude/advisor-loop.ts), and the result reaches Claude Code
 * as the `server_tool_use` + `advisor_tool_result` pair it already parses. Native Anthropic
 * passthrough never reaches this module.
 */
import { isRec, type Rec } from "./inbound-records";

/** Function name the routed model sees and the loop intercepts. */
export const ADVISOR_TOOL_NAME = "advisor";

/** `max_uses` fallback when the client does not bound the advisor itself. */
export const DEFAULT_ADVISOR_MAX_USES = 3;

/** Error codes Claude Code 2.1.280 recognizes on `advisor_tool_result_error`. */
export const ADVISOR_ERROR_CODES = [
  "max_uses_exceeded",
  "too_many_requests",
  "overloaded",
  "prompt_too_long",
  "execution_time_exceeded",
  "unavailable",
] as const;
export type AdvisorErrorCode = typeof ADVISOR_ERROR_CODES[number];

/** Internal Responses output item the loop emits; outbound turns it into the Anthropic pair. */
export const ADVISOR_CALL_ITEM_TYPE = "advisor_call";

export interface AdvisorToolSpec {
  /** Advisor model exactly as the client named it (a Claude id or an OpenCodex alias). */
  model: string;
  maxUses: number;
}

export const ADVISOR_SYSTEM_PROMPT =
  "You are a stronger reviewer model advising another model that is in the middle of a task. "
  + "The transcript below is everything that model has seen and done so far: its instructions, the "
  + "user's requests, its tool calls, and their results. Read the whole transcript and give concise, "
  + "concrete advice for the next steps: point out mistakes, wrong assumptions, missing checks, and a "
  + "better approach when there is one. Reply in plain text addressed to that model. You cannot call "
  + "tools and must not pretend to.";

const ADVISOR_TOOL_DESCRIPTION =
  "Consult a stronger reviewer model. Takes no parameters: the whole conversation so far is "
  + "forwarded automatically, and the tool returns the reviewer's advice as text.";

export function isAdvisorToolType(type: unknown): boolean {
  return typeof type === "string" && /^advisor_\d{8}$/.test(type);
}

/** The first advisor server tool in an Anthropic `tools` array, or null. */
export function advisorToolSpec(tools: unknown): AdvisorToolSpec | null {
  if (!Array.isArray(tools)) return null;
  for (const tool of tools) {
    if (!isRec(tool) || !isAdvisorToolType(tool.type)) continue;
    if (typeof tool.model !== "string" || tool.model.trim().length === 0) return null;
    const maxUses = typeof tool.max_uses === "number" && Number.isInteger(tool.max_uses) && tool.max_uses > 0
      ? tool.max_uses
      : DEFAULT_ADVISOR_MAX_USES;
    return { model: tool.model.trim(), maxUses };
  }
  return null;
}

/** The synthetic Responses function tool that stands in for the advisor server tool. */
export function advisorFunctionTool(): Rec {
  return {
    type: "function",
    name: ADVISOR_TOOL_NAME,
    description: ADVISOR_TOOL_DESCRIPTION,
    parameters: { type: "object", properties: {}, additionalProperties: false },
    strict: false,
  };
}

/** Text the routed model receives as the advisor call's function output. */
export function advisorOutputText(content: unknown): string {
  if (isRec(content)) {
    if (content.type === "advisor_result" && typeof content.text === "string") return content.text;
    if (content.type === "advisor_redacted_result") return "[advisor result redacted]";
    if (content.type === "advisor_tool_result_error") {
      const code = typeof content.error_code === "string" ? content.error_code : "unavailable";
      return `The advisor was unavailable (${code}). Continue without its advice.`;
    }
  }
  return "The advisor returned no advice. Continue without it.";
}

/** Anthropic content for one advisor outcome: advice text, or the error variant. */
export function advisorResultContent(outcome: { text: string } | { errorCode: AdvisorErrorCode }): Rec {
  return "text" in outcome
    ? { type: "advisor_result", text: outcome.text }
    : { type: "advisor_tool_result_error", error_code: outcome.errorCode };
}

const DISABLING_ERROR_TEXT = /advisor tool is not available|cannot be used as an advisor|Input tag 'advisor_/i;

/**
 * Claude Code turns the advisor off for the whole process when an API error message contains one
 * of these phrases. Nothing this proxy relays on an advisor turn may carry them.
 */
export function neutralizeAdvisorErrorText(message: string): string {
  return DISABLING_ERROR_TEXT.test(message) ? "upstream request failed" : message;
}

const TRANSCRIPT_MAX_CHARS = 400_000;
const TRANSCRIPT_INSTRUCTIONS_MAX_CHARS = 100_000;
const TRANSCRIPT_ENTRY_MAX_CHARS = 20_000;

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}\n[... ${text.length - max} characters truncated]`;
}

function partsText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const out: string[] = [];
  for (const part of content) {
    if (!isRec(part)) continue;
    if ((part.type === "input_text" || part.type === "output_text" || part.type === "text"
      || part.type === "summary_text" || part.type === "reasoning_text") && typeof part.text === "string") {
      out.push(part.text);
    } else if (part.type === "input_image") {
      out.push("[image]");
    } else if (part.type === "input_file") {
      out.push(`[file${typeof part.filename === "string" ? `: ${part.filename}` : ""}]`);
    }
  }
  return out.join("\n");
}

function renderItem(item: Rec): string | null {
  switch (item.type) {
    case "message": {
      const text = partsText(item.content);
      return text.length > 0 ? `[${typeof item.role === "string" ? item.role : "user"}]\n${text}` : null;
    }
    case "function_call": {
      const name = typeof item.name === "string" ? item.name : "tool";
      const args = typeof item.arguments === "string" ? item.arguments : "";
      return `[assistant tool call: ${name} id=${String(item.call_id ?? "")}]\n${args}`;
    }
    case "function_call_output":
      return `[tool result id=${String(item.call_id ?? "")}]\n${partsText(item.output)}`;
    case "reasoning": {
      const text = [partsText(item.summary), partsText(item.content)].filter(t => t.length > 0).join("\n");
      return text.length > 0 ? `[assistant reasoning]\n${text}` : null;
    }
    default:
      return null;
  }
}

function toolNames(tools: unknown): string[] {
  if (!Array.isArray(tools)) return [];
  const names: string[] = [];
  for (const tool of tools) {
    if (!isRec(tool)) continue;
    if (typeof tool.name === "string" && tool.name !== ADVISOR_TOOL_NAME) names.push(tool.name);
    else if (typeof tool.type === "string" && tool.type !== "function") names.push(tool.type);
  }
  return names;
}

/**
 * Render the routed turn as plain text for the advisor. A transcript is used instead of replaying
 * structured items because the advisor may be any model: tool calls without declarations and
 * reasoning signatures minted for another provider are not portable, while text is. Images and
 * files become placeholders. When the transcript exceeds its bound, the oldest items are dropped
 * and the latest context is kept.
 */
export function buildAdvisorTranscript(body: Rec, extraItems: readonly Rec[] = []): string {
  const header: string[] = [];
  if (typeof body.instructions === "string" && body.instructions.length > 0) {
    header.push(`[system instructions of the agent]\n${clip(body.instructions, TRANSCRIPT_INSTRUCTIONS_MAX_CHARS)}`);
  }
  const names = toolNames(body.tools);
  if (names.length > 0) header.push(`[tools available to the agent]\n${names.join(", ")}`);
  const items = [...(Array.isArray(body.input) ? body.input : []), ...extraItems];
  const rendered: string[] = [];
  for (const item of items) {
    if (!isRec(item)) continue;
    const text = renderItem(item);
    if (text !== null) rendered.push(clip(text, TRANSCRIPT_ENTRY_MAX_CHARS));
  }
  let budget = TRANSCRIPT_MAX_CHARS - header.reduce((sum, part) => sum + part.length + 2, 0);
  const kept: string[] = [];
  for (let index = rendered.length - 1; index >= 0; index--) {
    const entry = rendered[index]!;
    if (entry.length + 2 > budget) {
      kept.unshift(`[... ${index + 1} earlier transcript entries omitted ...]`);
      break;
    }
    budget -= entry.length + 2;
    kept.unshift(entry);
  }
  return [
    ...header,
    "[transcript]",
    ...kept,
    "[end of transcript] The agent has just called the advisor. Give your advice now.",
  ].join("\n\n");
}

/** Anthropic `server_tool_use` id + `advisor_tool_result` content for an internal advisor_call item. */
export function advisorPairFromItem(item: Rec): { id: string; resultContent: Rec } {
  const id = typeof item.id === "string" && item.id.length > 0 ? item.id : `srvtoolu_${crypto.randomUUID().replace(/-/g, "")}`;
  if (item.status === "completed" && typeof item.text === "string" && item.text.length > 0) {
    return { id, resultContent: advisorResultContent({ text: item.text }) };
  }
  const code = ADVISOR_ERROR_CODES.find(candidate => candidate === item.error_code) ?? "unavailable";
  return { id, resultContent: advisorResultContent({ errorCode: code }) };
}

/** Output budget for an advisor consultation sent straight to Anthropic. */
export const NATIVE_ADVISOR_MAX_TOKENS = 8192;

/**
 * A subscription (claude.ai OAuth) credential is accepted only for requests that identify as
 * Claude Code, which every request Claude Code itself sends does in its first system block.
 */
const CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude.";

/**
 * The Anthropic Messages form of an advisor consultation: the same instruction and transcript as
 * the routed Responses body, with no tools.
 */
export function nativeAdvisorMessagesBody(model: string, advisorBody: Rec): Rec {
  const texts: string[] = [];
  for (const item of Array.isArray(advisorBody.input) ? advisorBody.input : []) {
    if (isRec(item) && item.type === "message") texts.push(partsText(item.content));
  }
  return {
    model,
    max_tokens: NATIVE_ADVISOR_MAX_TOKENS,
    stream: false,
    system: [
      { type: "text", text: CLAUDE_CODE_IDENTITY },
      { type: "text", text: typeof advisorBody.instructions === "string" ? advisorBody.instructions : ADVISOR_SYSTEM_PROMPT },
    ],
    messages: [{ role: "user", content: [{ type: "text", text: texts.join("\n\n") }] }],
  };
}

/** Drop the advisor beta from a forwarded `anthropic-beta` list; the consultation has no advisor tool. */
export function withoutAdvisorBeta(value: string): string {
  return value.split(",").map(entry => entry.trim()).filter(entry => entry.length > 0 && !entry.startsWith("advisor-tool-")).join(",");
}

/**
 * Adapt an Anthropic Messages reply into the Responses JSON shape the advisor loop reads.
 * Errors keep their status (a "prompt is too long" 400 becomes 413) and a neutralized message.
 */
export async function nativeAdvisorResponse(response: Response): Promise<Response> {
  const json = (response.headers.get("content-type") ?? "").includes("json")
    ? await response.json().catch(() => null) as unknown
    : (await response.body?.cancel().catch(() => {}), null);
  if (!response.ok) {
    const error = isRec(json) && isRec(json.error) ? json.error : {};
    const message = typeof error.message === "string" ? error.message : `anthropic error (${response.status})`;
    const status = response.status === 400 && /prompt is too long/i.test(message) ? 413 : response.status;
    return new Response(JSON.stringify({ error: { message: neutralizeAdvisorErrorText(message) } }), {
      status,
      headers: { "content-type": "application/json" },
    });
  }
  const content = isRec(json) && Array.isArray(json.content) ? json.content : [];
  const text = content
    .filter((block): block is Rec => isRec(block) && block.type === "text" && typeof block.text === "string")
    .map(block => block.text as string)
    .join("");
  return new Response(JSON.stringify({
    output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text }] }],
  }), { status: 200, headers: { "content-type": "application/json" } });
}
