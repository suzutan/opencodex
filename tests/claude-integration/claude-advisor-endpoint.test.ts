import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { saveConfig } from "../../src/config";
import { handleClaudeMessages } from "../../src/server/claude-messages";
import { tryAdmitTurn } from "../../src/server/lifecycle";
import { acquireOwnedSpendHome } from "../helpers/owned-spend-home";
import { clearRequestLogsForTests, getRequestLogEntries, type RequestLogContext } from "../../src/server/request-log";
import type { OcxConfig } from "../../src/types";
import { installIsolatedCodexHome, type IsolatedCodexHome } from "../helpers/isolated-codex-home";
import { removeTreeWithRetry } from "../helpers/remove-tree";

/**
 * End to end through the real translation, routing and chat adapter: proves the synthetic
 * `advisor` function survives the bridge under its own name, that an OpenCodex alias as the
 * advisor model routes like a client-selected model, and that the continuation carries the
 * advice back to the routed model.
 */

type Rec = Record<string, unknown>;

let testDir = "";
let previousHome: string | undefined;
let isolatedCodexHome: IsolatedCodexHome | null = null;

beforeEach(() => {
  previousHome = process.env.OPENCODEX_HOME;
  isolatedCodexHome = installIsolatedCodexHome("ocx-claude-advisor-");
  testDir = mkdtempSync(join(tmpdir(), "ocx-claude-advisor-"));
  process.env.OPENCODEX_HOME = testDir;
});

afterEach(() => {
  if (previousHome === undefined) delete process.env.OPENCODEX_HOME;
  else process.env.OPENCODEX_HOME = previousHome;
  isolatedCodexHome?.restore();
  isolatedCodexHome = null;
  if (testDir) removeTreeWithRetry(testDir);
});

function chatFrames(delta: Rec, finish: string): string {
  return [
    `data: ${JSON.stringify({ choices: [{ index: 0, delta: { role: "assistant", ...delta } }] })}\n\n`,
    `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: finish }], usage: { prompt_tokens: 10, completion_tokens: 2 } })}\n\n`,
    "data: [DONE]\n\n",
  ].join("");
}

function mockUpstream() {
  const captured: Rec[] = [];
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      const body = await req.json() as Rec;
      captured.push(body);
      const messages = Array.isArray(body.messages) ? body.messages as Rec[] : [];
      let frames: string;
      if (body.model !== "main") {
        frames = chatFrames({ content: "Check the edge case first." }, "stop");
      } else if (messages.some(message => message.role === "tool")) {
        frames = chatFrames({ content: "Following the advice." }, "stop");
      } else {
        frames = chatFrames({
          tool_calls: [{ index: 0, id: "call_adv", type: "function", function: { name: "advisor", arguments: "{}" } }],
        }, "tool_calls");
      }
      return new Response(frames, { headers: { "Content-Type": "text/event-stream" } });
    },
  });
  return { server, captured };
}

test("a routed turn consults an aliased advisor model and continues with its advice", async () => {
  const upstream = mockUpstream();
  const config = {
    port: 0,
    defaultProvider: "mock",
    providers: {
      mock: { adapter: "openai-chat", baseUrl: `${upstream.server.url.toString().replace(/\/$/, "")}/v1`, apiKey: "k", allowPrivateNetwork: true },
    },
  } as OcxConfig;
  saveConfig(config);
  try {
    const response = await handleClaudeMessages(
      new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: { "content-type": "application/json", "x-api-key": "placeholder", "anthropic-version": "2023-06-01" },
        body: JSON.stringify({
          model: "mock/main",
          max_tokens: 256,
          stream: false,
          system: "You are the executor.",
          tools: [{ type: "advisor_20260301", name: "advisor", model: "claude-ocx-mock--advisor" }],
          messages: [{ role: "user", content: "Fix the parser." }],
        }),
      }),
      config,
      { model: "unknown", provider: "unknown", inboundProtocol: "messages" } as RequestLogContext,
    );
    expect(response.status).toBe(200);
    const message = await response.json() as Rec;
    const content = message.content as Rec[];
    expect(content.map(block => block.type)).toEqual(["server_tool_use", "advisor_tool_result", "text"]);
    expect(content[1]!.content).toEqual({ type: "advisor_result", text: "Check the edge case first." });
    expect(content[2]!.text).toBe("Following the advice.");
    expect(message.stop_reason).toBe("end_turn");

    expect(upstream.captured.map(body => body.model)).toEqual(["main", "advisor", "main"]);
    const [first, advisor, continuation] = upstream.captured as [Rec, Rec, Rec];
    expect((first.tools as Rec[]).map(tool => (tool.function as Rec).name)).toEqual(["advisor"]);
    expect(advisor.tools).toBeUndefined();
    expect(JSON.stringify(advisor.messages)).toContain("stronger reviewer model");
    expect(JSON.stringify(advisor.messages)).toContain("Fix the parser.");
    const toolMessage = (continuation.messages as Rec[]).find(message => message.role === "tool");
    expect(toolMessage).toMatchObject({ tool_call_id: "call_adv", content: "Check the edge case first." });
  } finally {
    upstream.server.stop(true);
  }
});

function responsesFrames(items: Rec[], inputTokens = 10): string {
  const frames = [`event: response.created\ndata: ${JSON.stringify({ type: "response.created", response: { id: "resp_1", status: "in_progress" } })}\n\n`];
  items.forEach((item, index) => {
    frames.push(`event: response.output_item.added\ndata: ${JSON.stringify({ type: "response.output_item.added", output_index: index, item })}\n\n`);
    if (item.type === "message") {
      const text = ((item.content as Rec[])[0] as Rec).text;
      frames.push(`event: response.output_text.delta\ndata: ${JSON.stringify({ type: "response.output_text.delta", item_id: item.id, output_index: index, content_index: 0, delta: text })}\n\n`);
    }
    frames.push(`event: response.output_item.done\ndata: ${JSON.stringify({ type: "response.output_item.done", output_index: index, item })}\n\n`);
  });
  frames.push(`event: response.completed\ndata: ${JSON.stringify({ type: "response.completed", response: { id: "resp_1", status: "completed", output: items, usage: { input_tokens: inputTokens, output_tokens: 2, total_tokens: inputTokens + 2 } } })}\n\n`);
  return frames.join("");
}

test("on a Responses-wire route each advisor dispatch is admitted as its own turn", async () => {
  const captured: Rec[] = [];
  const upstream = Bun.serve({
    port: 0,
    async fetch(req) {
      const body = await req.json() as Rec;
      captured.push(body);
      const input = Array.isArray(body.input) ? body.input as Rec[] : [];
      const message = (id: string, text: string) => ({ type: "message", id, role: "assistant", status: "completed", content: [{ type: "output_text", text, annotations: [] }] });
      const items = body.model === "advisor"
        ? [message("msg_adv", "Check the edge case first.")]
        : input.some(item => item.type === "function_call_output")
          ? [message("msg_final", "Following the advice.")]
          : [{ type: "function_call", id: "fc_1", call_id: "call_adv", name: "advisor", arguments: "{}", status: "completed" }];
      // Distinct per-dispatch input sizes make any cross-row usage accumulation visible.
      const inputTokens = body.model === "advisor" ? 300 : input.some(item => item.type === "function_call_output") ? 1100 : 1000;
      return new Response(responsesFrames(items, inputTokens), { headers: { "Content-Type": "text/event-stream" } });
    },
  });
  const config = {
    port: 0,
    defaultProvider: "gw",
    providers: {
      gw: { adapter: "openai-responses", baseUrl: `${upstream.url.toString().replace(/\/$/, "")}/v1`, apiKey: "k", allowPrivateNetwork: true },
    },
  } as OcxConfig;
  saveConfig(config);
  // Admitted like the HTTP route admits it. The first iteration's stream settles this lease, so
  // a later dispatch that reused it would be aborted with "turn already settled".
  const turnAdmissionLease = tryAdmitTurn();
  if (!turnAdmissionLease) throw new Error("test turn admission unavailable");
  const releaseSpendHome = acquireOwnedSpendHome();
  clearRequestLogsForTests();
  try {
    const response = await handleClaudeMessages(
      new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: { "content-type": "application/json", "x-api-key": "placeholder", "anthropic-version": "2023-06-01" },
        body: JSON.stringify({
          model: "gw/main",
          max_tokens: 256,
          stream: false,
          tools: [{ type: "advisor_20260301", name: "advisor", model: "claude-ocx-gw--advisor" }],
          messages: [{ role: "user", content: "Fix the parser." }],
        }),
      }),
      config,
      { model: "unknown", provider: "unknown", inboundProtocol: "messages" } as RequestLogContext,
      { requestId: "claude-advisor-responses", start: Date.now(), turnAdmissionLease },
    );
    const message = await response.json() as Rec;
    expect(response.status).toBe(200);
    expect((message.content as Rec[]).map(block => block.type)).toEqual(["server_tool_use", "advisor_tool_result", "text"]);
    expect((message.content as Rec[])[1]!.content).toEqual({ type: "advisor_result", text: "Check the edge case first." });
    expect(captured.map(body => body.model)).toEqual(["main", "advisor", "main"]);
    expect((captured[2]!.input as Rec[]).at(-1)).toMatchObject({ type: "function_call_output", call_id: "call_adv", output: "Check the edge case first." });
    // The continuation is the first iteration's input plus the call and its output, nothing more.
    expect(captured[2]!.input).toEqual([...(captured[0]!.input as Rec[]), (captured[2]!.input as Rec[]).at(-2), (captured[2]!.input as Rec[]).at(-1)]);
    // The turn, the consultation and the continuation each keep a request-log row, including on
    // this native Responses forward, which logs through terminal callbacks instead of the tap.
    const rows = getRequestLogEntries().filter(entry => entry.surface === "claude");
    expect(rows.map(entry => entry.model).sort()).toEqual(["advisor", "main", "main"]);
    // Each row carries only its own dispatch's usage: the continuation is not the sum of the turn.
    expect(rows.map(entry => [entry.model, entry.usage?.inputTokens, entry.usage?.outputTokens])).toEqual([
      ["main", 1000, 2], ["advisor", 300, 2], ["main", 1100, 2],
    ]);
  } finally {
    releaseSpendHome();
    turnAdmissionLease.release();
    upstream.stop(true);
  }
});

test("a Claude main model on native passthrough forwards the advisor tool untouched", async () => {
  const captured: Rec[] = [];
  const anthropic = Bun.serve({
    port: 0,
    async fetch(req) {
      captured.push(await req.json() as Rec);
      return Response.json({
        id: "msg_up", type: "message", role: "assistant", model: "claude-fable-5",
        content: [{ type: "text", text: "native" }], stop_reason: "end_turn", stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1 },
      });
    },
  });
  const config = {
    port: 0,
    defaultProvider: "mock",
    providers: { mock: { adapter: "openai-chat", baseUrl: "http://127.0.0.1:1/v1", apiKey: "k", allowPrivateNetwork: true } },
    claudeCode: { anthropicBaseUrl: anthropic.url.toString().replace(/\/$/, "") },
  } as OcxConfig;
  saveConfig(config);
  const body = {
    model: "claude-fable-5",
    max_tokens: 64,
    stream: false,
    tools: [{ type: "advisor_20260301", name: "advisor", model: "claude-opus-5-5" }],
    messages: [{ role: "user", content: "hi" }],
  };
  try {
    const response = await handleClaudeMessages(
      new Request("http://localhost/v1/messages", {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "anthropic-version": "2023-06-01",
          "anthropic-beta": "advisor-tool-2026-03-01",
          authorization: "Bearer sk-ant-oat01-tst",
        },
        body: JSON.stringify(body),
      }),
      config,
      { model: "unknown", provider: "unknown", inboundProtocol: "messages" } as RequestLogContext,
    );
    expect(response.status).toBe(200);
    await response.text();
    expect(captured).toEqual([body]);
  } finally {
    anthropic.stop(true);
  }
});

interface NativeCapture { url: string; headers: Headers; body: Rec }

function mockAnthropic(reply: () => Response) {
  const captured: NativeCapture[] = [];
  const server = Bun.serve({
    port: 0,
    async fetch(req) {
      captured.push({ url: req.url, headers: req.headers, body: await req.json() as Rec });
      return reply();
    },
  });
  return { server, captured };
}

const anthropicAdvice = () => Response.json({
  id: "msg_adv", type: "message", role: "assistant", model: "claude-opus-5-5",
  content: [{ type: "text", text: "Native: check the edge case first." }],
  stop_reason: "end_turn", stop_sequence: null, usage: { input_tokens: 30, output_tokens: 8 },
});

const SUBSCRIPTION_HEADERS = {
  "content-type": "application/json",
  "anthropic-version": "2023-06-01",
  "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,advisor-tool-2026-03-01",
  authorization: "Bearer sk-ant-oat01-tst",
};

async function runNativeCase(opts: {
  advisorModel: string;
  headers: Record<string, string>;
  claudeCode?: Record<string, unknown>;
  anthropicReply?: () => Response;
}) {
  const upstream = mockUpstream();
  const anthropic = mockAnthropic(opts.anthropicReply ?? anthropicAdvice);
  const config = {
    port: 0,
    defaultProvider: "mock",
    providers: {
      mock: { adapter: "openai-chat", baseUrl: `${upstream.server.url.toString().replace(/\/$/, "")}/v1`, apiKey: "k", allowPrivateNetwork: true },
    },
    claudeCode: { anthropicBaseUrl: anthropic.server.url.toString().replace(/\/$/, ""), ...opts.claudeCode },
  } as OcxConfig;
  saveConfig(config);
  const turnAdmissionLease = tryAdmitTurn();
  if (!turnAdmissionLease) throw new Error("test turn admission unavailable");
  const releaseSpendHome = acquireOwnedSpendHome();
  clearRequestLogsForTests();
  try {
    const response = await handleClaudeMessages(
      new Request("http://localhost/v1/messages?beta=true", {
        method: "POST",
        headers: opts.headers,
        body: JSON.stringify({
          model: "mock/main",
          max_tokens: 256,
          stream: false,
          tools: [{ type: "advisor_20260301", name: "advisor", model: opts.advisorModel }],
          messages: [{ role: "user", content: "Fix the parser." }],
        }),
      }),
      config,
      { model: "unknown", provider: "unknown", inboundProtocol: "messages" } as RequestLogContext,
      { requestId: "claude-advisor-native", start: Date.now(), turnAdmissionLease },
    );
    const message = await response.json() as Rec;
    // The turn's own row is finalized when its response body closes, after the JSON is read.
    const claudeRows = () => getRequestLogEntries().filter(entry => entry.surface === "claude");
    for (let tick = 0; tick < 100 && claudeRows().length < 3; tick++) await Bun.sleep(10);
    const rows = claudeRows().map(entry => [entry.provider, entry.model, entry.usage?.inputTokens]).sort();
    return { status: response.status, message, routed: upstream.captured, native: anthropic.captured, rows };
  } finally {
    releaseSpendHome();
    turnAdmissionLease.release();
    upstream.server.stop(true);
    anthropic.server.stop(true);
  }
}

test("a Claude advisor on the caller's sk-ant credential goes straight to Anthropic", async () => {
  const result = await runNativeCase({ advisorModel: "claude-opus-5-5[1m]", headers: SUBSCRIPTION_HEADERS });
  expect(result.status).toBe(200);
  const content = result.message.content as Rec[];
  expect(content[1]!.content).toEqual({ type: "advisor_result", text: "Native: check the edge case first." });
  expect(result.routed.map(body => body.model)).toEqual(["main", "main"]);
  expect(result.native).toHaveLength(1);
  const hit = result.native[0]!;
  expect(new URL(hit.url).pathname + new URL(hit.url).search).toBe("/v1/messages?beta=true");
  expect(hit.headers.get("authorization")).toBe("Bearer sk-ant-oat01-tst");
  expect(hit.headers.get("anthropic-beta")).toBe("claude-code-20250219,oauth-2025-04-20");
  expect(hit.body.model).toBe("claude-opus-5-5");
  expect(hit.body.tools).toBeUndefined();
  expect(hit.body.stream).toBe(false);
  expect(JSON.stringify(hit.body.system)).toContain("stronger reviewer model");
  expect(JSON.stringify(hit.body.messages)).toContain("Fix the parser.");
  // One row per dispatch, each with only its own usage.
  expect(result.rows).toEqual([
    ["anthropic-native", "claude-opus-5-5", 30],
    ["mock", "main", 10],
    ["mock", "main", 10],
  ]);
});

test("an Anthropic error on the native advisor becomes the error variant with neutralized text", async () => {
  const result = await runNativeCase({
    advisorModel: "claude-opus-5-5",
    headers: SUBSCRIPTION_HEADERS,
    anthropicReply: () => Response.json(
      { type: "error", error: { type: "rate_limit_error", message: "The advisor tool is not available" } },
      { status: 429 },
    ),
  });
  expect(result.status).toBe(200);
  expect((result.message.content as Rec[])[1]!.content).toEqual({ type: "advisor_tool_result_error", error_code: "too_many_requests" });
  expect(JSON.stringify(result.message)).not.toContain("advisor tool is not available");
});

test("a Claude advisor id claimed by modelMap stays on the routed path", async () => {
  const result = await runNativeCase({
    advisorModel: "claude-opus-5-5",
    headers: SUBSCRIPTION_HEADERS,
    claudeCode: { modelMap: { "claude-opus-5-5": "mock/advisor" } },
  });
  expect(result.native).toHaveLength(0);
  expect(result.routed.map(body => body.model)).toEqual(["main", "advisor", "main"]);
});

test("a Claude advisor without an sk-ant caller credential stays on the routed path", async () => {
  const result = await runNativeCase({
    advisorModel: "claude-opus-5-5",
    headers: { "content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "placeholder" },
  });
  expect(result.native).toHaveLength(0);
  expect(result.routed.map(body => body.model)).toEqual(["main", "claude-opus-5-5", "main"]);
});
