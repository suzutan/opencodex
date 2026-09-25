import { describe, expect, test } from "bun:test";
import {
  ADVISOR_TOOL_NAME,
  advisorToolSpec,
  buildAdvisorTranscript,
  DEFAULT_ADVISOR_MAX_USES,
  nativeAdvisorMessagesBody,
  nativeAdvisorResponse,
  neutralizeAdvisorErrorText,
  withoutAdvisorBeta,
} from "../../src/claude/advisor";
import { anthropicToResponsesTranslation } from "../../src/claude/inbound";
import { analyzeClaudeCompatibility } from "../../src/claude/compatibility";
import {
  collectAnthropicMessage,
  responsesJsonToAnthropicMessage,
  responsesSseToAnthropicSse,
} from "../../src/claude/outbound";
import { createTestTranslatorBudget } from "../helpers/translator-budget";

type Rec = Record<string, unknown>;

const ADVISOR_TOOL = { type: "advisor_20260301", name: "advisor", model: "claude-opus-5-5" };

function translate(body: Rec) {
  return anthropicToResponsesTranslation({ model: "mock/m", max_tokens: 64, ...body }, undefined, createTestTranslatorBudget());
}

function sse(name: string, data: Rec): string {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

function streamFrom(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({ start(controller) { controller.enqueue(bytes); controller.close(); } });
}

describe("advisor inbound mapping", () => {
  test("advisor server tool becomes a parameterless synthetic function and its spec leaves the body", () => {
    const { body, advisor } = translate({
      tools: [ADVISOR_TOOL, { name: "Read", description: "read", input_schema: { type: "object", properties: {} } }],
      messages: [{ role: "user", content: "hi" }],
    });
    const tools = body.tools as Rec[];
    expect(tools.map(t => t.name)).toEqual([ADVISOR_TOOL_NAME, "Read"]);
    expect(tools[0]).toMatchObject({
      type: "function",
      name: "advisor",
      parameters: { type: "object", properties: {}, additionalProperties: false },
      strict: false,
    });
    expect(advisor).toEqual({ model: "claude-opus-5-5", maxUses: DEFAULT_ADVISOR_MAX_USES });
    expect(JSON.stringify(body)).not.toContain("advisor_20260301");
    expect(JSON.stringify(body)).not.toContain("claude-opus-5-5");
  });

  test("max_uses is honoured and a model-less advisor tool yields no spec", () => {
    expect(advisorToolSpec([{ ...ADVISOR_TOOL, max_uses: 1 }])).toEqual({ model: "claude-opus-5-5", maxUses: 1 });
    expect(advisorToolSpec([{ type: "advisor_20260301", name: "advisor" }])).toBeNull();
    expect(translate({ messages: [{ role: "user", content: "hi" }] }).advisor).toBeUndefined();
  });

  test("forcing the advisor tool maps to a function tool choice", () => {
    const { body } = translate({
      tools: [ADVISOR_TOOL],
      tool_choice: { type: "tool", name: "advisor" },
      messages: [{ role: "user", content: "hi" }],
    });
    expect(body.tool_choice).toEqual({ type: "function", name: "advisor" });
  });

  test("history pair replays as a function call and its output", () => {
    const { body } = translate({
      tools: [ADVISOR_TOOL],
      messages: [
        { role: "user", content: "fix the bug" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me ask." },
            { type: "server_tool_use", id: "srvtoolu_a", name: "advisor", input: {} },
            { type: "advisor_tool_result", tool_use_id: "srvtoolu_a", content: { type: "advisor_result", text: "Check the null path." } },
            { type: "server_tool_use", id: "srvtoolu_b", name: "advisor", input: {} },
            { type: "advisor_tool_result", tool_use_id: "srvtoolu_b", content: { type: "advisor_tool_result_error", error_code: "overloaded" } },
            { type: "text", text: "Done." },
          ],
        },
        { role: "user", content: "thanks" },
      ],
    });
    const input = body.input as Rec[];
    expect(input.map(item => item.type)).toEqual([
      "message", "message", "function_call", "function_call_output", "function_call", "function_call_output", "message", "message",
    ]);
    expect(input[2]).toEqual({ type: "function_call", call_id: "srvtoolu_a", name: "advisor", arguments: "{}" });
    expect(input[3]).toEqual({ type: "function_call_output", call_id: "srvtoolu_a", output: "Check the null path." });
    expect(input[5]!.output).toContain("overloaded");
  });

  test("an unpaired advisor block is dropped rather than replayed as an unanswered call", () => {
    const { body } = translate({
      messages: [
        { role: "user", content: "q" },
        {
          role: "assistant",
          content: [
            { type: "server_tool_use", id: "srvtoolu_x", name: "advisor", input: {} },
            { type: "advisor_tool_result", tool_use_id: "srvtoolu_other", content: { type: "advisor_result", text: "t" } },
            { type: "text", text: "answer" },
          ],
        },
      ],
    });
    expect((body.input as Rec[]).map(item => item.type)).toEqual(["message", "message"]);
  });

  test("compatibility tolerates the emulated advisor in tools and history", () => {
    const result = analyzeClaudeCompatibility({
      model: "m",
      messages: [
        { role: "assistant", content: [
          { type: "server_tool_use", id: "s", name: "advisor", input: {} },
          { type: "advisor_tool_result", tool_use_id: "s", content: { type: "advisor_result", text: "x" } },
        ] },
      ],
      tools: [ADVISOR_TOOL],
    }, { mode: "enforce" });
    expect(result.featureCodes).toEqual(["advisor_tool"]);
    expect(result.decision).toBe("allow");
  });
});

describe("advisor transcript", () => {
  test("renders instructions, tool names, calls and results, and keeps the tail when over budget", () => {
    const transcript = buildAdvisorTranscript({
      instructions: "SYSTEM RULES",
      tools: [{ type: "function", name: "Read" }, { type: "function", name: "advisor" }, { type: "web_search" }],
      input: [
        { type: "message", role: "user", content: [{ type: "input_text", text: "please fix" }, { type: "input_image", image_url: "data:x" }] },
        { type: "function_call", call_id: "c1", name: "Read", arguments: "{\"path\":\"a.ts\"}" },
        { type: "function_call_output", call_id: "c1", output: "file body" },
      ],
    }, [{ type: "reasoning", summary: [{ type: "summary_text", text: "thinking out loud" }] }]);
    expect(transcript).toContain("SYSTEM RULES");
    expect(transcript).toContain("Read, web_search");
    expect(transcript).not.toContain("Read, advisor");
    expect(transcript).toContain("please fix\n[image]");
    expect(transcript).toContain("[assistant tool call: Read id=c1]\n{\"path\":\"a.ts\"}");
    expect(transcript).toContain("[tool result id=c1]\nfile body");
    expect(transcript).toContain("thinking out loud");

    const big = "y".repeat(15_000);
    const many = Array.from({ length: 40 }, (_, i) => ({ type: "message", role: "user", content: `${i}:${big}` }));
    const bounded = buildAdvisorTranscript({ input: many });
    expect(bounded.length).toBeLessThan(410_000);
    expect(bounded).toContain("earlier transcript entries omitted");
    expect(bounded).toContain("39:");
    expect(bounded).not.toContain("\n0:");
  });

  test("error texts that disable the client advisor are neutralized", () => {
    expect(neutralizeAdvisorErrorText("The advisor tool is not available for this model")).toBe("upstream request failed");
    expect(neutralizeAdvisorErrorText("x cannot be used as an advisor")).toBe("upstream request failed");
    expect(neutralizeAdvisorErrorText("Input tag 'advisor_x' found")).toBe("upstream request failed");
    expect(neutralizeAdvisorErrorText("rate limited")).toBe("rate limited");
  });
});

describe("advisor outbound blocks", () => {
  const advisorItem = { type: "advisor_call", id: "srvtoolu_1", status: "completed", text: "Use a map." };

  test("SSE: advisor_call becomes server_tool_use + advisor_tool_result without a tool_use stop", async () => {
    const budget = createTestTranslatorBudget();
    const upstream = streamFrom([
      sse("response.output_text.delta", { delta: "Asking. " }),
      sse("response.output_item.done", { item: { type: "message", content: [] } }),
      sse("response.output_item.done", { item: advisorItem }),
      sse("response.output_text.delta", { delta: "Done." }),
      sse("response.completed", { response: { usage: { input_tokens: 5, output_tokens: 2 } } }),
    ].join(""));
    const anthropic = responsesSseToAnthropicSse(upstream, "m", { translatorBudget: budget, pingIntervalMs: 0 });
    const message = await collectAnthropicMessage(anthropic, "m", budget);
    expect(message.stop_reason).toBe("end_turn");
    expect(message.content).toEqual([
      { type: "text", text: "Asking. " },
      { type: "server_tool_use", id: "srvtoolu_1", name: "advisor", input: {} },
      { type: "advisor_tool_result", tool_use_id: "srvtoolu_1", content: { type: "advisor_result", text: "Use a map." } },
      { type: "text", text: "Done." },
    ]);
  });

  test("SSE: a failed consultation carries the error variant with a known code", async () => {
    const budget = createTestTranslatorBudget();
    const upstream = streamFrom([
      sse("response.output_item.done", { item: { type: "advisor_call", id: "srvtoolu_2", status: "failed", error_code: "bogus" } }),
      sse("response.completed", { response: {} }),
    ].join(""));
    const text = await new Response(responsesSseToAnthropicSse(upstream, "m", { translatorBudget: budget, pingIntervalMs: 0 })).text();
    expect(text).toContain("\"type\":\"server_tool_use\",\"id\":\"srvtoolu_2\",\"name\":\"advisor\"");
    expect(text).toContain("{\"type\":\"advisor_tool_result_error\",\"error_code\":\"unavailable\"}");
  });

  test("JSON: advisor_call maps to the same pair", () => {
    const message = responsesJsonToAnthropicMessage({
      status: "completed",
      output: [advisorItem, { type: "message", content: [{ type: "output_text", text: "ok" }] }],
    }, "m", createTestTranslatorBudget());
    expect((message.content as Rec[]).slice(0, 2)).toEqual([
      { type: "server_tool_use", id: "srvtoolu_1", name: "advisor", input: {} },
      { type: "advisor_tool_result", tool_use_id: "srvtoolu_1", content: { type: "advisor_result", text: "Use a map." } },
    ]);
    expect(message.stop_reason).toBe("end_turn");
  });
});

describe("native advisor adapter", () => {
  test("builds a tool-less Messages body that identifies as Claude Code", () => {
    const body = nativeAdvisorMessagesBody("claude-opus-5-5", {
      instructions: "ADVISE",
      input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "TRANSCRIPT" }] }],
    });
    expect(body).toEqual({
      model: "claude-opus-5-5",
      max_tokens: 8192,
      stream: false,
      system: [
        { type: "text", text: "You are Claude Code, Anthropic's official CLI for Claude." },
        { type: "text", text: "ADVISE" },
      ],
      messages: [{ role: "user", content: [{ type: "text", text: "TRANSCRIPT" }] }],
    });
  });

  test("drops only the advisor beta", () => {
    expect(withoutAdvisorBeta("oauth-2025-04-20, advisor-tool-2026-03-01,claude-code-20250219")).toBe("oauth-2025-04-20,claude-code-20250219");
    expect(withoutAdvisorBeta("advisor-tool-2026-03-01")).toBe("");
  });

  test("maps the Anthropic reply to the Responses shape and a too-long prompt to 413", async () => {
    const ok = await nativeAdvisorResponse(Response.json({ content: [{ type: "thinking", thinking: "x" }, { type: "text", text: "Advice." }] }));
    expect(await ok.json()).toEqual({ output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "Advice." }] }] });
    const tooLong = await nativeAdvisorResponse(Response.json(
      { type: "error", error: { type: "invalid_request_error", message: "prompt is too long: 250000 tokens > 200000 maximum" } },
      { status: 400 },
    ));
    expect(tooLong.status).toBe(413);
  });
});
