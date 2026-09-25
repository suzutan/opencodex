import { describe, expect, test } from "bun:test";
import { runAdvisorLoop, type AdvisorLoopDeps } from "../../src/claude/advisor-loop";
import { collectAnthropicMessage, responsesSseToAnthropicSse } from "../../src/claude/outbound";
import { createTestTranslatorBudget } from "../helpers/translator-budget";

type Rec = Record<string, unknown>;

function sse(name: string, data: Rec): string {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

function sseResponse(frames: string[], status = 200): Response {
  return new Response(frames.join(""), { status, headers: { "content-type": "text/event-stream" } });
}

function advisorCallFrames(outputIndex: number, callId: string): string[] {
  const item = { type: "function_call", id: `fc_${callId}`, call_id: callId, name: "advisor", arguments: "" };
  return [
    sse("response.output_item.added", { output_index: outputIndex, item }),
    sse("response.function_call_arguments.delta", { item_id: item.id, output_index: outputIndex, delta: "{}" }),
    sse("response.function_call_arguments.done", { item_id: item.id, output_index: outputIndex, arguments: "{}" }),
    sse("response.output_item.done", { output_index: outputIndex, item: { ...item, arguments: "{}", status: "completed" } }),
  ];
}

function textFrames(outputIndex: number, text: string): string[] {
  return [
    sse("response.output_item.added", { output_index: outputIndex, item: { type: "message", id: `msg_${outputIndex}`, role: "assistant", content: [] } }),
    sse("response.output_text.delta", { item_id: `msg_${outputIndex}`, output_index: outputIndex, delta: text }),
    sse("response.output_item.done", {
      output_index: outputIndex,
      item: { type: "message", id: `msg_${outputIndex}`, role: "assistant", content: [{ type: "output_text", text }] },
    }),
  ];
}

function completed(output: number, input = 100): string {
  return sse("response.completed", { response: { status: "completed", usage: { input_tokens: input, output_tokens: output } } });
}

const baseBody: Rec = {
  model: "mock/main",
  instructions: "You are Claude Code.",
  input: [{ type: "message", role: "user", content: [{ type: "input_text", text: "Refactor foo()" }] }],
  tools: [
    { type: "function", name: "advisor", parameters: { type: "object", properties: {} } },
    { type: "function", name: "Read", parameters: { type: "object", properties: {} } },
  ],
  stream: true,
};

interface Harness {
  stream: ReadableStream<Uint8Array>;
  advisorBodies: Rec[];
  turnBodies: Rec[];
}

function harness(overrides: Partial<AdvisorLoopDeps> & {
  first: string[];
  advisorAnswer?: () => Response | Promise<Response>;
  turns?: Array<() => Response>;
}): Harness {
  const advisorBodies: Rec[] = [];
  const turnBodies: Rec[] = [];
  const turns = [...(overrides.turns ?? [])];
  const stream = runAdvisorLoop({
    body: baseBody,
    spec: { model: "claude-opus-5-5", maxUses: 3 },
    advisorModel: "mock/advisor",
    translatorBudget: createTestTranslatorBudget(),
    dispatchAdvisor: async body => {
      advisorBodies.push(body);
      return overrides.advisorAnswer
        ? overrides.advisorAnswer()
        : sseResponse([
          sse("response.created", { response: {} }),
          sse("response.output_text.delta", { delta: "Split foo " }),
          sse("response.output_text.delta", { delta: "into two." }),
          completed(4),
        ]);
    },
    dispatchTurn: async body => {
      turnBodies.push(body);
      const next = turns.shift();
      if (!next) throw new Error("unexpected continuation");
      return next();
    },
    ...overrides,
    first: sseResponse(overrides.first),
  });
  return { stream, advisorBodies, turnBodies };
}

async function anthropicMessage(stream: ReadableStream<Uint8Array>): Promise<Rec> {
  const budget = createTestTranslatorBudget();
  return collectAnthropicMessage(responsesSseToAnthropicSse(stream, "m", { translatorBudget: budget, pingIntervalMs: 0 }), "m", budget);
}

describe("advisor loop", () => {
  test("consults the advisor, streams the pair, and continues the routed model with the advice", async () => {
    const h = harness({
      first: [sse("response.created", { response: {} }), ...textFrames(0, "Let me check. "), ...advisorCallFrames(1, "call_adv"), completed(7)],
      turns: [() => sseResponse([sse("response.created", { response: {} }), ...textFrames(0, "Splitting now."), completed(5, 140)])],
    });
    const raw = await new Response(h.stream).text();
    expect(raw).not.toContain("\"name\":\"advisor\"");
    expect(raw.match(/event: response\.created/g)).toHaveLength(1);
    expect(raw.match(/event: response\.completed/g)).toHaveLength(1);
    // Input stays the final iteration's; output accumulates across the turn.
    expect(raw).toContain("\"input_tokens\":140,\"output_tokens\":12");

    expect(h.advisorBodies).toHaveLength(1);
    const advisorBody = h.advisorBodies[0]!;
    expect(advisorBody.model).toBe("mock/advisor");
    expect(advisorBody.tools).toBeUndefined();
    expect(String(advisorBody.instructions)).toContain("stronger reviewer model");
    const transcript = JSON.stringify(advisorBody.input);
    expect(transcript).toContain("You are Claude Code.");
    expect(transcript).toContain("Refactor foo()");
    expect(transcript).toContain("Let me check.");

    expect(h.turnBodies).toHaveLength(1);
    const input = h.turnBodies[0]!.input as Rec[];
    expect(input.slice(1)).toEqual([
      { type: "message", role: "assistant", content: [{ type: "output_text", text: "Let me check. " }] },
      { type: "function_call", call_id: "call_adv", name: "advisor", arguments: "{}" },
      { type: "function_call_output", call_id: "call_adv", output: "Split foo into two." },
    ]);

    const message = await anthropicMessage(sseResponse([raw]).body!);
    expect(message.stop_reason).toBe("end_turn");
    const content = message.content as Rec[];
    expect(content.map(block => block.type)).toEqual(["text", "server_tool_use", "advisor_tool_result", "text"]);
    expect(content[1]).toMatchObject({ name: "advisor", input: {} });
    expect(String(content[1]!.id)).toStartWith("srvtoolu_");
    expect(content[2]).toEqual({
      type: "advisor_tool_result",
      tool_use_id: content[1]!.id,
      content: { type: "advisor_result", text: "Split foo into two." },
    });
  });

  test("an advisor call alongside a client tool ends the turn for the client to run the tool", async () => {
    const readCall = { type: "function_call", id: "fc_read", call_id: "call_read", name: "Read", arguments: "{\"p\":1}" };
    const h = harness({
      first: [
        ...advisorCallFrames(0, "call_adv"),
        sse("response.output_item.added", { output_index: 1, item: { ...readCall, arguments: "" } }),
        sse("response.function_call_arguments.delta", { item_id: "fc_read", output_index: 1, delta: "{\"p\":1}" }),
        sse("response.output_item.done", { output_index: 1, item: readCall }),
        completed(3),
      ],
    });
    const message = await anthropicMessage(h.stream);
    expect(h.turnBodies).toHaveLength(0);
    expect(h.advisorBodies).toHaveLength(1);
    expect(message.stop_reason).toBe("tool_use");
    expect((message.content as Rec[]).map(block => block.type)).toEqual(["tool_use", "server_tool_use", "advisor_tool_result"]);
    expect((message.content as Rec[])[0]).toMatchObject({ name: "Read", input: { p: 1 } });
  });

  test("an advisor failure becomes the error variant and the model continues without advice", async () => {
    const h = harness({
      first: [...advisorCallFrames(0, "call_adv"), completed(2)],
      advisorAnswer: () => new Response(JSON.stringify({ error: { message: "slow down" } }), { status: 429 }),
      turns: [() => sseResponse([...textFrames(0, "Proceeding alone."), completed(1)])],
    });
    const message = await anthropicMessage(h.stream);
    const content = message.content as Rec[];
    expect(content[1]).toMatchObject({ type: "advisor_tool_result", content: { type: "advisor_tool_result_error", error_code: "too_many_requests" } });
    const output = (h.turnBodies[0]!.input as Rec[]).at(-1)!;
    expect(output).toMatchObject({ type: "function_call_output", call_id: "call_adv" });
    expect(String(output.output)).toContain("unavailable (too_many_requests)");
  });

  test("an unresolvable advisor model reports unavailable without dispatching", async () => {
    const h = harness({
      advisorModel: null,
      first: [...advisorCallFrames(0, "call_adv"), completed(2)],
      turns: [() => sseResponse([...textFrames(0, "ok"), completed(1)])],
    });
    const message = await anthropicMessage(h.stream);
    expect(h.advisorBodies).toHaveLength(0);
    expect((message.content as Rec[])[1]).toMatchObject({ content: { error_code: "unavailable" } });
  });

  test("max_uses caps consultations, then the synthetic tool leaves the continuation", async () => {
    const h = harness({
      spec: { model: "claude-opus-5-5", maxUses: 1 },
      first: [...advisorCallFrames(0, "call_1"), completed(1)],
      turns: [
        () => sseResponse([...advisorCallFrames(0, "call_2"), completed(1)]),
        () => sseResponse([...textFrames(0, "final"), completed(1)]),
      ],
    });
    const message = await anthropicMessage(h.stream);
    expect(h.advisorBodies).toHaveLength(1);
    const results = (message.content as Rec[]).filter(block => block.type === "advisor_tool_result");
    expect(results.map(block => (block.content as Rec).type)).toEqual(["advisor_result", "advisor_tool_result_error"]);
    expect((results[1]!.content as Rec).error_code).toBe("max_uses_exceeded");
    expect((h.turnBodies[0]!.tools as Rec[]).map(tool => tool.name)).toEqual(["Read"]);
    expect((message.content as Rec[]).at(-1)).toEqual({ type: "text", text: "final" });
  });

  test("a forced advisor tool choice is released on the continuation", async () => {
    const h = harness({
      body: { ...baseBody, tool_choice: { type: "function", name: "advisor" } },
      first: [...advisorCallFrames(0, "call_1"), completed(1)],
      turns: [() => sseResponse([...textFrames(0, "done"), completed(1)])],
    });
    await new Response(h.stream).text();
    expect(h.turnBodies[0]!.tool_choice).toBe("auto");
  });

  test("a failed continuation surfaces as response.failed without advisor-disabling text", async () => {
    const h = harness({
      first: [...advisorCallFrames(0, "call_1"), completed(1)],
      turns: [() => new Response(JSON.stringify({ error: { message: "The advisor tool is not available" } }), { status: 400 })],
    });
    const raw = await new Response(h.stream).text();
    expect(raw).toContain("event: response.failed");
    expect(raw).not.toContain("advisor tool is not available");
    expect(raw).toContain("upstream request failed");
  });

  test("heartbeats keep the stream alive while the advisor runs", async () => {
    const h = harness({
      heartbeatMs: 5,
      first: [...advisorCallFrames(0, "call_1"), completed(1)],
      advisorAnswer: async () => {
        await Bun.sleep(40);
        return sseResponse([sse("response.output_text.delta", { delta: "late advice" }), completed(1)]);
      },
      turns: [() => sseResponse([...textFrames(0, "done"), completed(1)])],
    });
    const raw = await new Response(h.stream).text();
    expect((raw.match(/event: response\.heartbeat/g) ?? []).length).toBeGreaterThan(0);
    expect(raw).toContain("late advice");
  });

  test("a turn without an advisor call passes through unchanged", async () => {
    const frames = [sse("response.created", { response: {} }), ...textFrames(0, "plain"), completed(3)];
    const h = harness({ first: frames });
    expect(await new Response(h.stream).text()).toBe(frames.join(""));
  });
});
