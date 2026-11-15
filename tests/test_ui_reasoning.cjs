/* Tests the actual browser helper code without a DOM or network dependency. */
"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");

const client = fs.readFileSync(
  path.join(__dirname, "..", "ironasu", "web", "ui.js"), "utf8",
);
const start = client.indexOf("  function applySegment(answer, type, chunk)");
const end = client.indexOf("  async function streamCompletion", start);
assert.ok(start >= 0 && end > start, "expected metric/reasoning helpers in client");
const display = new Map();
const ctx = {
  performance: {now: () => 1500},
  byId: (id) => {
    if (!display.has(id)) display.set(id, {textContent: ""});
    return display.get(id);
  },
};
vm.createContext(ctx);
vm.runInContext(client.slice(start, end), ctx);

test("fragmented think tags are separated from final response", () => {
  const answer = {content: "", reasoning: ""};
  ["<thi", "nk>analysis", " here</thi", "nk>Final", " text"].forEach(
    (chunk) => ctx.appendTaggedContent(answer, chunk),
  );
  ctx.appendTaggedContent(answer, "", true);
  assert.equal(answer.reasoning, "analysis here");
  assert.equal(answer.content, "Final text");
  assert.equal(answer._firstToken, 1500);
});

test("upstream reasoning protocol fields are accepted, not fabricated", () => {
  assert.equal(ctx.extractReasoning({delta:{reasoning_content: "reason"}}), "reason");
  assert.equal(ctx.extractReasoning({delta:{reasoning: "thought"}}), "thought");
  assert.equal(ctx.extractReasoning({message:{reasoning_details:[
    {type: "reasoning.text", text: "part1"}, {text: "part2"}]}}), "part1part2");
  assert.equal(ctx.extractReasoning({delta:{content:"answer"}}), "");
});

test("TTFT and throughput are shown only when measurements exist", () => {
  ctx.paintPerformance({
    _started: 1000, _firstToken: 1200, _reasoningStart: 1200,
    _answerStart: 1400, _reasoningEnd: 1375,
    content: "Hello", tokens: 20,
  }, 2200);
  assert.equal(display.get("stat-ttft").textContent, "0.20s");
  assert.equal(display.get("stat-reasoning-time").textContent, "0.20s");
  assert.equal(display.get("stat-tokens-sec").textContent, "20.0");
  ctx.paintPerformance({_started: 1000, _firstToken: undefined,
                        content: "Final", tokens: 12}, 2000);
  assert.equal(display.get("stat-ttft").textContent, "—");
  assert.equal(display.get("stat-tokens-sec").textContent, "—");
});
