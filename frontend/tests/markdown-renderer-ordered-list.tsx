import assert from "node:assert/strict";

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { MarkdownRenderer } from "../src/components/MarkdownRenderer";

const text = [
  "1. Nguyen Minh Hieu",
  "",
  "   - GPA: 3.08.",
  "   - Expected graduation: 2026.",
  "",
  "2. Khang Do",
  "",
  "   - GPA: 3.67.",
  "   - IELTS: 7.5.",
  "",
  "3. Nguyen Van An",
  "",
  "   - Backend experience: 5 years.",
].join("\n");

const html = renderToStaticMarkup(<MarkdownRenderer text={text} />);

const orderedListCount = (html.match(/<ol\b/g) ?? []).length;
const unorderedListCount = (html.match(/<ul\b/g) ?? []).length;

assert.equal(
  orderedListCount,
  1,
  `Expected a single ordered list, received ${orderedListCount}.\n${html}`,
);
assert.equal(
  unorderedListCount,
  3,
  `Expected one nested unordered list per candidate, received ${unorderedListCount}.\n${html}`,
);
assert.ok(!html.includes("</ol><ul"), `Expected nested bullets inside the ordered list item.\n${html}`);
assert.match(html, /Nguyen Minh Hieu[\s\S]*?<ul\b/, `Expected nested bullets under Nguyen Minh Hieu.\n${html}`);
assert.match(html, /Khang Do[\s\S]*?<ul\b/, `Expected nested bullets under Khang Do.\n${html}`);
assert.match(html, /Nguyen Van An[\s\S]*?<ul\b/, `Expected nested bullets under Nguyen Van An.\n${html}`);

console.log("Markdown ordered-list nesting regression passed.");
