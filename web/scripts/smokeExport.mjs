import { readFileSync, writeFileSync } from "node:fs";
import { reportToHtml } from "../src/reportExport.js";
const report = JSON.parse(readFileSync("../results/integrity_smoke/controlled_reuse_report.json", "utf8"));
const html = reportToHtml(report);
if (!html.includes("Scientific image integrity review") || !html.includes("Image reuse comparison") ||
    !html.includes("page 1") || !html.includes("page 2") || !html.includes("Human review required") ||
    !html.includes("<!doctype html>")) {
  throw Error("Export missing required evidence and disclaimers");
}
const malicious = { ...report, filename: '<script>alert("x")</script>' };
if (reportToHtml(malicious).includes('<script>alert("x")</script>')) {
  throw Error("Export failed escaping unsafe input");
}
writeFileSync("../results/integrity_smoke/controlled_reuse_review.html", html);
console.log("SUCCESS: HTML report generated with image evidence, source labels and escaped untrusted text");
