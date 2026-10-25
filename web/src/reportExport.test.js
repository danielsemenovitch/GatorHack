import assert from "node:assert/strict";
import { test } from "node:test";
import { reportToHtml } from "./reportExport.js";

const tinyJpeg = "data:image/jpeg;base64,/9j/AA==";
function fixture() {
  return {
    filename: "controlled-demo.pdf",
    figures: [
      { id: 0, label: "page 1, image 1", width: 100, height: 80, image: tinyJpeg, panels: [] },
      { id: 1, label: "page 2, image 1", width: 100, height: 80, image: tinyJpeg, panels: [] },
    ],
    summary: { figures: 2, panels: 0, flagged_panels: 0, library: {figures:112,papers:62} },
    checks: { local_region_reuse: "between_figures_only" },
    region_reuse: [{
      a: { figure:0, label:"page 1, image 1", crop:tinyJpeg },
      b: { figure:1, label:"page 2, image 1", crop:tinyJpeg },
      inliers: 32, scale:0.82, flipped:true
    }],
    library_matches: [{
      label:"page 2, image 1",crop:tinyJpeg,source_crop:tinyJpeg,inliers:31,
      source:{figLabel:"Figure 2",pmcid:"PMC10701888",year:"2023",title:"Earlier figure",
        doi:"10.1234/example", url:"https://europepmc.org/article/PMC/10701888"}
    }],
    image_reuse: [], duplicates: [], limitations:["No definitive misconduct finding."],
  };
}
test("offline report includes new PR8 region and library comparison evidence", () => {
  const html = reportToHtml(fixture());
  assert.match(html, /Reused regions between figures in this paper/);
  assert.match(html, /Matching region in published reference/);
  assert.match(html, /32 aligned keypoints/);
  assert.match(html, /Source: <a href="https:\/\/europepmc.org\/article\/PMC\/10701888"/);
  assert.match(html, /112 figures from 62 papers/);
  assert.match(html, /Online Europe PMC search was not requested/);
  assert.doesNotMatch(html, /published before/);
});
test("report includes only an actually executed search and its scope", () => {
  const x = fixture();
  x.web_search = {
    status:"done", planner:"keywords", figures_compared:6, seconds:12.5,
    papers:[{pmcid:"PMC10701888",title:"Earlier figure",figures:6}],
    queries:[{query:'TITLE_ABS:("Western" AND "blot")',why:"same topic",results:2}],
    matches:[{crop:tinyJpeg,source_crop:tinyJpeg,figure_label:"page 1, image 1",inliers:29,
      source:{pmcid:"PMC10701888",title:"Earlier figure",url:"https://europepmc.org/article/PMC/10701888"}}],
  };
  const html=reportToHtml(x);
  assert.match(html, /Search agent match 1/);
  assert.match(html, /6; papers considered: 1/);
  assert.match(html, /TITLE_ABS:/);
  assert.match(html, /Search status: done/);
});
test("agent content is escaped and only allowlisted links and images are rendered", () => {
  const x=fixture();
  x.library_matches[0].source.title='<img src=x onerror=alert(1)>';
  x.library_matches[0].source.url='javascript:alert(1)';
  x.web_search={status:"offline",note:'<script>alert(1)</script>',queries:[],papers:[],matches:[]};
  const html=reportToHtml(x);
  assert.doesNotMatch(html, /<script>/);
  assert.doesNotMatch(html, /href="javascript:/);
  assert.doesNotMatch(html, /<img src=x/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /Search status: offline/);
});
