/* Build a self-contained, printable evidence review. No server-side retention.
 * Every label/note is escaped; generated image data URLs are allow-listed.
 */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (ch) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);

const safeImg = (src) => /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(src || "")
  ? src : "";

export function reportToHtml(report) {
  const figs = report.figures || [];
  const matches = report.image_reuse || [];
  const checks = report.checks || {};
  const fmt = (n) => Number.isFinite(Number(n)) ? Number(n).toFixed(4) : "n/a";
  const duplicates = report.duplicates || [];
  const pairs = matches.map((m, i) => {
    const a = figs[m.a.figure], b = figs[m.b.figure];
    if (!a || !b) return "";
    return '<article><h3>Image reuse comparison ' + (i + 1) + '</h3>' +
      '<p><strong>' + esc(m.method.replaceAll("_", " ")) + '</strong> — ' +
      esc(a.label) + ' ↔ ' + esc(b.label) +
      (typeof m.similarity === "number" ? ' (' + (100 * m.similarity).toFixed(1) + '% similarity)' : '') +
      '</p><div class="pair"><figure><img src="' + safeImg(a.image) + '"><figcaption>' + esc(a.label) +
      '</figcaption></figure><figure><img src="' + safeImg(b.image) + '"><figcaption>' + esc(b.label) +
      '</figcaption></figure></div><p>' + esc(m.note) + ' ' + esc(m.limitations) + '</p></article>';
  }).join("");
  const duplicatedBands = duplicates.map((d, i) =>
    '<article><h3>Blot match candidate ' + (i + 1) + '</h3>' +
    '<p>' + esc(d.a.figure_label || "figure") + ', panel ' + esc(d.a.panel_id + 1) +
    ', band ' + esc(d.a.id + 1) + ' compared with ' + esc(d.b.figure_label || "figure") +
    ', panel ' + esc(d.b.panel_id + 1) +
    (d.b.is_detected_band ? ', band ' + esc(d.b.id + 1) : ', matched image location') + '</p>' +
    '<p>Residual correlation ' + esc(fmt(d.noise_similarity)) +
    ', pixel-shape correlation ' + esc(fmt(d.shape_similarity)) +
    '. Neither is a probability of copying. Repeated natural patterns can match.</p>' +
    (safeImg(d.a.crop) && safeImg(d.b.crop)
      ? '<div class="pair"><figure><img src="' + safeImg(d.a.crop) +
        '"><figcaption>Source band crop</figcaption></figure><figure><img src="' +
        safeImg(d.b.crop) + '"><figcaption>Matched region (not necessarily a band)</figcaption></figure></div>'
      : '<p>Matched image crops unavailable in this older report.</p>') +
    '</article>').join("");
  const all = figs.map((fig) =>
    '<article><h3>' + esc(fig.label) + '</h3><p>' + esc(fig.width) + ' × ' + esc(fig.height) +
    ' pixels. Western blot screening: ' + esc(fig.blot_screen || (fig.panels?.length ? "experimental" : "not assessed")) +
    '. ' + esc(fig.blot_note || "") + '</p><img class="overview" src="' + safeImg(fig.image) + '">' +
    (fig.panels || []).map((p) => '<p>Panel ' + esc(p.id + 1) + ': experimental anomaly index ' +
      esc(fmt(p.score)) + ', flagged for review: ' + (p.flagged ? "yes" : "no") +
      '</p><div class="pair"><img src="' + safeImg(p.crop) + '"><img src="' + safeImg(p.heat) + '"></div>').join("") +
    '</article>').join("");
  const rows = Object.entries(checks).map(([key, status]) =>
    '<tr><td>' + esc(key.replaceAll("_", " ")) + '</td><td>' + esc(status) + '</td></tr>').join("");
  const provenance = report.provenance || {};
  const limits = (report.limitations || [
    "This report does not establish authenticity or scientific misconduct.",
    "Missing credentials and negative scores are inconclusive.",
  ]).map(x => '<li>' + esc(x) + '</li>').join("");
  return '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Image integrity review — ' +
    esc(report.filename) + '</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#1f2b33;max-width:900px;margin:40px auto;padding:0 22px}' +
    'h1,h2,h3{line-height:1.2}article{border-top:1px solid #bac9cf;padding:18px 0;margin:20px 0}img{max-width:100%;height:auto;object-fit:contain}' +
    '.overview{max-height:540px}.pair{display:flex;gap:18px;align-items:flex-start}.pair>*{width:48%;min-width:0}.pair figure{margin:0}' +
    '.pair figcaption{font-size:13px}th,td{text-align:left;padding:8px 14px;border-bottom:1px solid #ccd7dd}table{border-collapse:collapse}' +
    '.warning{padding:12px 16px;background:#f6f0eb}@media print{body{margin:8mm auto}article{break-inside:avoid}}' +
    '</style></head><body><h1>Scientific image integrity review</h1><p><strong>File:</strong> ' + esc(report.filename) +
    ' · <strong>Figures:</strong> ' + esc(figs.length) + ' · <strong>Potential whole-image reuse pairs:</strong> ' + esc(matches.length) +
    '</p><p class="warning"><strong>Human review required.</strong> Findings may reflect legitimate image reuse. ' +
    'No result proves fabrication; no findings do not establish authenticity.</p>' +
    '<h2>Checks performed</h2><table><thead><tr><th>Check</th><th>Status</th></tr></thead><tbody>' + rows +
    '</tbody></table><h2>Provenance</h2><p>' + esc(provenance.summary || "Not assessed") + '</p>' +
    '<h2>Potential whole-image reuse</h2>' + (pairs || '<p>No whole-image reuse was identified among the extracted figures.</p>') +
    '<h2>Experimental copied-band candidates</h2>' +
      (duplicatedBands || '<p>No copied-band candidate met the detector threshold, or the method was not run.</p>') +
    '<h2>Western blot screen summary</h2><p>Panel threshold: ' + esc(fmt(report.summary?.threshold)) +
    '; scored panels: ' + esc(report.summary?.panels) + '; flagged panels: ' +
    esc(report.summary?.flagged_panels) +
    '; unassessed candidates due to limit: ' + esc(report.summary?.skipped_panels || 0) +
    '. These scores are not probabilities of tampering.</p>' +
    (report.summary?.widespread_flag_figures?.length ?
      '<p>Numerous panels in the same figure were flagged. Shared imaging and publishing artifacts may explain multiple alerts.</p>' : '') +
    '<h2>Figure inventory and blot-specific screening</h2>' + all + '<h2>Limitations</h2><ul>' + limits +
    '</ul><p>Prepared by Below the Watermark for evidence review. No allegation about any original author is implied.</p></body></html>';
}

export function downloadReviewReport(report) {
  const blob = new Blob([reportToHtml(report)], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "below-the-watermark-review.html";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
