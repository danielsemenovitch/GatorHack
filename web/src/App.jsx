import React, { useCallback, useEffect, useRef, useState } from "react";
import { downloadReviewReport } from "./reportExport";

const API = import.meta.env.VITE_API || "";

const WHY = {
  res_std_ratio: "Its noise level differs from the rest of the blot.",
  res_std_bg_ratio: "The membrane grain around it differs from the rest of the blot.",
  res_kurt: "Its noise has an unusual shape, as if it came from another source.",
  lap_energy_ratio: "Its fine texture differs from the rest of the blot.",
  ela90_ratio: "It recompresses differently from the rest of the image.",
  ela75_ratio: "It recompresses differently from the rest of the image.",
  block_ratio: "Its JPEG block pattern is out of step with the rest of the image.",
  block_phase_ratio: "Its JPEG block grid is shifted, a sign of pasting.",
  grad_max_ratio: "Its edges are sharper or softer than the other bands.",
  edge_sharp_ratio: "Its edges are sharper or softer than the other bands.",
  bg_level_dev: "The background brightness around it does not match.",
  bg_flat_ratio: "The background around it is unusually smooth or grainy.",
  dark_mean: "Band density is unusual for this blot.",
  dark_frac: "Band coverage is unusual for this blot.",
  clip_frac_dev: "It has more fully saturated pixels than elsewhere.",
  quant_hist_dev: "Its pixel values have gaps, a sign of rescaling.",
  res_std_vs_neighbors: "Its noise jumps compared with the area right around it.",
  ela_vs_neighbors: "It recompresses differently from the area right around it.",
  noise_hf_ratio: "Its high frequency detail differs from the rest of the blot.",
  smooth_frac_ratio: "It has unusually smooth pixels.",
};

function BlotMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="6" fill="#1F2B33" />
      <rect x="5" y="9" width="9" height="4" rx="2" fill="#F6F8F9" />
      <rect x="18" y="9" width="9" height="4" rx="2" fill="#C8285F" />
      <rect x="5" y="19" width="9" height="3" rx="1.5" fill="#F6F8F9" />
      <rect x="18" y="19" width="9" height="3" rx="1.5" fill="#F6F8F9" />
    </svg>
  );
}

const pct = (v, total) => `${(100 * v) / total}%`;
const scoreFmt = (v) => Number.isFinite(Number(v)) ? Number(v).toFixed(4) : "n/a";

/* The figure on the lightbox: panels, heat, band boxes, duplicate links. */
function Stage({ fig, duplicates = [], showHeat = true, activeBand, onBand, threshold, knownEdit, knownEditLabel = "known edit" }) {
  const W = fig.width, H = fig.height;
  const links = duplicates.filter((d) => d.a.figure === fig.id && d.b.figure === fig.id);
  const panelById = Object.fromEntries(fig.panels.map((p) => [p.id, p]));
  return (
    <div className="stage">
      <img className="base" src={fig.image} alt={`Figure: ${fig.label}`} />
      {fig.panels.map((p) => (
        <React.Fragment key={p.id}>
          <div className={`panel-box ${p.flagged ? "flagged" : ""}`}
               style={{ left: pct(p.x, W), top: pct(p.y, H), width: pct(p.w, W), height: pct(p.h, H) }} />
          {/* Heat and hot band outlines are drawn ONLY for panels above the review
              threshold. A single hot 32 px patch inside an unflagged panel used to
              paint pink boxes that contradicted the "no flags" verdict. */}
          {p.flagged && <img className="heat" src={p.heat} alt="" style={{ left: pct(p.x, W), top: pct(p.y, H),
               width: pct(p.w, W), height: pct(p.h, H), opacity: showHeat ? 1 : 0 }} />}
          {onBand && p.bands.map((b) => {
            const key = `${p.id}-${b.id}`;
            const hot = p.flagged && b.score > threshold;
            return (
              <button key={key} type="button"
                className={`band ${hot ? "hot" : ""} ${activeBand === key ? "active" : ""}`}
                aria-label={`Band in panel ${p.id + 1}, experimental anomaly index ${scoreFmt(b.score)}`}
                onClick={() => onBand(key, p, b)}
                style={{ left: pct(p.x + b.x - 2, W), top: pct(p.y + b.y - 2, H),
                         width: pct(b.w + 4, W), height: pct(b.h + 4, H) }} />
            );
          })}
        </React.Fragment>
      ))}
      {knownEdit && (
        <div className="known-edit" title={knownEdit.note}
             style={{ left: `${100 * knownEdit.x / knownEdit.image_width}%`, top: `${100 * knownEdit.y / knownEdit.image_height}%`,
                      width: `${100 * knownEdit.w / knownEdit.image_width}%`, height: `${100 * knownEdit.h / knownEdit.image_height}%` }}>
          <span>{knownEditLabel}</span>
        </div>
      )}
      {links.length > 0 && (
        <svg className="links" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          {links.map((d, i) => {
            const pa = panelById[d.a.panel_id], pb = panelById[d.b.panel_id];
            if (!pa || !pb) return null;
            const x1 = pa.x + d.a.x + d.a.w / 2, y1 = pa.y + d.a.y + d.a.h / 2;
            const x2 = pb.x + d.b.x + d.b.w / 2, y2 = pb.y + d.b.y + d.b.h / 2;
            return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#C8285F" strokeWidth={Math.max(1.5, W / 400)}
                         strokeDasharray={`${W / 120} ${W / 200}`} vectorEffect="non-scaling-stroke" />;
          })}
        </svg>
      )}
    </div>
  );
}

function Meter({ value, threshold }) {
  return (
    <div className="meter" role="img" aria-label={`Score ${Math.round(value * 100)} of 100, flag line at ${Math.round(threshold * 100)}`}>
      <span style={{ width: `${Math.min(100, value * 100)}%` }} />
      <i style={{ left: `${threshold * 100}%` }} />
    </div>
  );
}

function downloadDataUrl(dataUrl, name) {
  const a = document.createElement("a");
  a.href = dataUrl; a.download = name; document.body.appendChild(a); a.click(); a.remove();
}

function Report({ report }) {
  const [showHeat, setShowHeat] = useState(true);
  const [active, setActive] = useState(null);
  const s = report.summary;
  const thr = s.threshold;
  const flagged = report.figures.flatMap((f) => f.panels.filter((p) => p.flagged).map((p) => ({...p, figure_label: f.label})));
  const prov = report.provenance || {};
  const imageReuse = report.image_reuse || [];
  const aiDeclared = !!s.credential_declares_ai;
  const title = aiDeclared ? "Content Credentials declare this image AI generated" :
    report.verdict === "suspicious" ? "Potential findings for human review" :
    "Figure inventory and available checks";
  const skippedImages = report.skipped_images || [];
  const skippedCounts = skippedImages.reduce((acc, x) => ({ ...acc, [x.reason]: (acc[x.reason] || 0) + 1 }), {});
  const SKIP_WORDS = { too_small: "too small to analyze", too_large: "over the 12 megapixel limit", figure_limit: "beyond the 48 figure limit", pixel_budget: "beyond the 150 megapixel total per upload", unreadable: "could not be decoded" };
  const sub = imageReuse.length
    ? `${imageReuse.length === 1 ? "One image appears" : `${imageReuse.length} images appear`} twice in this upload${s.panels ? `, and ${s.panels} Western blot panel${s.panels === 1 ? " was" : "s were"} screened` : ""}.`
    : s.panels ? `${s.panels} Western blot panel${s.panels === 1 ? "" : "s"} screened with an experimental classifier. A negative result does not establish authenticity.`
    : "No confidently identified Western blot panels. General image inventory remains available; the blot classifier was not run on other image types.";

  return (
    <section className="report" aria-live="polite">
      <div>
        <h2>{title}</h2>
        <p className="sub">{sub}</p>
        {(() => {
          // Notes that change how to read this report, most important first.
          const notes = [];
          if (report.known_edit) {
            const k = report.known_edit, f0 = report.figures[0];
            const sx = f0 ? f0.width / k.image_width : 1, sy = f0 ? f0.height / k.image_height : 1;
            const ex0 = k.x * sx, ey0 = k.y * sy, ex1 = (k.x + k.w) * sx, ey1 = (k.y + k.h) * sy;
            const hits = (f0?.panels || []).filter((p) => p.flagged && p.x < ex1 && p.x + p.w > ex0 && p.y < ey1 && p.y + p.h > ey0);
            const others = (f0?.panels || []).filter((p) => p.flagged).length - hits.length;
            notes.push(["Known edit", <>The dashed box marks the band we pasted in for this demo. The model {hits.length ? "flagged the panel that contains it" : "did not flag the panel that contains it"}{others > 0 ? `, and flagged ${others} untouched panel${others === 1 ? "" : "s"} (a false positive)` : ""}.</>, hits.length ? "good" : "bad"]);
          }
          if (aiDeclared) notes.push(["Provenance", <>{prov.summary} This is a declaration attached to the file, not a pixel analysis, and any re-save removes it.</>, "bad"]);
          if (skippedImages.length) notes.push(["Not analyzed", <>{Object.entries(skippedCounts).map(([k, n]) => `${n} image${n === 1 ? "" : "s"} ${SKIP_WORDS[k] || k}`).join("; ")}. Those images were not checked at all.</>, "bad"]);
          if (s.skipped_panels) notes.push(["Incomplete screening", <>{s.skipped_panels} candidate blot panels were skipped after the {s.max_panels} panel limit. That does not mean they are clean.</>, "bad"]);
          if ((s.widespread_flag_figures || []).length) notes.push(["Many flags in one figure", <>{s.widespread_flag_figures.map((f) => f.label).join(", ")}: shared background, scanner or publishing artifacts can explain several alerts at once.</>, "info"]);
          if (s.resized_for_analysis) notes.push(["Resolution", <>{s.resized_for_analysis} figure{s.resized_for_analysis === 1 ? " is" : "s are"} sharper than the web figures (about {s.analysis_width} px wide) the blot model learned from, so blot panels were scored on copies resized to that scale.</>, "info"]);
          return notes.length > 0 && <ul className="notes">{notes.map(([k, v, tone]) => <li key={k} className={tone}><strong>{k}.</strong> {v}</li>)}</ul>;
        })()}
        <div className="toolbar">
          <label className="toggle"><input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} disabled={!s.flagged_panels} /> Show heat on flagged panels</label>
          <button className="btn ghost small" type="button" onClick={() => downloadReviewReport(report)}>Download review report (HTML)</button>
        </div>
        {imageReuse.length > 0 && (
          <div className="detail reuse">
            <h3>The same image appears twice</h3>
            {imageReuse.map((m, i) => {
              const a = report.figures[m.a.figure], b = report.figures[m.b.figure];
              if (!a || !b) return null;
              return <div key={i} className="reuse-pair">
                <p><strong>{m.method === "exact_pixel_duplicate" ? "Identical pixels" : "Near duplicate"}:</strong> {a.label} and {b.label}. Reuse can be legitimate, so compare the captions and methods before drawing any conclusion.</p>
                <div className="pair">
                  {[a, b].map((f) => <figure key={f.id}>
                    <img src={f.image} alt={f.label} />
                    <figcaption>{f.label}, {f.native_width || f.width} × {f.native_height || f.height} px</figcaption>
                  </figure>)}
                </div>
                <p className="sub">{m.note} This check compares whole images; it does not look for copied regions inside different images.</p>
              </div>;
            })}
          </div>
        )}
        {report.figures.map((fig) => (
          <div key={fig.id} style={{ marginBottom: 20 }}>
            <p className="figlabel"><strong>{fig.label}</strong> <span>{fig.native_width || fig.width} × {fig.native_height || fig.height} px</span>
              {imageReuse.some((m) => m.a.figure === fig.id || m.b.figure === fig.id) ? <span className="tag bad">same image as {imageReuse.filter((m) => m.a.figure === fig.id || m.b.figure === fig.id).map((m) => report.figures[m.a.figure === fig.id ? m.b.figure : m.a.figure]?.label).join(", ")}</span> : null}
              <span className={`tag ${fig.panels.some((p) => p.flagged) ? "bad" : ""}`}>{fig.blot_screen === "experimental" ? `${fig.panels.length} blot panel${fig.panels.length === 1 ? "" : "s"} screened${fig.panels.some((p) => p.flagged) ? `, ${fig.panels.filter((p) => p.flagged).length} flagged` : ""}` : fig.blot_screen === "partial" ? "Partly screened" : fig.blot_screen === "not_assessed_limit" ? "Not screened (limit reached)" : "No blot panels found"}</span>
              {fig.repeated_placements?.length ? <span className="tag">also drawn {fig.repeated_placements.length} more time{fig.repeated_placements.length === 1 ? "" : "s"}</span> : null}</p>
            <div className="lightbox">
              <Stage fig={fig} duplicates={report.duplicates || []} showHeat={showHeat} threshold={thr || 0.5}
                     knownEdit={fig.id === 0 ? report.known_edit : null}
                     activeBand={active?.key} onBand={(key, p, b) => setActive({ key, p, b })} />
              {fig.id === 0 && report.credit && <p className="credit">{report.credit}</p>}
            </div>
          </div>
        ))}
        {active && (
          <div className="detail">
            <h3>Band in panel {active.p.id + 1}: <span className="score">{scoreFmt(active.b.score)}</span></h3>
            <p>Experimental band anomaly index; panel review threshold {scoreFmt(thr)}. This is not a probability of editing.</p>
            <p className="sub" style={{ margin: 0 }}>
              {active.p.flagged && active.b.score > thr ? "In a flagged panel and above the experimental review threshold. Investigate the underlying pixels and controls." :
               active.b.score > thr ? "This band's local index is high, but its panel as a whole is below the review threshold, so it is not flagged." :
               "Below the experimental review threshold. That does not establish an unmodified band."}
            </p>
            <p className="sub">Local texture and band intensity can vary naturally; no causal explanation for a CNN flag is established.</p>
          </div>
        )}
        {flagged.length > 0 && (
          <div className="panels">
            {flagged.map((p) => (
              <figure className="pcard" key={p.global_index}>
                <div className="stage">
                  <img className="base" src={p.crop} alt={`Blot panel ${p.id + 1}`} />
                  <img className="heat" src={p.heat} alt="" style={{ left: 6, top: 6, width: "calc(100% - 12px)", height: "calc(100% - 12px)" }} />
                </div>
                <figcaption>
                  <span>{p.figure_label}, panel {p.id + 1}: index <span className="score">{scoreFmt(p.score)}</span></span>
                  <button className="btn ghost small" type="button" onClick={() => downloadDataUrl(p.crop, `panel_${p.global_index + 1}.png`)}>Download panel crop</button>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>
      <aside className="rail" aria-label="Checks">
        <div className="check">
          <h3>Band splice detector</h3>
          <span className={`state ${s.flagged_panels ? "bad" : "info"}`}>{!s.panels ? "Not assessed" : s.flagged_panels ? `${s.flagged_panels} review flag${s.flagged_panels === 1 ? "" : "s"}` : "No flags found"}</span>
          <p>Experimental model, run only on panels that look like Western blots. {s.policy_note}</p>
          {s.panels > 0 && s.screening_policy === "benchmark" && s.model_version === "v3" &&
            <p><strong>Expect some chance flags:</strong> this threshold flags about 1 in 20 unedited panels, so with {s.panels} panel{s.panels === 1 ? "" : "s"} screened about {(s.panels * 0.05).toFixed(1)} flag{s.panels * 0.05 === 1 ? "" : "s"} would be expected even if nothing was edited.</p>}
          {s.panels > 0 && thr != null && <p>Model {s.model_version || report.cnn_version || "not recorded"}: highest panel index <strong>{scoreFmt(s.max_splice_score)}</strong>, flag line <strong>{scoreFmt(thr)}</strong>. Indices are not probabilities. Heat shows overlapping 32 pixel patches, not the exact edge of an edit.</p>}
        </div>
        <div className="check">
          <h3>Whole-image reuse</h3>
          <span className={`state ${imageReuse.length ? "bad" : "info"}`}>{imageReuse.length ? `${imageReuse.length} pair${imageReuse.length === 1 ? "" : "s"}` : "No match identified"}</span>
          <p>Compares extracted figures using decoded-pixel hashes and verified visual similarity. Currently detects whole-image copies, not pasted patches.</p>
        </div>
        <div className="check">
          <h3>Copied blot bands</h3>
          <span className={`state ${s.duplicate_pairs ? "bad" : "info"}`}>{!s.panels ? "Not assessed" : s.duplicate_pairs ? `${s.duplicate_pairs} candidate pairs` : "No match identified"}</span>
          <p>Experimental within-blot noise template matching; unreliable after resizing and compression. Repeated patterns can be legitimate.</p>
          {(report.duplicates || []).slice(0, 5).map((d, i) => (
            <div key={i} style={{ marginBottom: 18 }}>
              <p><strong>Candidate {i + 1}:</strong> {d.a.figure_label || "Figure"} panel {d.a.panel_id + 1} band {d.a.id + 1} compared with {d.b.figure_label || "Figure"} panel {d.b.panel_id + 1}{d.b.is_detected_band ? ` band ${d.b.id + 1}` : " matched image region"}.
                {d.transform !== "none" ? ` Transformation: ${d.transform.replace("_", " ")}.` : ""}
              </p>
              {d.a.crop && d.b.crop && <div style={{ display: "flex", gap: 10 }}>
                <figure style={{ width: "50%", margin: 0 }}><img src={d.a.crop} alt="Source band crop" style={{ width: "100%" }} /><figcaption>Source region</figcaption></figure>
                <figure style={{ width: "50%", margin: 0 }}><img src={d.b.crop} alt="Matched image region" style={{ width: "100%" }} /><figcaption>Matched region</figcaption></figure>
              </div>}
              <p className="sub">Noise correlation {scoreFmt(d.noise_similarity)}; shape correlation {scoreFmt(d.shape_similarity)}. Neither is a probability that pixels were copied.</p>
            </div>
          ))}
        </div>
        <div className="check">
          <h3>Content credentials</h3>
          <span className={`state ${aiDeclared ? "bad" : "info"}`}>{prov.status === "not_assessed" ? "Not assessed" : prov.present ? (prov.validated ? "Credential validated" : prov.integrity_ok ? "Credential read" : "Unverified") : "Not found"}</span>
          <p>{prov.summary} Absence does not imply manipulation or authenticity.</p>
        </div>
        <div className="check">
          <h3>SynthID watermark</h3>
          <span className="state info">Check by hand</span>
          <p>{report.synthid.note}</p>
          <div className="row">
            <a className="btn ghost small" href={report.synthid.url} target="_blank" rel="noreferrer">Open SynthID Detector</a>
          </div>
        </div>
        <p className="sub" style={{ marginTop: 16, fontSize: "0.8rem" }}>Analyzed in {report.seconds}s. Scores flag regions for a human to review; they are not proof of misconduct.</p>
      </aside>
    </section>
  );
}

function CheckPage({ examples, report, setReport, results }) {
  // Measured trade-off of the two v3 alert lines on held-out benchmark papers
  // (results/audit_resolution.json, written by scripts/audit_resolution.py).
  const op = results?.audit_resolution?.benchmark?.native;
  const main = results?.cnn_v3_metrics; // the headline benchmark (same numbers as the deck and Devpost)
  const [busy, setBusy] = useState(false);
  const [blotMode, setBlotMode] = useState("auto");
  const [cnnVersion, setCnnVersion] = useState("v3");
  const [screeningPolicy, setScreeningPolicy] = useState("benchmark");
  const [error, setError] = useState(null);
  const [over, setOver] = useState(false);
  const input = useRef(null);
  const hero = examples.find((e) => e.hero) || examples[0];

  const send = useCallback(async (file, extras = {}) => {
    setBusy(true); setError(null);
    try {
      const fd = new FormData(); fd.append("file", file); fd.append("blot_mode", blotMode);
      fd.append("cnn_version", cnnVersion);
      fd.append("screening_policy", screeningPolicy);
      const r = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "The analysis failed.");
      setReport({ ...j, ...extras });
      setTimeout(() => document.getElementById("report")?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (e) {
      setError(`${e.message} Try a PNG, JPEG or PDF under 25 MB.`);
    } finally { setBusy(false); }
  }, [setReport, blotMode, cnnVersion, screeningPolicy]);

  const loadExample = async (ex) => {
    // Reports in /examples/*.report.json are historical snapshots. Recompute
    // on the source bytes with today's CNN/policy so stale heat, thresholds
    // or falsely flagged panels cannot masquerade as fresh analysis.
    setError(null); setBusy(true);
    try {
      if (!ex.file) throw new Error("Example source image missing");
      const r = await fetch(`${API}/api/examples/${encodeURIComponent(ex.file)}`);
      if (!r.ok) throw new Error("Could not load the example image");
      const blob = await r.blob();
      // Keep the example's attribution and (for our own edits) the ground
      // truth with the live report, so the view and the HTML export say who
      // made the figure, that we made the edit, and where it is.
      await send(new File([blob], ex.file, { type: blob.type || "image/jpeg" }),
                 { credit: ex.credit, known_edit: ex.known_edit, example_title: ex.title });
    } catch (e) {
      setError(`Could not analyze this example: ${e.message}`);
      setBusy(false);
    }
  };

  return (
    <>
      <section className="hero">
        <div>
          <h1>AI watermarks miss a single pasted band.</h1>
          <p className="lede">
            We pasted bands made with Gemini into four published Western blots. Google's SynthID Detector found none of them.
            Below the Watermark reviews the figures a watermark can't vouch for.
          </p>
          <div className="actions">
            <button className="btn" type="button" onClick={() => input.current?.click()} disabled={busy}>{busy ? "Reviewing…" : "Review a figure or PDF"}</button>
            {hero && <button className="btn ghost" type="button" onClick={() => loadExample(hero)}>Run our checks on this blot</button>}
          </div>
        </div>
        {hero?.preview && (
          <div className="lightbox">
            <figure>
              <Stage fig={{ ...hero.preview, panels: [] }} showHeat={false} threshold={1}
                     knownEdit={hero.known_edit} knownEditLabel="AI band we added" />
              <figcaption>
                <span className="verdict-chip">SynthID Detector: <span className={hero.synthid === "not detected" ? "no" : ""}>{hero.synthid === "pending" ? "check pending" : hero.synthid}</span></span>
                <span className="verdict-chip">Content credential: <span>none</span></span>
              </figcaption>
              {hero.credit && <p className="credit">{hero.credit}</p>}
            </figure>
          </div>
        )}
      </section>

      <div className={`drop ${over ? "over" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }}
           onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) send(f); }}>
        <p><strong>Drop a figure or a paper PDF here.</strong> We list every embedded figure, look for images used twice, read content credentials and screen Western blots band by band. Results are leads for a human reviewer, not verdicts. Uploads are analyzed in memory and not stored.</p>
        <div className="controls">
          <span className="field">
            <label htmlFor="blot-mode">Western blot screen</label>
            <select id="blot-mode" value={blotMode} onChange={(e) => setBlotMode(e.target.value)}>
              <option value="auto">Automatic</option>
              <option value="blot">I confirm this image includes Western blots</option>
              <option value="off">Skip blot-specific screening</option>
            </select>
          </span>
          <span className="field">
            <label htmlFor="cnn-version">Experimental CNN</label>
            <select id="cnn-version" value={cnnVersion} onChange={(e) => setCnnVersion(e.target.value)}>
              <option value="v3">V3, larger training set (experimental)</option>
              <option value="v2">V2, legacy research scores (more sensitive)</option>
            </select>
          </span>
          <span className="field">
            <label htmlFor="screening-policy">Alert sensitivity</label>
            <select id="screening-policy" value={screeningPolicy} onChange={(e) => setScreeningPolicy(e.target.value)}>
              <option value="benchmark">Research threshold (default)</option>
              <option value="conservative">Strict (very few alerts, misses most edits)</option>
            </select>
          </span>
          {main && op && cnnVersion === "v3" && <span className="sub" style={{ fontSize: 13, margin: 0, flexBasis: "100%" }}>
            On {main.n_test_papers} held-out papers with simulated edits, the research threshold caught {Math.round(100 * main.detect_rate_by_method.ai_splice)}% of
            pasted AI bands and flagged {Math.round(100 * main.false_flag_rate_clean)}% of clean panels. On {results.audit_resolution.benchmark.n_test_papers} of
            those papers, strict caught {Math.round(100 * op["ai_splice_caught@0.9"])}% and flagged {Math.round(100 * op["clean_flagged@0.9"])}%.</span>}
          {cnnVersion === "v2" && screeningPolicy === "conservative" &&
            <span className="sub" style={{ fontSize: 13, margin: 0 }}>V2 scores saturate near 1.0; it keeps its legacy threshold. Use V3 for the strict setting.</span>}
        </div>
        <div className="examples">
          <input ref={input} className="sr" type="file" accept="image/*,.pdf" onChange={(e) => e.target.files?.[0] && send(e.target.files[0])} />
          <button className="btn" type="button" onClick={() => input.current?.click()} disabled={busy}>{busy ? "Checking…" : "Choose a file"}</button>
          {examples.filter((e) => !e.hero).map((ex) => (
            <button key={ex.id} className="btn ghost" type="button" onClick={() => loadExample(ex)}>{ex.title}</button>
          ))}
        </div>
      </div>
      {error && <div className="error" role="alert">{error}</div>}
      {busy && <p className="loading">Finding panels, scoring bands and comparing noise between bands…</p>}
      <div id="report">{report && <Report report={report} />}</div>
    </>
  );
}

function cellClass(v) {
  if (!v) return "unsure";
  if (/^(detected|caught|present)/.test(v)) return "caught";
  if (/not detected|missed|stripped/.test(v)) return "missed";
  return "unsure";
}

/* Small schematic of what was uploaded: a whole blot, one band, or one band inside a real blot. */
function ScaleGlyph({ kind }) {
  const band = (x, y, w, h, fill) => <rect key={`${x}-${y}`} x={x} y={y} width={w} height={h} rx={h / 2} fill={fill} />;
  const ink = "#1F2B33", pink = "#C8285F";
  return (
    <svg className="glyph" viewBox="0 0 96 56" aria-hidden="true">
      {kind !== "band" && <rect x="1" y="1" width="94" height="54" rx="3" fill="#fff" stroke={ink} strokeWidth="1.5" />}
      {kind === "whole" && [10, 31, 52, 73].flatMap((x) => [band(x, 14, 14, 6, pink), band(x, 34, 14, 5, pink)])}
      {kind === "band" && band(30, 22, 36, 12, pink)}
      {kind === "pasted" && [10, 31, 52, 73].flatMap((x) => [band(x, 14, 14, 6, x === 52 ? pink : ink), band(x, 34, 14, 5, ink)])}
    </svg>
  );
}

// Which hand run SynthID row used the same publishing step (whole AI blot).
const ROB_TO_SYNTHID = { none: "A", resave: "B", resize_50: "C", jpeg_q60: "D", contrast: "F", pdf_roundtrip: "H" };

function GapPage({ results }) {
  const sid = results?.synthid_manual;
  // The app's default model is v3, so its own metrics lead this page; v2 is
  // shown only as the legacy comparison (audit Oct 24: the page used to show
  // v2 numbers next to a v3 default).
  const det = results?.cnn_v3_metrics || results?.cnn_metrics || results?.detector_metrics;
  const isV3 = !!results?.cnn_v3_metrics;
  const legacy = isV3 ? results?.cnn_metrics : null;
  const base = results?.detector_metrics;
  const rob = (isV3 ? results?.cnn_v3_robustness : null) || results?.robustness;
  const robNoResize = rob ? rob.rows.filter((r) => !/resize/.test(r.transform) && r.transform !== "none") : [];
  const robResize = rob ? rob.rows.find((r) => r.transform === "resize_50") : null;
  const pctR = (v) => `${Math.round(v * 100)}%`;
  const groups = [
    ["Whole AI image", (r) => !/Single|pasted|untouched/.test(r.scenario)],
    ["AI band on its own", (r) => /Single/.test(r.scenario)],
    ["AI band pasted into a real blot", (r) => /pasted/.test(r.scenario)],
    ["Control", (r) => /untouched/.test(r.scenario)],
  ];
  return (
    <section className="page">
      <h1>Where the watermark stops working</h1>
      <div className="prose">
        <p>We made Western blots with Gemini's image tools and checked them with SynthID Detector, which Google opened to the public on October 7, 2026. Every upload was done by hand, one fixed scenario at a time. The content credential that Gemini attaches was stripped by the first re-save every time, so the watermark is the only signal left.</p>
      </div>
      {sid && (() => {
        const rowsOf = (fn) => sid.rows.filter(fn).filter((r) => r.synthid !== "pending");
        const whole = rowsOf(groups[0][1]), single = rowsOf(groups[1][1]), pastedDoneRows = rowsOf(groups[2][1]);
        const steps = [
          ["whole", "A whole AI blot", whole.filter((r) => r.synthid === "detected").length, whole.length, "detected", "after re-saving, resizing, JPEG, cropping, contrast and PDF export", "caught"],
          ["band", "One AI band on its own", single.filter((r) => r.synthid === "not enough details").length, single.length, "not enough details", "too small for the detector to judge", "unsure"],
          ["pasted", "One AI band in a published blot", pastedDoneRows.filter((r) => r.synthid === "not detected").length, pastedDoneRows.length, "not detected", "each band was cut from a watermarked Gemini image", "missed"],
        ];
        return <ol className="ladder" aria-label="SynthID Detector results by size of the AI region">
          {steps.filter((st) => st[3] > 0).map(([kind, label, n, total, verdict, note, tone]) => (
            <li key={kind}>
              <ScaleGlyph kind={kind} />
              <span className="what">{label}</span>
              <span className={`verdict ${tone}`}>{verdict}</span>
              <span className="count">{n} of {total} uploads, {note}</span>
            </li>
          ))}
        </ol>;
      })()}
      {sid && (
        <div className="scroll">
          <table className="matrix">
            <thead><tr><th>What was uploaded</th><th>SynthID Detector</th><th>Content credentials</th></tr></thead>
            <tbody>
              {groups.map(([name, fn]) => (
                <React.Fragment key={name}>
                  <tr className="group"><td colSpan={3}>{name}</td></tr>
                  {sid.rows.filter(fn).map((r) => (
                    <tr key={r.id}>
                      <td>{r.scenario}</td>
                      <td className={`cell ${r.id === "S0" && r.synthid === "not detected" ? "caught" : cellClass(r.synthid)}`}>{r.synthid}</td>
                      <td className={`cell ${r.c2pa === "present" ? "caught" : "unsure"}`}>{r.c2pa}</td>
                    </tr>
                  ))}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="sub" style={{ marginTop: 10 }}>{sid?.notes?.join(" ")}</p>

      <h2>Our detector on the same problem</h2>
      {det ? (
        <>
          <div className="prose"><p>{isV3 ? "CNN v3, the app's default model: a" : "A"} small convolutional network trained and tested on the Below the Watermark benchmark: real blot panels from CC BY papers, forged five ways and pushed through a random publishing chain. Test papers never appear in training, and this flag line (the research benchmark threshold) is set on separate calibration papers. The app uses this research line by default; its optional strict setting flags far fewer panels and catches far fewer edits. All edits are simulated; this is a first baseline on a hard problem, not a validated detector.</p></div>
          <div className="stats">
            <div className="stat"><b>{det.image_auroc}</b><span>AUROC, forged vs clean panels</span></div>
            <div className="stat"><b>{Math.round(det.detect_rate_by_method?.ai_splice * 100)}%</b><span>of pasted AI bands caught</span></div>
            <div className="stat"><b>{Math.round(det.false_flag_rate_clean * 100)}%</b><span>of clean panels wrongly flagged</span></div>
            <div className="stat"><b>{det.n_test_papers}</b><span>held out papers in the test set</span></div>
          </div>
          {results?.audit_resolution?.benchmark?.native && (() => {
            const op = results.audit_resolution.benchmark.native, n = results.audit_resolution.benchmark.n;
            return <p className="sub" style={{ marginTop: 10 }}><strong>The optional strict line (0.90) is much stricter:</strong> on {results.audit_resolution.benchmark.n_test_papers} of these held-out papers ({n.clean} clean and {n.ai_splice} AI splice images) it caught {Math.round(100 * op["ai_splice_caught@0.9"])}% of pasted AI bands and flagged {Math.round(100 * op["clean_flagged@0.9"])}% of clean panels, versus {Math.round(100 * op["ai_splice_caught@0.5"])}% and {Math.round(100 * op["clean_flagged@0.5"])}% at the research line.</p>;
          })()}
          {results?.figure_level_eval && (() => {
            const r = results.figure_level_eval.summary.benchmark;
            return <p style={{ marginTop: 14 }}><strong>End to end on whole figures:</strong> we pasted one Gemini band into one blot panel of each of {r.figures} held-out published figures and ran the app exactly as on an upload. At the default research threshold it flagged the edited panel in {r.edited_panel_flagged} of {r.figures}, flagged an untouched panel in {r.edited_figure_with_untouched_panel_flagged} of the edited figures, and flagged {r.original_figure_with_any_flag} of the {r.figures} unedited originals.</p>;
          })()}
          {legacy && <p className="sub" style={{ marginTop: 10 }}>Legacy CNN v2 on its own (smaller) test set: AUROC {legacy.image_auroc}, {pctR(legacy.detect_rate_by_method?.ai_splice)} of pasted AI bands caught, {pctR(legacy.false_flag_rate_clean)} of clean panels flagged.</p>}
          {base && <p className="sub" style={{ marginTop: 6 }}>A gradient boosted model on hand built forensic features reached an AUROC of {base.image_auroc} on the earlier, smaller benchmark.</p>}
          <div className="scroll">
            <table className="matrix">
              <thead><tr><th>Forgery</th><th>Caught at the flag line</th></tr></thead>
              <tbody>
                {Object.entries(det.detect_rate_by_method).filter(([k]) => k !== "clean").map(([k, v]) => (
                  <tr key={k}><td>{k.replace("_", " ")}</td><td className="cell">{Math.round(v * 100)}%</td></tr>
                ))}
                {det.held_out_method?.ai_splice && !det.held_out_method.ai_splice.note && (
                  <tr><td>AI splice, with no AI splices in training ({Math.round(det.held_out_method.ai_splice.false_flag_rate * 100)}% false flags)</td><td className="cell">{Math.round(det.held_out_method.ai_splice.detect_rate * 100)}%</td></tr>
                )}
              </tbody>
            </table>
          </div>
          {rob && (
            <>
              <h2>Same publishing steps, side by side</h2>
              <div className="prose"><p>We pasted one Gemini lane into each of {rob.n_panels} held out test panels, applied one publishing step, and ran our detector with its fixed flag line. The SynthID column is the hand test on a whole AI blot from the table above; SynthID could not check single bands at all.</p></div>
              <div className="scroll">
                <table className="matrix">
                  <thead><tr><th>Publishing step</th><th>SynthID, whole AI blot</th><th>Our detector, pasted AI band</th><th>Our detector, clean panels flagged</th></tr></thead>
                  <tbody>{rob.rows.map((r) => {
                    const sidRow = sid?.rows.find((x) => x.id === ROB_TO_SYNTHID[r.transform]);
                    const v = sidRow ? sidRow.synthid : "not tested";
                    return (
                      <tr key={r.transform}><td>{r.label}</td>
                        <td className={`cell ${sidRow ? cellClass(v) : "unsure"}`}>{v}</td>
                        <td className="cell">{Math.round(r.detect_rate * 100)}%</td>
                        <td className="cell">{Math.round(r.false_flag_rate * 100)}%</td></tr>
                    );
                  })}</tbody>
                </table>
              </div>
              {robNoResize.length > 0 && robResize && <p className="sub" style={{ marginTop: 10 }}>In this test our detector caught {pctR(Math.min(...robNoResize.map((r) => r.detect_rate)))} to {pctR(Math.max(...robNoResize.map((r) => r.detect_rate)))} of pasted AI bands after re-saving, JPEG, contrast changes or PDF export, with {pctR(Math.min(...robNoResize.map((r) => r.false_flag_rate)))} to {pctR(Math.max(...robNoResize.map((r) => r.false_flag_rate)))} of clean panels flagged, but only {pctR(robResize.detect_rate)} after resizing to 50%. SynthID survived all of these on whole images; on a single band it had too little to check.</p>}
            </>
          )}
        </>
      ) : <p className="sub">Detector results appear here once the model is trained.</p>}
    </section>
  );
}

const PIPELINE = [
  ["List the figures",
   "A PDF is opened and every embedded image is pulled out with its page number. Logos repeated on every page are listed once. Images too small, too large or unreadable are named in the report, never dropped silently.",
   "Every raster figure in the file.", "Vector graphics and text are not analyzed."],
  ["Read content credentials",
   "Gemini signs its downloads with a C2PA credential that says the image was made by AI. We read it and check its integrity. SynthID has no public API, so for that check we hand you the image.",
   "AI images straight from the generator.", "Any re-save strips the credential; in PDFs it is not checked."],
  ["Find reused images",
   "Every pair of figures is compared by an exact hash of the decoded pixels and by a perceptual hash confirmed with structural similarity, so a re-saved or resized copy still matches.",
   "The same image used twice in one paper.", "Copied patches inside different images, and anything in other papers."],
  ["Screen Western blots",
   "Panels that look like blots (grey, framed, rows of bands) go to a small neural network. It scores 32 pixel windows against the rest of the same blot, at the resolution it was trained on, and a noise matcher looks for copied bands.",
   "On held-out figures, 7 of 22 pasted AI bands, with about 1 in 20 unedited panels flagged.", "Most edits after resizing, most erased bands, any image type other than blots."],
  ["Write the review report",
   "Everything found, everything skipped and every limit goes into one HTML file you can save, print or send, with the flagged regions drawn on the figures.",
   "A record a reviewer can check.", "It never says a figure is authentic, and a flag is never proof of misconduct."],
];

function MethodPage() {
  return (
    <section className="page">
      <h1>What happens to a file you upload</h1>
      <ol className="pipeline">
        {PIPELINE.map(([title, body, finds, misses]) => (
          <li key={title}>
            <h3>{title}</h3>
            <p>{body}</p>
            <dl>
              <div><dt>Finds</dt><dd>{finds}</dd></div>
              <div><dt>Misses</dt><dd>{misses}</dd></div>
            </dl>
          </li>
        ))}
      </ol>
      <h2>Limits we know about</h2>
      <div className="prose">
        <p>The blot model was trained on forgeries we generated, using bands from two AI blots. A careful forger using other tools can get past it, and resizing erases most of what it relies on. A flag means look closer, never proof of misconduct.</p>
        <p>It learned from web resolution figures about 750 pixels wide. Sharper figures, such as those in publisher PDFs, are scored on a copy resized to that scale. On one published paper, scoring at full resolution flagged half of its blot panels, while the same panels at training scale produced none; the paper has no ground truth labels, so that shows a resolution effect, not that the panels are unedited.</p>
        <p>The SynthID results are a hand run case study of a few dozen uploads, not a benchmark.</p>
      </div>
      <h2>The benchmark</h2>
      <div className="prose">
        <p>Real blot panels cut from openly licensed (CC BY) papers in Europe PMC, each forged by copy move, cross splice, AI splice, intensify or erase, then pushed through a random publishing chain of contrast, resizing, JPEG and PDF steps. Every forgery ships with a pixel mask of exactly what changed. Splits are by paper, so no paper is in both training and testing. Every edit was made by us; the published figures and their authors are not implicated.</p>
      </div>
    </section>
  );
}

export default function App() {
  const [page, setPage] = useState("check");
  const [examples, setExamples] = useState([]);
  const [results, setResults] = useState(null);
  const [report, setReport] = useState(null);
  useEffect(() => {
    fetch(`${API}/api/examples`).then((r) => r.json()).then(setExamples).catch(() => {});
    fetch(`${API}/api/results`).then((r) => r.json()).then(setResults).catch(() => {});
  }, []);
  const nav = [["check", "Review images"], ["gap", "The gap"], ["method", "How it works"]];
  return (
    <>
      <div className="wrap">
        <header className="top">
          <button className="brand" type="button" onClick={() => setPage("check")}><BlotMark /> Below the Watermark</button>
          <nav className="nav" aria-label="Main">
            {nav.map(([id, label]) => (
              <button key={id} type="button" aria-current={page === id ? "page" : undefined} onClick={() => setPage(id)}>{label}</button>
            ))}
          </nav>
        </header>
        <main>
          {page === "check" && <CheckPage examples={examples} report={report} setReport={setReport} results={results} />}
          {page === "gap" && <GapPage results={results} />}
          {page === "method" && <MethodPage />}
        </main>
      </div>
      <footer className="foot"><div className="wrap">Built at Gator Hack V, UF AI Days 2026. Open source. Blot images from CC BY licensed papers via Europe PMC.</div></footer>
    </>
  );
}
