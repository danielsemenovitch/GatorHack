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
function Stage({ fig, duplicates = [], showHeat = true, activeBand, onBand, threshold, knownEdit }) {
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
          <span>known edit</span>
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
  const SKIP_WORDS = { too_small: "too small to analyze", too_large: "over the 12 megapixel limit", figure_limit: "beyond the 48 figure limit", unreadable: "could not be decoded" };
  const sub = imageReuse.length
    ? `${imageReuse.length} potential whole-image reuse pair${imageReuse.length === 1 ? "" : "s"} found. Image reuse can be legitimate; compare captions and methods.`
    : s.panels ? `${s.panels} Western blot panel${s.panels === 1 ? "" : "s"} screened with an experimental classifier. A negative result does not establish authenticity.`
    : "No confidently identified Western blot panels. General image inventory remains available; the blot classifier was not run on other image types.";

  return (
    <section className="report" aria-live="polite">
      <div>
        <h2>{title}</h2>
        <p className="sub">{sub} These are review candidates, not independent findings of manipulation.</p>
        {s.policy_note && s.panels > 0 && <p className="sub"><strong>Screening policy:</strong> {s.policy_note}</p>}
        {!!s.skipped_panels && <p role="status"><strong>Incomplete screening:</strong> {s.skipped_panels} candidate blot panels were skipped after the {s.max_panels}-panel processing limit. This does not mean those panels are clean.</p>}
        {(s.widespread_flag_figures || []).length > 0 && <p role="status"><strong>Concentrated alerts:</strong> Many panels in {s.widespread_flag_figures.map(f => f.label).join(", ")} were flagged. Shared background, scanner or publishing artifacts may be responsible; inspect the entire figure before drawing conclusions.</p>}
        {report.credit && <p className="credit" style={{ marginBottom: 16 }}>{report.credit}</p>}
        {aiDeclared && <p role="status"><strong>Provenance:</strong> {prov.summary} This is a declaration attached to the file, not a pixel analysis; any re-save removes it.</p>}
        {report.known_edit && (() => {
          // Does any flagged panel actually contain the region we edited?
          const k = report.known_edit, f0 = report.figures[0];
          const sx = f0 ? f0.width / k.image_width : 1, sy = f0 ? f0.height / k.image_height : 1;
          const ex0 = k.x * sx, ey0 = k.y * sy, ex1 = (k.x + k.w) * sx, ey1 = (k.y + k.h) * sy;
          const hits = (f0?.panels || []).filter((p) => p.flagged && p.x < ex1 && p.x + p.w > ex0 && p.y < ey1 && p.y + p.h > ey0);
          const others = (f0?.panels || []).filter((p) => p.flagged).length - hits.length;
          const msg = hits.length ? "the model flagged the panel that contains the edit" :
            "the model did not flag the panel that contains the edit";
          return <p role="status"><strong>Demo ground truth:</strong> the dashed outline marks the region we edited. At this setting {msg}{others > 0 ? `, and flagged ${others} untouched panel${others === 1 ? "" : "s"} (false positive${others === 1 ? "" : "s"})` : ""}.</p>;
        })()}
        {skippedImages.length > 0 && <p role="status"><strong>Not analyzed:</strong> {Object.entries(skippedCounts).map(([k, n]) => `${n} image${n === 1 ? "" : "s"} ${SKIP_WORDS[k] || k}`).join("; ")}. Those images were not checked at all.</p>}
        {!!s.resized_for_analysis && <p className="sub">{s.resized_for_analysis} figure{s.resized_for_analysis === 1 ? " is" : "s are"} larger than the blot model's training resolution (figures about {s.analysis_width} px wide), so {s.resized_for_analysis === 1 ? "its" : "their"} blot panels were scored on copies resized to that scale. Without this, scores on high resolution figures run high on unedited blots. Edits made at full resolution may be harder to detect after resizing.</p>}
        <div className="toolbar">
          <label className="toggle"><input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} disabled={!s.panels} /> Show blot heat</label>
          <button className="btn ghost small" type="button" onClick={() => downloadReviewReport(report)}>Download review report (HTML)</button>
        </div>
        <p className="sub">Heat is drawn only on panels above the review threshold. It visualizes overlapping 32-pixel CNN patches, not pixel-accurate locations or boundaries of altered bands.</p>
        {report.figures.map((fig) => (
          <div key={fig.id} style={{ marginBottom: 20 }}>
            <p className="sub" style={{ marginBottom: 6 }}><strong>{fig.label}</strong> — {fig.native_width || fig.width} × {fig.native_height || fig.height} px
              {fig.analysis_scale && fig.analysis_scale < 1 && fig.panels.length ? ` (blot panels scored at ${Math.round(fig.analysis_scale * 100)}% size, the model's training resolution)` : ""}
              {fig.repeated_placements?.length ? ` · also drawn at ${fig.repeated_placements.length} more place${fig.repeated_placements.length === 1 ? "" : "s"} (same embedded object, e.g. a logo)` : ""}
              {fig.blot_screen === "experimental" ? " · Experimental blot screen" : fig.blot_screen === "partial" ? " · Partially screened" : fig.blot_screen === "not_assessed_limit" ? " · Not assessed (limit reached)" : " · Blot model not applied"}</p>
            <div className="lightbox">
              <Stage fig={fig} duplicates={report.duplicates || []} showHeat={showHeat} threshold={thr || 0.5}
                     knownEdit={fig.id === 0 ? report.known_edit : null}
                     activeBand={active?.key} onBand={(key, p, b) => setActive({ key, p, b })} />
            </div>
          </div>
        ))}
        {imageReuse.length > 0 && (
          <div className="detail" style={{ marginBottom: 18 }}>
            <h3>Potential whole-image reuse — check the context</h3>
            {imageReuse.map((m, i) => {
              const a = report.figures[m.a.figure], b = report.figures[m.b.figure];
              if (!a || !b) return null;
              return <div key={i} style={{ marginBottom: 16 }}>
                <p><strong>{m.method === "exact_pixel_duplicate" ? "Identical image pixels" : "Near-duplicate images"}</strong>:
                  {" "}{a.label} ↔ {b.label}. A match does not mean misconduct.</p>
                <div style={{ display: "flex", gap: 14, alignItems: "start" }}>
                  {[a, b].map((f) => <figure key={f.id} style={{ width: "50%", margin: 0 }}>
                    <img src={f.image} alt={f.label} style={{ width: "100%", maxHeight: 220, objectFit: "contain" }} />
                    <figcaption>{f.label}</figcaption>
                  </figure>)}
                </div>
                <p className="sub">{m.note} {m.limitations}</p>
              </div>;
            })}
          </div>
        )}
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
          <p>Experimental model for blot-like panels only. Stricter settings reduce review alerts but can miss more real edits. The displayed alert threshold is not an accuracy guarantee.</p>
          {s.panels > 0 && thr != null && <div className="row" style={{ display: "block" }}>
            <p style={{ margin: 0 }}>Model {s.model_version || report.cnn_version || "not recorded"}: highest panel anomaly index <strong>{scoreFmt(s.max_splice_score)}</strong>; review threshold <strong>{scoreFmt(thr)}</strong>. These are uncalibrated model indices, not percentages or probabilities.</p>
          </div>}
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
  const [busy, setBusy] = useState(false);
  const [blotMode, setBlotMode] = useState("auto");
  const [cnnVersion, setCnnVersion] = useState("v3");
  const [screeningPolicy, setScreeningPolicy] = useState("conservative");
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
          <h1>Scientific images deserve a closer look.</h1>
          <p className="lede">
            Upload a scientific figure or paper PDF to inspect available image provenance and potential image reuse.
            We also provide experimental band-level screening for Western blots, inspired by our studies of where AI watermark checks become inconclusive.
          </p>
          <div className="actions">
            <button className="btn" type="button" onClick={() => input.current?.click()} disabled={busy}>{busy ? "Reviewing…" : "Review a scientific image"}</button>
            {hero && <button className="btn ghost" type="button" onClick={() => loadExample(hero)}>See this example</button>}
          </div>
        </div>
        {hero?.preview && (
          <div className="lightbox">
            <figure>
              <Stage fig={{ ...hero.preview, panels: [] }} showHeat={false} threshold={1} />
              <figcaption>
                <span className="verdict-chip">SynthID: <span className={hero.synthid === "not detected" ? "no" : ""}>{hero.synthid === "pending" ? "check pending" : hero.synthid}</span></span>
                <span className="verdict-chip">CNN: <span>Run example for current analysis</span></span>
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
        <p><strong>Drop an image or research paper PDF here.</strong> We inventory embedded raster figures, compare whole images for reuse, check available provenance, and screen blot-like panels experimentally. Results are evidence for human review, not authenticity verdicts. The server does not intentionally retain uploads.</p>
        <div className="controls">
          <span className="field">
            <label htmlFor="blot-mode">Western blot screen</label>
            <select id="blot-mode" value={blotMode} onChange={(e) => setBlotMode(e.target.value)}>
              <option value="auto">Automatic (conservative)</option>
              <option value="blot">I confirm this image includes Western blots</option>
              <option value="off">Skip blot-specific screening</option>
            </select>
          </span>
          <span className="field">
            <label htmlFor="cnn-version">Experimental CNN</label>
            <select id="cnn-version" value={cnnVersion} onChange={(e) => setCnnVersion(e.target.value)}>
              <option value="v3">V3 — larger training set (experimental)</option>
              <option value="v2">V2 — legacy research scores (more sensitive)</option>
            </select>
          </span>
          <span className="field">
            <label htmlFor="screening-policy">Alert sensitivity</label>
            <select id="screening-policy" value={screeningPolicy} onChange={(e) => setScreeningPolicy(e.target.value)}>
              <option value="conservative">Conservative (fewer alerts, default)</option>
              <option value="benchmark">Research benchmark threshold (more alerts)</option>
            </select>
          </span>
          {op && cnnVersion === "v3" && <span className="sub" style={{ fontSize: 13, margin: 0, flexBasis: "100%" }}>
            Measured on held-out papers with simulated edits: conservative caught {Math.round(100 * op["ai_splice_caught@0.9"])}% of pasted AI bands
            ({Math.round(100 * op["clean_flagged@0.9"])}% of clean panels flagged); the research threshold caught {Math.round(100 * op["ai_splice_caught@0.5"])}%
            ({Math.round(100 * op["clean_flagged@0.5"])}% of clean panels flagged).</span>}
          {cnnVersion === "v2" && screeningPolicy === "conservative" &&
            <span className="sub" style={{ fontSize: 13, margin: 0 }}>V2 scores saturate near 1.0; it keeps its legacy threshold. Use V3 for conservative screening.</span>}
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
  const pasted = sid?.rows.filter((r) => /pasted/.test(r.scenario)) || [];
  const pastedMissed = pasted.filter((r) => r.synthid === "not detected").length;
  const pastedDone = pasted.filter((r) => r.synthid !== "pending").length;
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
        <p>We made western blots with Gemini's image tools and checked them with SynthID Detector, which Google opened to the public on October 7, 2026. Every upload was done by hand, one fixed scenario at a time.</p>
        <p><strong>Whole AI images were caught every time,</strong> through re-saving, resizing, heavy JPEG, cropping, contrast changes and a PDF round trip. The visible content credential did not survive a single re-save. <strong>Single AI bands were too small for it to check.</strong>{" "}
          {pastedDone > 0
            ? <>An AI band pasted into a published blot came back as not detected in <strong>{pastedMissed} of {pastedDone}</strong> tests.</>
            : <>We are re-running the pasted band test on openly licensed figures; those rows show as pending until then.</>}
        </p>
      </div>
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
                      <td className={`cell ${r.c2pa === "present" ? "caught" : r.c2pa === "stripped" ? "missed" : "unsure"}`}>{r.c2pa}</td>
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
          <div className="prose"><p>{isV3 ? "CNN v3, the app's default model: a" : "A"} small convolutional network trained and tested on the Below the Watermark benchmark: real blot panels from CC BY papers, forged five ways and pushed through a random publishing chain. Test papers never appear in training, and this flag line (the research benchmark threshold) is set on separate calibration papers. The app's default conservative setting uses a stricter line, so it flags fewer panels and catches fewer edits than shown here. All edits are simulated; this is a first baseline on a hard problem, not a validated detector.</p></div>
          <div className="stats">
            <div className="stat"><b>{det.image_auroc}</b><span>AUROC, forged vs clean panels</span></div>
            <div className="stat"><b>{Math.round(det.detect_rate_by_method?.ai_splice * 100)}%</b><span>of pasted AI bands caught</span></div>
            <div className="stat"><b>{Math.round(det.false_flag_rate_clean * 100)}%</b><span>of clean panels wrongly flagged</span></div>
            <div className="stat"><b>{det.n_test_papers}</b><span>held out papers in the test set</span></div>
          </div>
          {results?.audit_resolution?.benchmark?.native && (() => {
            const op = results.audit_resolution.benchmark.native, n = results.audit_resolution.benchmark.n;
            return <p className="sub" style={{ marginTop: 10 }}><strong>The app's default conservative line (0.90) is much stricter:</strong> on {results.audit_resolution.benchmark.n_test_papers} of these held-out papers ({n.clean} clean and {n.ai_splice} AI splice images) it caught {Math.round(100 * op["ai_splice_caught@0.9"])}% of pasted AI bands and flagged {Math.round(100 * op["clean_flagged@0.9"])}% of clean panels, versus {Math.round(100 * op["ai_splice_caught@0.5"])}% and {Math.round(100 * op["clean_flagged@0.5"])}% at the research line.</p>;
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

function MethodPage() {
  return (
    <section className="page">
      <h1>Three checks, each covering what the others miss</h1>
      <div className="layers">
        <div className="layer">
          <h3>Content credentials and SynthID</h3>
          <p>Gemini signs its downloads with a C2PA credential and hides a SynthID watermark in the pixels. We read the credential automatically. SynthID has no public API, so we hand you the panel to check.</p>
          <p className="catches">Catches whole AI images. The credential dies on any re-save; the watermark survives publishing.</p>
        </div>
        <div className="layer">
          <h3>Band splice detector</h3>
          <p>Every panel is cut into small overlapping windows. A small neural network looks at each window's pixels and noise, normalized against the rest of the same blot, and scores how out of place it looks. Real bands from clean blots are used as hard negatives so it cannot simply flag anything that looks like a band.</p>
          <p className="catches">Aimed at pasted, darkened or erased bands. On held out papers it catches a minority of our simulated edits (see The gap), misses most erased bands, and is weakest after resizing.</p>
        </div>
        <div className="layer">
          <h3>Copied bands</h3>
          <p>Two real bands can share a shape but should not share fine noise. We compare the noise left after removing each band's shape, including flipped and rotated copies.</p>
          <p className="catches">Targets copy and paste duplication, the kind of problem Bik and colleagues found in 3.8% of 20,621 papers in 2016 (all image types). At the app's strict setting it finds only a small fraction of our simulated copies, and resizing erases most of the evidence.</p>
        </div>
      </div>
      <h2>Limits we know about</h2>
      <div className="prose">
        <p>The detector was trained on forgeries we generated. A careful forger using tools we did not simulate can still get past it, and heavy compression erases the traces it relies on. A flag means look closer, never proof of misconduct.</p>
        <p>The SynthID results are a hand run case study of a few dozen uploads, not a benchmark.</p>
        <p>The blot model learned from web resolution figures about 750 pixels wide. Larger figures, such as those embedded in publisher PDFs, are screened on a copy resized to that width; on one real paper, screening at full resolution flagged half of its blot panels at the research threshold, a sign of that mismatch rather than of editing.</p>
      </div>
      <h2>The benchmark</h2>
      <div className="prose">
        <p>Real blot panels cut from openly licensed (CC BY) papers in Europe PMC, each forged by copy move, cross splice, AI splice, intensify or erase, then pushed through a random publishing chain of contrast, resizing, JPEG and PDF steps. Every forgery ships with a pixel mask of exactly what changed. Splits are by paper, so no paper is in both training and testing.</p>
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
