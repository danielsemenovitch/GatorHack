import React, { useCallback, useEffect, useRef, useState } from "react";

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

/* The figure on the lightbox: panels, heat, band boxes, duplicate links. */
function Stage({ fig, duplicates = [], showHeat = true, activeBand, onBand, threshold }) {
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
          <img className="heat" src={p.heat} alt="" style={{ left: pct(p.x, W), top: pct(p.y, H),
               width: pct(p.w, W), height: pct(p.h, H), opacity: showHeat ? 1 : 0 }} />
          {onBand && p.bands.map((b) => {
            const key = `${p.id}-${b.id}`;
            const hot = b.score > threshold;
            return (
              <button key={key} type="button"
                className={`band ${hot ? "hot" : ""} ${activeBand === key ? "active" : ""}`}
                aria-label={`Band in panel ${p.id + 1}, suspicion ${Math.round(b.score * 100)} out of 100`}
                onClick={() => onBand(key, p, b)}
                style={{ left: pct(p.x + b.x - 2, W), top: pct(p.y + b.y - 2, H),
                         width: pct(b.w + 4, W), height: pct(b.h + 4, H) }} />
            );
          })}
        </React.Fragment>
      ))}
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
  const flagged = report.figures.flatMap((f) => f.panels.filter((p) => p.flagged));
  const prov = report.provenance;
  const title = report.verdict === "suspicious" ? "This figure needs a closer look"
    : report.verdict === "no_blots" ? "No blot panels found in this file" : "Nothing suspicious found";
  const sub = report.verdict === "suspicious"
    ? `${s.flagged_panels} of ${s.panels} blot panels have regions that do not match the rest of their blot${s.duplicate_pairs ? `, and ${s.duplicate_pairs} band pair${s.duplicate_pairs > 1 ? "s look" : " looks"} copied` : ""}.`
    : report.verdict === "no_blots" ? "Upload a figure that contains a western blot or gel."
    : `Checked ${s.panels} blot panel${s.panels === 1 ? "" : "s"}. No band level edits or copied bands stood out.`;

  return (
    <section className="report" aria-live="polite">
      <div>
        <h2>{title}</h2>
        <p className="sub">{sub}</p>
        {report.credit && <p className="credit" style={{ marginTop: -12, marginBottom: 16 }}>{report.credit}</p>}
        <div className="toolbar">
          <label className="toggle"><input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Show suspicion heat</label>
          <span className="sub" style={{ margin: 0 }}>Click a band to see why it was scored.</span>
        </div>
        {report.figures.map((fig) => (
          <div className="lightbox" key={fig.id} style={{ marginBottom: 20 }}>
            <Stage fig={fig} duplicates={report.duplicates} showHeat={showHeat} threshold={thr}
                   activeBand={active?.key} onBand={(key, p, b) => setActive({ key, p, b })} />
          </div>
        ))}
        {active && (
          <div className="detail">
            <h3>Band in panel {active.p.id + 1}: <span className="score">{Math.round(active.b.score * 100)}</span> / 100</h3>
            <Meter value={active.b.score} threshold={thr} />
            <p className="sub" style={{ margin: 0 }}>
              {active.b.score > thr ? "Above the flag line. The pixels here behave differently from the rest of this blot." : "Below the flag line. This band looks consistent with the rest of its blot."}
            </p>
            {active.p.why?.length > 0 && active.b.score > thr && (
              <ul>{[...new Set(active.p.why.map((w) => WHY[w.feature]).filter(Boolean))].slice(0, 3).map((t) => <li key={t}>{t}</li>)}</ul>
            )}
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
                  <span>Panel {p.id + 1}: <span className="score">{Math.round(p.score * 100)}</span></span>
                  <button className="btn ghost small" type="button" onClick={() => downloadDataUrl(p.crop, `panel_${p.global_index + 1}.png`)}>Download for SynthID</button>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>
      <aside className="rail" aria-label="Checks">
        <div className="check">
          <h3>Band splice detector</h3>
          <span className={`state ${s.flagged_panels ? "bad" : "ok"}`}>{s.flagged_panels ? `${s.flagged_panels} flagged` : "Clear"}</span>
          <p>Looks for bands whose noise, compression and edges disagree with the rest of their blot. Covers the edits SynthID misses.</p>
          <div className="row" style={{ display: "block" }}>
            <Meter value={s.max_splice_score} threshold={thr} />
            <p style={{ margin: 0 }}>Highest panel score {Math.round(s.max_splice_score * 100)}, flag line {Math.round(thr * 100)}.</p>
          </div>
        </div>
        <div className="check">
          <h3>Copied bands</h3>
          <span className={`state ${s.duplicate_pairs ? "bad" : "ok"}`}>{s.duplicate_pairs ? `${s.duplicate_pairs} pair${s.duplicate_pairs > 1 ? "s" : ""}` : "None"}</span>
          <p>Compares the noise inside every pair of bands, including flipped copies. Real bands never share noise.</p>
          {report.duplicates.slice(0, 3).map((d, i) => (
            <p key={i}>Panel {d.a.panel_id + 1} band {d.a.id + 1} matches panel {d.b.panel_id + 1} band {d.b.id + 1}{d.transform !== "none" ? ` (${d.transform.replace("_", " ")})` : ""}, noise match {Math.round(d.noise_similarity * 100)}%.</p>
          ))}
        </div>
        <div className="check">
          <h3>Content credentials</h3>
          <span className={`state ${prov.present ? (prov.ai_generated ? "bad" : "info") : "info"}`}>{prov.present ? (prov.ai_generated ? "AI declared" : "Found") : "None"}</span>
          <p>{prov.summary}</p>
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

function CheckPage({ examples, report, setReport }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [over, setOver] = useState(false);
  const input = useRef(null);
  const hero = examples.find((e) => e.hero) || examples[0];

  const send = useCallback(async (file) => {
    setBusy(true); setError(null);
    try {
      const fd = new FormData(); fd.append("file", file);
      const r = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "The analysis failed.");
      setReport(j);
      setTimeout(() => document.getElementById("report")?.scrollIntoView({ behavior: "smooth" }), 50);
    } catch (e) {
      setError(`${e.message} Try a PNG, JPEG or PDF under 25 MB.`);
    } finally { setBusy(false); }
  }, [setReport]);

  const loadExample = async (ex) => {
    setError(null);
    const r = await fetch(`${API}/api/examples/${ex.report}`);
    setReport(await r.json());
    setTimeout(() => document.getElementById("report")?.scrollIntoView({ behavior: "smooth" }), 50);
  };

  return (
    <>
      <section className="hero">
        <div>
          <h1>One AI band, pasted into a real blot.</h1>
          <p className="lede">
            Google's SynthID Detector catches whole AI images, even after compression and PDF export.
            {hero?.synthid === "not detected" ? " It missed this one." : " Single bands are another story."}
            {" "}Below the Watermark checks blots band by band, where scientific images actually get faked.
          </p>
          <div className="actions">
            <button className="btn" type="button" onClick={() => input.current?.click()} disabled={busy}>{busy ? "Checking…" : "Check a blot"}</button>
            {hero && <button className="btn ghost" type="button" onClick={() => loadExample(hero)}>See this example</button>}
          </div>
        </div>
        {hero?.preview && (
          <div className="lightbox">
            <figure>
              <Stage fig={hero.preview} showHeat threshold={1} />
              <figcaption>
                <span className="verdict-chip">SynthID: <span className={hero.synthid === "not detected" ? "no" : ""}>{hero.synthid === "pending" ? "check pending" : hero.synthid}</span></span>
                <span className="verdict-chip">Below the Watermark: <span className={hero.preview_verdict === "flagged" ? "no" : ""}>{hero.preview_verdict}</span></span>
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
        <p><strong>Drop a figure or a paper PDF here.</strong> We find the blot panels, score every band, look for copied bands and read any content credentials. Nothing is stored.</p>
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

function GapPage({ results }) {
  const sid = results?.synthid_manual;
  const det = results?.cnn_metrics || results?.detector_metrics;
  const base = results?.cnn_metrics ? results?.detector_metrics : null;
  const rob = results?.robustness;
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
          <div className="prose"><p>A small convolutional network trained and tested on the Below the Watermark benchmark: real blot panels from CC BY papers, forged five ways and pushed through a random publishing chain. Test papers never appear in training, and the flag line is set on separate calibration papers. This is a first baseline on a hard problem, not a finished detector.</p></div>
          <div className="stats">
            <div className="stat"><b>{det.image_auroc}</b><span>AUROC, forged vs clean panels</span></div>
            <div className="stat"><b>{Math.round(det.detect_rate_by_method?.ai_splice * 100)}%</b><span>of pasted AI bands caught</span></div>
            <div className="stat"><b>{Math.round(det.false_flag_rate_clean * 100)}%</b><span>of clean panels wrongly flagged</span></div>
            <div className="stat"><b>{det.n_test_papers}</b><span>held out papers in the test set</span></div>
          </div>
          {base && <p className="sub" style={{ marginTop: 10 }}>For comparison, a gradient boosted model on hand built forensic features reached an AUROC of {base.image_auroc} on the same task.</p>}
          <div className="scroll">
            <table className="matrix">
              <thead><tr><th>Forgery</th><th>Caught at the flag line</th></tr></thead>
              <tbody>
                {Object.entries(det.detect_rate_by_method).filter(([k]) => k !== "clean").map(([k, v]) => (
                  <tr key={k}><td>{k.replace("_", " ")}</td><td className="cell">{Math.round(v * 100)}%</td></tr>
                ))}
                {det.held_out_method?.ai_splice && (
                  <tr><td>AI splice, with no AI splices in training ({Math.round(det.held_out_method.ai_splice.false_flag_rate * 100)}% false flags)</td><td className="cell">{Math.round(det.held_out_method.ai_splice.detect_rate * 100)}%</td></tr>
                )}
              </tbody>
            </table>
          </div>
          {rob && (
            <div className="scroll">
              <table className="matrix">
                <thead><tr><th>One publishing step after the edit</th><th>Pasted AI bands caught</th><th>Clean panels flagged</th></tr></thead>
                <tbody>{rob.rows.map((r) => (
                  <tr key={r.transform}><td>{r.label}</td><td className="cell">{Math.round(r.detect_rate * 100)}%</td><td className="cell">{Math.round(r.false_flag_rate * 100)}%</td></tr>
                ))}</tbody>
              </table>
            </div>
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
          <p className="catches">Catches pasted bands, AI or not, and bands darkened or erased in place.</p>
        </div>
        <div className="layer">
          <h3>Copied bands</h3>
          <p>Two real bands can share a shape but never share noise. We compare the noise left after removing each band's shape, including flipped and rotated copies.</p>
          <p className="catches">Targets the classic copy and paste fraud, found in 3.8% of 20,621 papers screened in 2016. Works on unresized images; resizing erases most of the evidence.</p>
        </div>
      </div>
      <h2>Limits we know about</h2>
      <div className="prose">
        <p>The detector was trained on forgeries we generated. A careful forger using tools we did not simulate can still get past it, and heavy compression erases the traces it relies on. A flag means look closer, never proof of misconduct.</p>
        <p>The SynthID results are a hand run case study of a few dozen uploads, not a benchmark. We are reporting what we saw to Google.</p>
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
  const nav = [["check", "Check a blot"], ["gap", "The gap"], ["method", "How it works"]];
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
          {page === "check" && <CheckPage examples={examples} report={report} setReport={setReport} />}
          {page === "gap" && <GapPage results={results} />}
          {page === "method" && <MethodPage />}
        </main>
      </div>
      <footer className="foot"><div className="wrap">Built at Gator Hack V, UF AI Days 2026. Open source. Blot images from CC BY licensed papers via Europe PMC.</div></footer>
    </>
  );
}
