import React, { useCallback, useEffect, useRef, useState } from "react";
import { downloadReviewReport } from "./reportExport";

const API = import.meta.env.VITE_API || "";

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
const scoreFmt = (v) => Number.isFinite(Number(v)) ? Number(v).toFixed(2) : "n/a";

/* The figure on the lightbox: panels, heat, band boxes, duplicate links. */
function Stage({ fig, duplicates = [], showHeat = true, activeBand, onBand, threshold, knownEdit, knownEditLabel = "known edit", marks = [] }) {
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
      {marks.map((m, i) => (
        <div key={i} className="mark" style={{ left: pct(m.box[0], W), top: pct(m.box[1], H), width: pct(m.box[2] - m.box[0], W), height: pct(m.box[3] - m.box[1], H) }}>
          <span>{m.label}</span>
        </div>
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

function downloadDataUrl(dataUrl, name) {
  const a = document.createElement("a");
  a.href = dataUrl; a.download = name; document.body.appendChild(a); a.click(); a.remove();
}

function Check({ name, state, tone = "info", children }) {
  return (
    <div className="check">
      <h3>{name}</h3>
      <span className={`state ${tone}`}>{state}</span>
      {children}
    </div>
  );
}

function MatchPair({ left, right, leftLabel, rightLabel, children }) {
  return (
    <div className="reuse-pair">
      <div className="pair">
        <figure><img src={left} alt={leftLabel} /><figcaption>{leftLabel}</figcaption></figure>
        <figure><img src={right} alt={rightLabel} /><figcaption>{rightLabel}</figcaption></figure>
      </div>
      {children}
    </div>
  );
}

const cite = (src) => `${src.figLabel ? src.figLabel.replace(/\.$/, "") + " of " : ""}${src.pmcid}${src.year ? ` (${src.year})` : ""}`;
const howMatched = (m) => [m.scale && Math.abs(m.scale - 1) > 0.04 ? `resized ${Math.round(m.scale * 100)}%` : null, m.flipped ? "mirrored" : null].filter(Boolean).join(", ");

function WebSearch({ file, search, onRetry }) {
  const state = search.result;
  const busy = search.status === "searching";
  const matches = state?.matches || [];
  const papers = state?.papers || [];
  const compared = state?.figures_compared || 0;
  const complete = state?.status === "done";
  const headline = busy ? "Looking for similar papers…"
    : search.status === "error" ? "Online search could not finish"
    : state?.status === "offline" ? "Online search is unavailable"
    : state?.status === "no_queries" ? "More context is needed to search"
    : complete && !compared ? "No published images were compared"
    : complete && matches.length ? `${matches.length} potential source match${matches.length === 1 ? "" : "es"} found`
    : complete ? "No matching images found in the papers checked"
    : "Preparing the online search…";
  return (
    <section className="online-search" aria-labelledby="online-search-title" aria-busy={busy}>
      <div className="online-heading">
        <div><p className="section-kicker">01 / Online paper search</p><h2 id="online-search-title">Find similar papers.</h2></div>
        <span className="search-badge">{busy ? "Searching" : complete ? "Search complete" : "Automatic search"}</span>
      </div>
      <p className="search-description">We look for related open-access papers on Europe PMC, then compare their images with <strong>{file.name}</strong> to identify possible reuse.</p>
      <div className={`search-status ${search.status === "error" || state?.status === "offline" ? "search-error" : ""}`} role={search.status === "error" ? "alert" : "status"}>
        {busy && <span className="search-spinner" aria-hidden="true" />}
        <div><h3>{headline}</h3>
          {busy && <p>Looking for related research and comparing images across papers. This can take about a minute. AI screening and local matching run alongside it.</p>}
          {search.error && <p>{search.error} Other analysis results are shown below when available.</p>}
          {state?.note && <p>{state.note}</p>}
          {complete && <p>{compared} images compared from {papers.length} papers{state.seconds != null ? ` in ${state.seconds}s` : ""}{state.planner ? ` · searches planned by ${state.planner}` : ""}. {compared ? "This is a targeted search, not an exhaustive check of the literature." : "An empty search does not establish that the uploaded images are original."}</p>}
        </div>
      </div>
      {complete && matches.length > 0 && <div className="online-matches">
        {matches.map((m, i) => (
          <MatchPair key={i} left={m.crop} right={m.source_crop} leftLabel={m.figure_label} rightLabel={cite(m.source)}>
            <p className="sub" style={{ margin: "4px 0 0" }}><a href={m.source.url} target="_blank" rel="noreferrer">{m.source.title}</a>{howMatched(m) ? `. Matched ${howMatched(m)}.` : "."} Compare the captions and intended reuse before drawing conclusions.</p>
          </MatchPair>
        ))}
      </div>}
      {state?.queries?.length > 0 && <details className="more">
        <summary>Search trail: queries and papers checked</summary>
        <ul className="log">{state.queries.map((q, i) => <li key={i}><code>{q.query}</code> {q.why ? `(${q.why})` : ""}{q.error ? `: failed, ${q.error}` : q.results != null ? `: ${q.results} papers` : ""}</li>)}</ul>
        {papers.length > 0 && <ul className="log">{papers.map((p) => <li key={p.pmcid}>{p.pmcid} ({p.year}) {p.title} {p.error ? `: not read, ${p.error}` : `: ${p.figures} figures`}</li>)}</ul>}
      </details>}
      {!busy && <button className="btn ghost small" type="button" onClick={onRetry} disabled={!file}>{complete ? "Search again" : "Retry online search"}</button>}
      <p className="search-limit">A match is a lead for human review, not proof of misconduct. No match does not establish authenticity.</p>
    </section>
  );
}

function Report({ report, search }) {
  const [showHeat, setShowHeat] = useState(true);
  const [active, setActive] = useState(null);
  const searchPending = search.status === "searching";
  const webSearchResults = search.result || {
    status: search.status,
    note: search.error || (searchPending ? "The automatic online search was still running when this report was downloaded. Online results are incomplete." : "No completed online search result is available."),
    matches: [], papers: [], queries: [], figures_compared: 0,
  };
  const regionReuse = report.region_reuse || [];
  const libraryMatches = report.library_matches || [];
  const s = report.summary;
  const thr = s.threshold;
  const flagged = report.figures.flatMap((f) => f.panels.filter((p) => p.flagged).map((p) => ({ ...p, figure_label: f.label })));
  const prov = report.provenance || {};
  const imageReuse = report.image_reuse || [];
  const aiDeclared = !!s.credential_declares_ai;
  const multi = report.figures.length > 1;
  const title = aiDeclared ? "The file says it was made by AI"
    : flagged.length || imageReuse.length || s.duplicate_pairs || regionReuse.length || libraryMatches.length ? "Worth a closer look"
    : s.panels ? "Nothing flagged" : "No Western blots found";
  const parts = [
    multi ? `${report.figures.length} figures` : null,
    s.panels ? `${s.panels} blot panel${s.panels === 1 ? "" : "s"} screened` : null,
    s.panels ? `${s.flagged_panels} flagged` : null,
    imageReuse.length ? `${imageReuse.length} image${imageReuse.length === 1 ? "" : "s"} used twice` : null,
    regionReuse.length ? `${regionReuse.length} region${regionReuse.length === 1 ? "" : "s"} reused between figures` : null,
    libraryMatches.length ? `${libraryMatches.length} match${libraryMatches.length === 1 ? "" : "es"} to published figures` : null,
  ].filter(Boolean);
  const skippedImages = report.skipped_images || [];
  const skippedCounts = skippedImages.reduce((acc, x) => ({ ...acc, [x.reason]: (acc[x.reason] || 0) + 1 }), {});
  const SKIP_WORDS = { too_small: "too small to analyze", too_large: "over the 12 megapixel limit", figure_limit: "beyond the 48 figure limit", pixel_budget: "beyond the 150 megapixel total per upload", unreadable: "could not be decoded" };
  const chance = s.chance_flag_rate ?? 0.05;

  // Notes that change how to read this report, most important first.
  const notes = [];
  if (report.known_edit) {
    const k = report.known_edit, f0 = report.figures[0];
    const sx = f0 ? f0.width / k.image_width : 1, sy = f0 ? f0.height / k.image_height : 1;
    const ex0 = k.x * sx, ey0 = k.y * sy, ex1 = (k.x + k.w) * sx, ey1 = (k.y + k.h) * sy;
    const hits = (f0?.panels || []).filter((p) => p.flagged && p.x < ex1 && p.x + p.w > ex0 && p.y < ey1 && p.y + p.h > ey0);
    const others = (f0?.panels || []).filter((p) => p.flagged).length - hits.length;
    notes.push(["Our edit", <>The dashed box is the band we pasted in. The model {hits.length ? "flagged the panel that contains it" : "did not flag the panel that contains it"}{others > 0 ? `, and flagged ${others} untouched panel${others === 1 ? "" : "s"}` : ""}.</>, hits.length ? "good" : "bad"]);
  }
  if (aiDeclared) notes.push(["Content credential", <>The file carries a signed credential declaring it AI generated. It is metadata, so any re-save removes it.</>, "bad"]);
  if (skippedImages.length) notes.push(["Not checked", <>{Object.entries(skippedCounts).map(([k, n]) => `${n} image${n === 1 ? "" : "s"} ${SKIP_WORDS[k] || k}`).join("; ")}.</>, "bad"]);
  if (s.skipped_panels) notes.push(["Not checked", <>{s.skipped_panels} more blot panels after the {s.max_panels} panel limit.</>, "bad"]);
  if ((s.widespread_flag_figures || []).length) notes.push(["Many flags in one figure", <>{s.widespread_flag_figures.map((f) => f.label).join(", ")}. A shared background or scanner artifact can explain several flags at once.</>, "info"]);

  return (
    <section className="report" aria-live="polite">
      <div>
        <p className="section-kicker">02 / AI-image screening</p>
        <h2>Experimental band screening</h2>
        <p className="local-verdict">{title}</p>
        <p className="sub">{parts.join(", ")}{parts.length ? ". " : ""}A flag means look closer, never proof of misconduct.</p>
        {notes.length > 0 && <ul className="notes">{notes.map(([k, v, tone], i) => <li key={i} className={tone}><strong>{k}.</strong> {v}</li>)}</ul>}
        <div className="toolbar">
          {s.flagged_panels > 0 ? <label className="toggle"><input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Show heat</label> : <span />}
          <button className="btn ghost small" type="button" onClick={() => downloadReviewReport({ ...report, web_search: webSearchResults })}>{searchPending ? "Download local results" : "Download report"}</button>
        </div>
        {searchPending && <p className="sub export-note">Online search is still running. Its results will be included in downloads once available.</p>}
        {report.figures.map((fig) => (
          <div key={fig.id} className="figure">
            {multi && <p className="figlabel"><strong>{fig.label}</strong>
              {imageReuse.some((m) => m.a.figure === fig.id || m.b.figure === fig.id) && <span className="tag bad">used twice</span>}
              {regionReuse.some((m) => m.a.figure === fig.id || m.b.figure === fig.id) && <span className="tag bad">shares a region</span>}
              {libraryMatches.some((m) => m.figure === fig.id) && <span className="tag bad">also in a paper</span>}
              {fig.panels.some((p) => p.flagged) && <span className="tag bad">{fig.panels.filter((p) => p.flagged).length} flagged</span>}
              {fig.repeated_placements?.length ? <span className="tag">repeated on {fig.repeated_placements.length + 1} pages</span> : null}</p>}
            <div className="lightbox">
              <Stage fig={fig} duplicates={report.duplicates || []} showHeat={showHeat} threshold={thr || 0.5}
                     marks={[
                       ...regionReuse.flatMap((m) => ["a", "b"].filter((k) => m[k].figure === fig.id).map((k) => ({ box: m[k].box, label: `also in ${m[k === "a" ? "b" : "a"].label}` }))),
                       ...libraryMatches.filter((m) => m.figure === fig.id).map((m) => ({ box: m.box, label: `also in ${m.source.pmcid}` })),
                     ]}
                     knownEdit={fig.id === 0 ? report.known_edit : null} knownEditLabel="our edit"
                     activeBand={active?.key} onBand={(key, p, b) => setActive({ key, p, b })} />
              {fig.id === 0 && report.credit && <p className="credit">{report.credit}</p>}
            </div>
          </div>
        ))}
        {active && (
          <div className="detail">
            <h3>Band in panel {active.p.id + 1}: {scoreFmt(active.b.score)}</h3>
            <p className="sub" style={{ margin: 0 }}>
              {active.p.flagged && active.b.score > thr ? "Above the flag line, in a flagged panel. Look at the pixels and the loading control." :
               active.b.score > thr ? "High on its own, but its panel as a whole is below the flag line." :
               "Below the flag line. That does not prove the band is unedited."} Scores rank bands for review; they are not probabilities.
            </p>
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
                  <span>{multi ? `${p.figure_label}, panel ${p.id + 1}` : `Panel ${p.id + 1}`}, score {scoreFmt(p.score)}</span>
                  <button className="btn ghost small" type="button" onClick={() => downloadDataUrl(p.crop, `panel_${p.global_index + 1}.png`)}>Save crop</button>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
        {regionReuse.length > 0 && (
          <div className="detail reuse">
            <h3>The same blot appears twice in this paper</h3>
            {regionReuse.map((m, i) => (
              <MatchPair key={i} left={m.a.crop} right={m.b.crop} leftLabel={m.a.label} rightLabel={m.b.label}>
                <p className="sub" style={{ margin: "4px 0 0" }}>{m.inliers} matching keypoints{howMatched(m) ? `, ${howMatched(m)}` : ""}. If these are labeled as different proteins or experiments, ask for the original images.</p>
              </MatchPair>
            ))}
          </div>
        )}
        {libraryMatches.length > 0 && (
          <div className="detail reuse">
            <h3>Also in a published paper</h3>
            {libraryMatches.map((m, i) => (
              <MatchPair key={i} left={m.crop} right={m.source_crop} leftLabel={m.label} rightLabel={cite(m.source)}>
                <p className="sub" style={{ margin: "4px 0 0" }}>{m.source.url ? <a href={m.source.url} target="_blank" rel="noreferrer">{m.source.title}</a> : m.source.title}{m.source.doi ? `, doi:${m.source.doi}` : ""}. {m.inliers} matching keypoints{howMatched(m) ? `, ${howMatched(m)}` : ""}. Check whether the figure is cited.</p>
              </MatchPair>
            ))}
          </div>
        )}
        {imageReuse.length > 0 && (
          <div className="detail reuse">
            <h3>The same image appears twice</h3>
            {imageReuse.map((m, i) => {
              const a = report.figures[m.a.figure], b = report.figures[m.b.figure];
              if (!a || !b) return null;
              return <div key={i} className="reuse-pair">
                <p>{a.label} and {b.label} {m.method === "exact_pixel_duplicate" ? "have identical pixels" : "are near identical, as a re-saved or resized copy would be"}{m.matching_bands ? `, and ${m.matching_bands} of their bands match each other` : ""}. Reuse can be legitimate, so check the captions.</p>
                <div className="pair">
                  {[a, b].map((f) => <figure key={f.id}>
                    <img src={f.image} alt={f.label} />
                    <figcaption>{f.label}</figcaption>
                  </figure>)}
                </div>
              </div>;
            })}
          </div>
        )}
      </div>
      <aside className="rail" aria-label="Checks">
        <Check name="Pasted or edited bands" tone={s.flagged_panels ? "bad" : "info"}
               state={!s.panels ? "No blots" : s.flagged_panels ? `${s.flagged_panels} flagged` : "None flagged"}>
          <p>{s.panels
            ? <>Highest score {scoreFmt(s.max_splice_score)}, flag line {scoreFmt(thr)}. About 1 in {Math.round(1 / chance)} unedited panels is flagged by chance{s.panels * chance >= 0.5 ? `, so about ${Math.round(s.panels * chance)} here` : ""}.</>
            : "Runs only on panels that look like Western blots."}</p>
        </Check>
        <Check name="Reused within this paper" tone={imageReuse.length || regionReuse.length ? "bad" : "info"}
               state={!multi ? "One image" : imageReuse.length + regionReuse.length ? `${imageReuse.length + regionReuse.length} found` : "None found"}>
          <p>{multi ? "Every pair of figures is compared, region by region, so a cropped, resized or mirrored copy still matches." : "Upload a PDF to compare its figures with each other."}</p>
        </Check>
        <Check name="Published elsewhere" tone={libraryMatches.length ? "bad" : "info"}
               state={libraryMatches.length ? `${libraryMatches.length} match${libraryMatches.length === 1 ? "" : "es"}` : "None found"}>
          <p>{s.library ? `Checked against ${s.library.figures} published figures from ${s.library.papers} papers. The automatic online search above checks additional candidate papers.` : "Reference library not available."}</p>
        </Check>
        <Check name="Copied bands" tone={s.duplicate_pairs ? "bad" : "info"}
               state={!s.panels ? "No blots" : s.duplicate_pairs ? `${s.duplicate_pairs} match${s.duplicate_pairs === 1 ? "" : "es"}` : "None found"}>
          {(report.duplicates || []).slice(0, 3).map((d, i) => (
            <div key={i} className="dupe">
              {d.a.crop && d.b.crop && <div className="dupe-pair">
                <img src={d.a.crop} alt="Band" /><img src={d.b.crop} alt="Matching region" />
              </div>}
              <p>{multi ? `${d.a.figure_label} ` : ""}panel {d.a.panel_id + 1} and {multi ? `${d.b.figure_label} ` : ""}panel {d.b.panel_id + 1}{d.transform !== "none" ? `, ${d.transform.replace("_", " ")}` : ""}</p>
            </div>
          ))}
          {!s.duplicate_pairs && <p>Looks for bands whose noise matches another band exactly.</p>}
        </Check>
        <Check name="Content credentials" tone={aiDeclared ? "bad" : "info"}
               state={prov.status === "not_assessed" ? "Not checked" : prov.present ? (aiDeclared ? "Says AI" : "Present") : "None"}>
          <p>{aiDeclared ? `Declares the image AI generated${prov.signed_by ? `, signed with a certificate issued to ${prov.signed_by}` : ""}. ${prov.integrity_ok ? "Integrity checks passed." : "It could not be verified."}`
            : prov.present ? (prov.integrity_ok ? "Present, and integrity checks passed." : "Present, but could not be verified.")
            : prov.status === "not_assessed" ? "Not read inside PDFs." : "None, which is normal for published figures: any re-save removes them."}</p>
        </Check>
        <Check name="SynthID watermark" state="Check by hand">
          <p>Google's detector has no public API. Upload the image there yourself.</p>
          <div className="row"><a className="btn ghost small" href={report.synthid.url} target="_blank" rel="noreferrer">Open SynthID Detector</a></div>
        </Check>
      </aside>
    </section>
  );
}

function CheckPage({ examples, report, setReport }) {
  const [busy, setBusy] = useState(false);
  const [strict, setStrict] = useState(false);
  const [lastFile, setLastFile] = useState(null);
  const [error, setError] = useState(null);
  const [over, setOver] = useState(false);
  const [search, setSearch] = useState({ status: "idle", result: null, error: null });
  const input = useRef(null);
  const reviewId = useRef(0);
  const reviewAbort = useRef(null);
  const searchAbort = useRef(null);
  const currentFile = useRef(null);
  const tryable = examples.filter((e) => !e.hero);

  useEffect(() => () => {
    reviewId.current += 1;
    reviewAbort.current?.abort();
    searchAbort.current?.abort();
  }, []);

  const runSearch = useCallback(async (file, id) => {
    if (!file || id !== reviewId.current) return;
    searchAbort.current?.abort();
    const controller = new AbortController();
    searchAbort.current = controller;
    setSearch({ status: "searching", result: null, error: null });
    try {
      const fd = new FormData(); fd.append("file", file);
      const response = await fetch(`${API}/api/search_web`, { method: "POST", body: fd, signal: controller.signal });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "The online search failed.");
      if (!["done", "offline", "no_queries"].includes(result.status)) throw new Error("The search service returned an incomplete response.");
      if (result.status === "done" && (!Array.isArray(result.matches) || !Array.isArray(result.papers) || !Number.isFinite(result.figures_compared))) {
        throw new Error("The search service returned an incomplete response.");
      }
      if (id === reviewId.current && searchAbort.current === controller) {
        setSearch({ status: "complete", result, error: null });
      }
    } catch (e) {
      if (id === reviewId.current && searchAbort.current === controller && !controller.signal.aborted) {
        setSearch({ status: "error", result: null, error: e.message });
      }
    }
  }, []);

  const beginReview = useCallback(() => {
    const id = ++reviewId.current;
    reviewAbort.current?.abort();
    searchAbort.current?.abort();
    const controller = new AbortController();
    reviewAbort.current = controller;
    currentFile.current = null;
    setBusy(true); setError(null); setLastFile(null); setReport(null);
    setSearch({ status: "idle", result: null, error: null });
    return { id, controller };
  }, [setReport]);

  const analyzeFile = useCallback(async (file, extras, review) => {
    if (review.id !== reviewId.current) return;
    currentFile.current = file;
    setLastFile(file);
    // Start the existing online service immediately, alongside local analysis.
    void runSearch(file, review.id);
    setTimeout(() => {
      if (review.id === reviewId.current) document.getElementById("report")?.scrollIntoView({ block: "start" });
    }, 50);
    try {
      const fd = new FormData(); fd.append("file", file);
      fd.append("screening_policy", strict ? "conservative" : "benchmark");
      const response = await fetch(`${API}/api/analyze`, { method: "POST", body: fd, signal: review.controller.signal });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "The analysis failed.");
      if (review.id === reviewId.current) setReport({ ...result, ...extras });
    } catch (e) {
      if (review.id === reviewId.current && !review.controller.signal.aborted) {
        setError(`${e.message} Try a PNG, JPEG or PDF under 25 MB.`);
      }
    } finally {
      if (review.id === reviewId.current) setBusy(false);
    }
  }, [setReport, strict, runSearch]);

  const send = useCallback((file, extras = {}) => {
    return analyzeFile(file, extras, beginReview());
  }, [analyzeFile, beginReview]);

  const loadExample = async (ex) => {
    const review = beginReview();
    try {
      const response = await fetch(`${API}/api/examples/${encodeURIComponent(ex.file)}`, { signal: review.controller.signal });
      if (!response.ok) throw new Error("Could not load the example file");
      const blob = await response.blob();
      await analyzeFile(new File([blob], ex.file, { type: blob.type || "image/jpeg" }),
        { credit: ex.credit, known_edit: ex.known_edit, example_title: ex.title }, review);
    } catch (e) {
      if (review.id === reviewId.current && !review.controller.signal.aborted) {
        setError(`Could not analyze this example: ${e.message}`);
        setBusy(false);
      }
    }
  };

  return (
    <>
      <section className="hero search-hero">
        <div>
          <p className="section-kicker">Scientific figure integrity</p>
          <h1>Find similar papers.</h1>
          <p className="lede">Upload a research paper or image. We automatically look for related open-access papers and compare their images, with experimental AI screening and matching evidence in one review.</p>
          <div className="actions">
            <button className="btn" type="button" onClick={() => input.current?.click()} disabled={busy}>Upload & search papers</button>
            <button className="btn ghost" type="button" onClick={() => document.getElementById("try")?.scrollIntoView({ behavior: "smooth" })}>Try an example</button>
          </div>
        </div>
        <div className="search-intro">
          <span className="search-badge">Online search starts automatically</span>
          <h2>Related research. Shared image evidence.</h2>
          <ol>
            <li><strong>Find similar papers</strong><span>Find related open-access research on Europe PMC.</span></li>
            <li><strong>Compare images across papers</strong><span>Inspect matching regions and follow the publication links.</span></li>
            <li><strong>Review the other signals</strong><span>Review experimental AI screening and the matching checks in the sidebar.</span></li>
          </ol>
          <p>Transparent search history. Evidence for human review.</p>
        </div>
      </section>

      <section className="upload" aria-labelledby="upload-title">
        <div className={`drop ${over ? "over" : ""}`}
             onDragOver={(e) => { e.preventDefault(); setOver(true); }}
             onDragLeave={() => setOver(false)}
             onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) send(f); }}>
          <div>
            <h2 id="upload-title">Drop a paper PDF or research image</h2>
            <p>PNG, JPEG or PDF up to 25 MB. Online search starts automatically and can take about a minute.</p>
            <p className="upload-search-note">Search uses Europe PMC and, when configured, Gemini to interpret manuscript context or figure labels.</p>
          </div>
          <input ref={input} className="sr" type="file" accept="image/*,.pdf" onChange={(e) => e.target.files?.[0] && send(e.target.files[0])} />
          <button className="btn" type="button" onClick={() => input.current?.click()} disabled={busy}>{busy ? "Checking…" : "Choose a file"}</button>
        </div>
        <div className="sensitivity" role="radiogroup" aria-label="Flag sensitivity">
          <span>AI screening sensitivity</span>
          <button type="button" role="radio" aria-checked={!strict} onClick={() => setStrict(false)}>Standard</button>
          <button type="button" role="radio" aria-checked={strict} onClick={() => setStrict(true)}>Strict</button>
          <small>{strict ? "About 1 in 40 unedited panels flagged; catches about a quarter of pasted AI bands." : "About 1 in 20 unedited panels flagged; catches about a third of pasted AI bands."}</small>
        </div>
        {tryable.length > 0 && (
          <div id="try" className="tries">
            {tryable.map((ex) => (
              <button key={ex.id} type="button" className="try" onClick={() => loadExample(ex)} disabled={busy}>
                {ex.thumb && <img src={ex.thumb} alt="" />}
                <span className="t">{ex.title}</span>
                {ex.blurb && <span className="b">{ex.blurb}</span>}
              </button>
            ))}
          </div>
        )}
      </section>
      <div id="report" className="review-flow">
        {lastFile && <WebSearch file={lastFile} search={search} onRetry={() => runSearch(currentFile.current, reviewId.current)} />}
        {error && <div className="error" role="alert"><strong>Local analysis could not finish.</strong> {error} The online search is shown above independently.</div>}
        {busy && <p className="loading local-loading" role="status">{lastFile ? "AI screening and local image matching are running alongside the online search…" : "Loading the selected example…"}</p>}
        {report && <Report key={reviewId.current} report={report} search={search} />}
      </div>
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
  const det = results?.cnn_v3_metrics;
  const strict = results?.strict_line;
  const strictRates = strict?.test?.[String(strict.chosen_line)];
  const rob = results?.cnn_v3_robustness || results?.robustness;
  const fig = results?.figure_level_eval?.summary?.benchmark;
  const figStrict = results?.figure_level_eval?.summary?.conservative;
  const robNoResize = rob ? rob.rows.filter((r) => !/resize/.test(r.transform) && r.transform !== "none") : [];
  const robResize = rob ? rob.rows.find((r) => r.transform === "resize_50") : null;
  const pctR = (v) => `${Math.round(v * 100)}%`;
  // One decimal for small rates, so 2.5% does not print as 3%.
  const pct1 = (v) => v < 0.1 ? `${(v * 100).toFixed(1).replace(/\.0$/, "")}%` : pctR(v);
  const groups = [
    ["Whole AI image", (r) => !/Single|pasted|untouched/.test(r.scenario)],
    ["AI band on its own", (r) => /Single/.test(r.scenario)],
    ["AI band pasted into a real blot", (r) => /pasted/.test(r.scenario)],
    ["Control", (r) => /untouched/.test(r.scenario)],
  ];
  const METHOD = { ai_splice: "AI band pasted in", cross_splice: "Band from another blot", copy_move: "Band copied within the blot", intensify: "Band darkened", erase: "Band erased" };
  return (
    <section className="page">
      <h1>Where the watermark stops working</h1>
      <div className="prose">
        <p>We made Western blots with Gemini and checked them with Google's SynthID Detector, public since October 7, 2026, one upload at a time by hand. The content credential Gemini attaches was gone after the first re-save every time, so the watermark is the only signal left.</p>
      </div>
      {sid && (() => {
        const rowsOf = (fn) => sid.rows.filter(fn).filter((r) => r.synthid !== "pending");
        const whole = rowsOf(groups[0][1]), single = rowsOf(groups[1][1]), pasted = rowsOf(groups[2][1]);
        const steps = [
          ["whole", "A whole AI blot", whole.filter((r) => r.synthid === "detected").length, whole.length, "detected", "even after re-saving, resizing, JPEG, cropping and PDF export", "caught"],
          ["band", "One AI band on its own", single.filter((r) => r.synthid === "not enough details").length, single.length, "not enough details", "too small for the detector to judge", "unsure"],
          ["pasted", "One AI band in a published blot", pasted.filter((r) => r.synthid === "not detected").length, pasted.length, "not detected", "each band cut from a watermarked Gemini image", "missed"],
        ];
        return <ol className="ladder" aria-label="SynthID Detector results by size of the AI region">
          {steps.filter((st) => st[3] > 0).map(([kind, label, n, total, verdict, note, tone]) => (
            <li key={kind}>
              <ScaleGlyph kind={kind} />
              <span className="what">{label}</span>
              <span className={`verdict ${tone}`}>{verdict}</span>
              <span className="count">{n} of {total}, {note}</span>
            </li>
          ))}
        </ol>;
      })()}
      {sid && (
        <details className="more">
          <summary>Every upload</summary>
          <div className="scroll">
            <table className="matrix">
              <thead><tr><th>What was uploaded</th><th>SynthID Detector</th><th>Content credential</th></tr></thead>
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
          <p className="sub">A small case study, not a benchmark. Every figure is openly licensed and every edit is ours.</p>
        </details>
      )}

      <h2>Our detector on the same problem</h2>
      {det ? (
        <>
          <div className="prose"><p>A small neural network trained on our open benchmark: real blot panels from CC BY papers, edited five ways and pushed through a random publishing chain. Every number below is on {det.n_test_papers} papers it never saw, with edits we simulated.</p></div>
          <div className="stats">
            <div className="stat"><b>{Math.round(det.detect_rate_by_method?.ai_splice * 100)}%</b><span>of pasted AI bands caught</span></div>
            <div className="stat"><b>{Math.round(det.false_flag_rate_clean * 100)}%</b><span>of unedited panels flagged</span></div>
            <div className="stat"><b>{det.image_auroc}</b><span>AUROC, edited vs unedited</span></div>
            {fig && <div className="stat"><b>{fig.edited_panel_flagged} of {fig.figures}</b><span>whole figures with the edited panel flagged</span></div>}
          </div>
          <div className="prose" style={{ marginTop: 16 }}>
            {fig && <p>On {fig.figures} whole published figures, each with one Gemini band pasted into one panel and run exactly like an upload, it flagged the edited panel in {fig.edited_panel_flagged} and flagged {fig.original_figure_with_any_flag} of the {fig.figures} unedited originals.</p>}
            {strictRates && strict.test["0.5"] && <p>On {strict.n_test_papers} held-out papers, the Strict setting caught {pct1(strictRates.ai_splice_caught)} of pasted AI bands and flagged {pct1(strictRates.clean_flagged)} of unedited panels, against {pct1(strict.test["0.5"].ai_splice_caught)} and {pct1(strict.test["0.5"].clean_flagged)} for Standard.{figStrict ? ` On the ${figStrict.figures} whole figures it flagged the edited panel in ${figStrict.edited_panel_flagged} and ${figStrict.original_figure_with_any_flag ? `${figStrict.original_figure_with_any_flag} of the originals` : "none of the originals"}.` : ""}</p>}
          </div>
          {strictRates && strict.test["0.5"] && (
            <details className="more">
              <summary>Caught by type of edit</summary>
              <div className="scroll">
                <table className="matrix">
                  <thead><tr><th>Edit, on {strict.n_test_papers} held-out papers</th><th>Standard</th><th>Strict</th></tr></thead>
                  <tbody>
                    {Object.keys(METHOD).filter((k) => strictRates[`${k}_caught`] != null).map((k) => (
                      <tr key={k}><td>{METHOD[k]}</td><td className="cell">{pct1(strict.test["0.5"][`${k}_caught`])}</td><td className="cell">{pct1(strictRates[`${k}_caught`])}</td></tr>
                    ))}
                    <tr><td>Unedited panels flagged</td><td className="cell">{pct1(strict.test["0.5"].clean_flagged)}</td><td className="cell">{pct1(strictRates.clean_flagged)}</td></tr>
                  </tbody>
                </table>
              </div>
            </details>
          )}
          {rob && (
            <>
              <h2>After each publishing step</h2>
              <div className="prose"><p>We pasted one Gemini lane into each of {rob.n_panels} held-out panels and applied one publishing step. The SynthID column is the hand test on a whole AI blot; on a single band SynthID had too little to check.</p></div>
              <div className="scroll">
                <table className="matrix">
                  <thead><tr><th>Publishing step</th><th>SynthID, whole AI blot</th><th>Ours, pasted AI band</th><th>Ours, unedited flagged</th></tr></thead>
                  <tbody>{rob.rows.map((r) => {
                    const sidRow = sid?.rows.find((x) => x.id === ROB_TO_SYNTHID[r.transform]);
                    const v = sidRow ? sidRow.synthid : "not tested";
                    return (
                      <tr key={r.transform}><td>{r.label}</td>
                        <td className={`cell ${sidRow ? cellClass(v) : "unsure"}`}>{v}</td>
                        <td className="cell">{pctR(r.detect_rate)}</td>
                        <td className="cell">{pctR(r.false_flag_rate)}</td></tr>
                    );
                  })}</tbody>
                </table>
              </div>
              {robNoResize.length > 0 && robResize && <p className="sub" style={{ marginTop: 10 }}>Re-saving, JPEG, contrast and PDF export barely change what we catch. Resizing to 50% drops it to {pctR(robResize.detect_rate)}.</p>}
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
   "Gemini signs its downloads with a C2PA credential that says the image was made by AI. We read it and check its integrity. SynthID has no public API, so that check stays by hand.",
   "AI images straight from the generator.", "Any re-save strips the credential; in PDFs it is not checked."],
  ["Find duplicated images",
   "Every pair of figures is compared region by region with keypoints that survive cropping, resizing, re-saving and mirroring, so a blot reused under a different label still lines up. Every figure is also checked against a library of published figures, and an agent can search Europe PMC for the same figure in other papers: Gemini plans the searches from the title, captions or figure labels.",
   "A panel reused inside the paper, or lifted from a published one.", "About 1 in 2 single small strips (larger reused panels: 93%), and papers that are not open access."],
  ["Screen Western blots",
   "Panels that look like blots (grey, framed, rows of bands) go to a small neural network that scores each band against the rest of the same blot. A noise matcher looks for copied bands.",
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
        <p>The blot model learned from edits we simulated, with bands from two AI blots. A careful forger with other tools can get past it, and resizing erases most of what it relies on. Sharp figures from publisher PDFs are scored on a copy resized to the web size it trained on. A flag means look closer, never proof of misconduct.</p>
        <p>The SynthID results are a hand run case study, not a benchmark.</p>
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
          <div hidden={page !== "check"}><CheckPage examples={examples} report={report} setReport={setReport} /></div>
          {page === "gap" && <GapPage results={results} />}
          {page === "method" && <MethodPage />}
        </main>
      </div>
      <footer className="foot"><div className="wrap">Built at Gator Hack V, UF AI Days 2026. Open source. Blot images from CC BY licensed papers via Europe PMC.</div></footer>
    </>
  );
}
