// Collect CC BY western blot figures from Europe PMC into ONE JSON bundle.
//
// Why a browser script: Europe PMC image URLs are easy to fetch from a browser
// (CORS allowed) but our build machine could not reach them. Open any normal
// https page (https://example.com works), open DevTools > Console, paste this,
// and wait. It downloads btw_real_blots_bundleN.json (figures as base64 + metadata).
// Chrome blocks repeated automatic downloads per site, so use a site that has
// not downloaded anything yet, or allow automatic downloads for it.
//
// Then:  python -m scripts.build_benchmark --bundle btw_real_blots_bundleN.json
//
// SKIP: PMCIDs already collected (see data/real_figures/metadata.json).
// YEARS / PER_YEAR / PER_ARTICLE control how much is collected.
(async () => {
  const SKIP = new Set([]);
  const YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022];
  const PER_YEAR = 45, PER_ARTICLE = 2;
  const toB64 = (blob) => new Promise((res) => { const fr = new FileReader(); fr.onload = () => res(fr.result.split(",")[1]); fr.readAsDataURL(blob); });
  const out = [];
  for (const year of YEARS) {
    let perYear = 0;
    const q = encodeURIComponent(`(METHODS:"western blot" OR METHODS:"immunoblot") AND OPEN_ACCESS:y AND LICENSE:"cc by" AND PUB_YEAR:${year} AND SRC:MED AND HAS_FT:y`);
    const r = await fetch(`https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=${q}&format=json&pageSize=200&resultType=core`).then((r) => r.json());
    const ids = r.resultList.result
      .filter((x) => x.pmcid && !SKIP.has(x.pmcid) && (x.license || "").toLowerCase() === "cc by")
      .map((x) => ({ pmcid: x.pmcid, journal: x.journalTitle || "", year: x.pubYear, title: x.title, doi: x.doi, license: x.license }));
    for (const a of ids) {
      if (perYear >= PER_YEAR) break;
      try {
        const xml = await fetch(`https://www.ebi.ac.uk/europepmc/webservices/rest/${a.pmcid}/fullTextXML`).then((r) => r.text());
        const doc = new DOMParser().parseFromString(xml, "text/xml");
        const figs = [...doc.getElementsByTagName("fig")].filter((f) => /western blot|immunoblot/i.test(f.getElementsByTagName("caption")[0]?.textContent || ""));
        let per = 0;
        for (const f of figs) {
          if (per >= PER_ARTICLE || perYear >= PER_YEAR) break;
          const href = f.getElementsByTagName("graphic")[0]?.getAttribute("xlink:href");
          if (!href) continue;
          const fn = /\.(jpg|jpeg|gif|tif|tiff|png)$/i.test(href) ? href.replace(/\.(tif|tiff|gif)$/i, ".jpg") : href + ".jpg";
          const u = new URL("https://europepmc.org/api/fulltextRepo");
          u.searchParams.set("pmcId", a.pmcid); u.searchParams.set("type", "FILE");
          u.searchParams.set("fileName", fn); u.searchParams.set("mimeType", "image/jpeg");
          const rr = await fetch(u.href);
          if (!rr.ok) continue;
          const b = await rr.blob();
          if (b.size < 30000 || !b.type.startsWith("image")) continue;
          out.push({ file: `${a.pmcid}_${(f.getAttribute("id") || "fig").replace(/[^A-Za-z0-9]/g, "")}.jpg`, ...a,
            figLabel: f.getElementsByTagName("label")[0]?.textContent || "",
            caption: (f.getElementsByTagName("caption")[0]?.textContent || "").trim().slice(0, 600), b64: await toB64(b) });
          per++; perYear++;
          console.log("saved", out.length, a.pmcid);
        }
      } catch (e) { console.warn(a.pmcid, e); }
    }
  }
  const blob = new Blob([JSON.stringify(out)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "btw_real_blots_bundle.json";
  document.body.appendChild(link); link.click(); link.remove();
  console.log("done:", out.length, "figures");
})();
