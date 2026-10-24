"""Read C2PA Content Credentials if a file still carries them.

In our tests, Gemini downloads carry a Google signed C2PA manifest stating
"Created by Google Generative AI" and "Applied imperceptible SynthID
watermark". Any re-save strips it, so in practice this check almost always
returns "none" for published figures. We report it anyway: when it IS
present it is strong, signed evidence.
"""
from __future__ import annotations

import io
import json
import re


def read_c2pa(data: bytes, mime: str = "image/jpeg") -> dict:
    present = b"c2pa" in data and b"jumb" in data
    if not present:
        return {"present": False, "summary": "No Content Credentials (C2PA) found. "
                "Normal for published figures: any re-save strips them."}
    result = {"present": True, "signed_by": None, "ai_generated": None, "actions": [],
              "validated": False}
    try:
        import c2pa  # type: ignore
        reader = c2pa.Reader(mime, io.BytesIO(data))
        store = json.loads(reader.json())
        active = store["manifests"][store["active_manifest"]]
        result["validated"] = not store.get("validation_status")
        result["signed_by"] = (active.get("signature_info") or {}).get("issuer")
        for a in active.get("assertions", []):
            if a.get("label", "").startswith("c2pa.actions"):
                for act in a.get("data", {}).get("actions", []):
                    result["actions"].append({"action": act.get("action"),
                                              "description": act.get("description"),
                                              "source": act.get("digitalSourceType")})
    except Exception as e:  # library missing or manifest unusual: fall back to bytes
        result["reader_error"] = str(e)[:200]
        txt = data[:400000]
        result["actions"] = [{"description": m.decode(errors="ignore")}
                             for m in re.findall(rb"(Created by [ -~]{3,60}|Applied [ -~]{3,60})", txt)][:5]
        if b"Google" in txt:
            result["signed_by"] = "Google LLC"
    result["ai_generated"] = any("trainedAlgorithmicMedia" in str(a.get("source") or "") or
                                 "Generative AI" in str(a.get("description") or "")
                                 for a in result["actions"]) or (b"trainedAlgorithmicMedia" in data[:400000])
    result["summary"] = ("Content Credentials found" + (f", signed by {result['signed_by']}" if result["signed_by"] else "")
                         + (". They declare this image AI generated." if result["ai_generated"] else "."))
    return result
