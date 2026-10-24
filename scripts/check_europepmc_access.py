"""Probe public Europe PMC figure endpoints from the CI runner.

This does not upload image data, change training data, or use SynthID.
"""
import json
import urllib.request
from xml.etree import ElementTree as ET

def grab(url):
    req=urllib.request.Request(url,headers={"User-Agent":"GatorHackResearchHackathon/1.0"})
    with urllib.request.urlopen(req,timeout=18) as response:
        data=response.read(1000000)
        return {"status":response.status,"type":response.headers.get("Content-Type"),"size":len(data)}, data

def main():
    prefix="https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4742107/fullTextXML"
    print("XML",flush=True)
    info,data=grab(prefix)
    root=ET.fromstring(data)
    ns="{http://www.w3.org/1999/xlink}href"
    figures=[(f.get("id"), (f.find(".//graphic").get(ns) if f.find(".//graphic") is not None else None))
             for f in root.findall(".//fig")][:8]
    print(json.dumps({"xml":info,"figures":figures}),flush=True)
    import urllib.parse
    href=next((href for fid,href in figures if fid=="F1" and href), None)
    if not href:
        href=next((href for _,href in figures if href),None)
    if not href:
        raise RuntimeError("no graphics found")
    fname=href if href.lower().endswith(".jpg") else href+".jpg"
    url="https://europepmc.org/api/fulltextRepo?"+urllib.parse.urlencode({
        "pmcId":"PMC4742107","type":"FILE","fileName":fname,"mimeType":"image/jpeg"})
    info,image=grab(url)
    print(json.dumps({"image_url":url,"image":info,"is_jpeg":image[:3]==b"\xff\xd8\xff"}),flush=True)

if __name__=="__main__":main()
