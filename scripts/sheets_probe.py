#!/usr/bin/env python3
"""Pre-flight probe: can the agent write formulas to a Google Sheet and read computed values back, unattended?

Auth: a service account key at ~/.config/spritesheet/sa-key.json (outside the repo; never printed).
Target: a spreadsheet owned by Brian and shared with the service account as Editor. Its id is in
~/.config/spritesheet/config.json as {"sheet_ids": {"probe": "<id>", "v32": "<id>", ...}}, or passed as argv[1].

Run:  uv run python scripts/sheets_probe.py [SPREADSHEET_ID]
Writes artifacts/preflight/sheets_probe.json. It adds a temporary tab and deletes it afterwards.
Also reports whether the service account can create its own spreadsheet (expected to fail: no Drive quota).
"""
import json, os, sys, time

from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.expanduser("~/.config/spritesheet")
KEY = os.path.join(CFG_DIR, "sa-key.json")
OUT = os.path.join(ROOT, "artifacts", "preflight", "sheets_probe.json")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive.file"]
API = "https://sheets.googleapis.com/v4/spreadsheets"


def session():
    if not os.path.exists(KEY):
        sys.exit(f"Missing service-account key at {KEY}")
    creds = service_account.Credentials.from_service_account_file(KEY, scopes=SCOPES)
    return AuthorizedSession(creds), creds.service_account_email


def sheet_id():
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        return json.load(open(os.path.join(CFG_DIR, "config.json")))["sheet_ids"]["probe"]
    except Exception:
        sys.exit("No spreadsheet id. Share a sheet with the service account and pass its id, or put it in "
                 "~/.config/spritesheet/config.json as {\"sheet_ids\": {\"probe\": \"<id>\"}}")


def check(r, what):
    if r.status_code >= 300:
        print(f"{what}: HTTP {r.status_code}\n{r.text[:600]}")
        if r.status_code in (403, 404):
            print("\nFix: share the spreadsheet with the service account email as Editor.")
        sys.exit(1)
    return r.json() if r.text else {}


def main():
    s, email = session()
    sid = sheet_id()
    report = {"service_account": email, "spreadsheet_id": sid, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    meta = check(s.get(f"{API}/{sid}?fields=properties.title,sheets.properties"), "read spreadsheet")
    report["title"] = meta["properties"]["title"]
    tab = f"probe_{int(time.time())}"
    t0 = time.time()
    add = check(s.post(f"{API}/{sid}:batchUpdate", json={"requests": [{"addSheet": {"properties": {"title": tab}}}]}), "add tab")
    gid = add["replies"][0]["addSheet"]["properties"]["sheetId"]
    try:
        vals = [[1, 2, 3], [4, 5, 6],
                ["=SUMPRODUCT(A1:C1,A2:C2)", "=EXP(1)", "=MATCH(MAX(A2:C2),A2:C2,0)"],
                ['=SPARKLINE(1,{"charttype","bar";"max",1;"color1","#e8705b"})', "=MOD(48271*123456789,2147483647)", "=TANH(0.5)"]]
        check(s.put(f"{API}/{sid}/values/{tab}!A1:C4", params={"valueInputOption": "USER_ENTERED"}, json={"values": vals}), "write")
        got = check(s.get(f"{API}/{sid}/values/{tab}!A3:C4", params={"valueRenderOption": "UNFORMATTED_VALUE"}), "read")
        # square cells: set 8 rows and 8 columns to 12 px
        check(s.post(f"{API}/{sid}:batchUpdate", json={"requests": [
            {"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": d, "startIndex": 4, "endIndex": 12},
                                           "properties": {"pixelSize": 12}, "fields": "pixelSize"}} for d in ("ROWS", "COLUMNS")]}),
              "square cells")
        dims = check(s.get(f"{API}/{sid}", params={"ranges": f"{tab}!E5:F6", "fields": "sheets.data(rowMetadata.pixelSize,columnMetadata.pixelSize)"}), "read dims")
        report["write_read_s"] = round(time.time() - t0, 3)
        res = got.get("values", [])
        report["values"] = res
        report["dims"] = dims
        import math
        report["checks"] = {
            "sumproduct": res[0][0] == 32,
            "exp_exact": res[0][1] == math.exp(1),
            "match": res[0][2] == 3,
            "lcg_exact": res[1][1] == (48271 * 123456789) % 2147483647,
            "tanh_exact": res[1][2] == math.tanh(0.5),
        }
    finally:
        check(s.post(f"{API}/{sid}:batchUpdate", json={"requests": [{"deleteSheet": {"sheetId": gid}}]}), "delete tab")
    r = s.post(API, json={"properties": {"title": "SpriteSheet probe (safe to delete)"}})
    report["sa_can_create_spreadsheets"] = r.status_code < 300
    if r.status_code < 300:
        new = r.json()["spreadsheetId"]
        s.delete(f"https://www.googleapis.com/drive/v3/files/{new}")
    else:
        report["sa_create_error"] = r.text[:300]
    report["ok"] = all(report["checks"].values())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "dims"}, indent=2))
    print("\nPASS: unattended Sheets access works." if report["ok"] else "\nFAIL: see checks above.")


if __name__ == "__main__":
    main()
