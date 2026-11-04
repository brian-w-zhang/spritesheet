#!/usr/bin/env python3
"""Pre-flight probe: can this machine create, write, recalc-read and delete a Google Sheet unattended?

Run by Brian BEFORE starting the overnight /goal run:
    python3 -I scripts/sheets_probe.py

It uses gcloud Application Default Credentials (never prints the token). On success it writes
artifacts/preflight/sheets_probe.json with timings and leaves no spreadsheet behind.
If it fails, it prints the likely fix. Stdlib only.
"""
import json, os, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "preflight", "sheets_probe.json")


def token():
    r = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("No ADC token. Run:\n  gcloud auth application-default login "
                 "--scopes=https://www.googleapis.com/auth/spreadsheets,"
                 "https://www.googleapis.com/auth/drive.file,"
                 "https://www.googleapis.com/auth/cloud-platform")
    return r.stdout.strip()


def quota_project():
    p = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    try:
        return json.load(open(p)).get("quota_project_id")
    except Exception:
        return None


def call(method, url, tok, body=None, qp=None):
    h = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    if qp:
        h["x-goog-user-project"] = qp
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:800]
        print(f"HTTP {e.code} on {method} {url.split('?')[0]}\n{msg}")
        if e.code == 403 and "insufficient" in msg.lower():
            print("\nFix: re-login with Sheets+Drive scopes (see top of this file).")
        if "quota project" in msg.lower() or "SERVICE_DISABLED" in msg or "has not been used" in msg:
            print("\nFix: gcloud auth application-default set-quota-project <PROJECT_ID>\n"
                  "     gcloud services enable sheets.googleapis.com drive.googleapis.com --project <PROJECT_ID>")
        sys.exit(1)


def main():
    tok, qp = token(), quota_project()
    t0 = time.time()
    ss = call("POST", "https://sheets.googleapis.com/v4/spreadsheets", tok,
              {"properties": {"title": "spritesheet-probe (safe to delete)"}}, qp)
    sid = ss["spreadsheetId"]
    t_create = time.time() - t0
    try:
        vals = [[1, 2, 3], [4, 5, 6], ["=SUMPRODUCT(A1:C1,A2:C2)", "=EXP(1)", "=MATCH(MAX(A2:C2),A2:C2,0)"],
                ['=SPARKLINE(1,{"charttype","bar";"max",1;"color1","#e8705b"})', "=MOD(48271*123456789,2147483647)", ""]]
        t1 = time.time()
        call("PUT", f"https://sheets.googleapis.com/v4/spreadsheets/{sid}/values/A1:C4?valueInputOption=USER_ENTERED",
             tok, {"values": vals}, qp)
        got = call("GET", f"https://sheets.googleapis.com/v4/spreadsheets/{sid}/values/A3:C4?valueRenderOption=UNFORMATTED_VALUE", tok, None, qp)
        t_rw = time.time() - t1
        res = got.get("values", [])
        expect = [32, 2.718281828459045, 3]
        ok = len(res) > 0 and abs(res[0][0] - 32) < 1e-12 and abs(res[0][1] - expect[1]) < 1e-15 and res[0][2] == 3
        lcg_ok = len(res) > 1 and len(res[1]) > 1 and res[1][1] == (48271 * 123456789) % 2147483647
    finally:
        call("DELETE", f"https://www.googleapis.com/drive/v3/files/{sid}", tok, None, qp)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    report = {"ok": bool(ok and lcg_ok), "create_s": round(t_create, 3), "write_read_s": round(t_rw, 3),
              "values": res, "quota_project_set": bool(qp), "deleted": True,
              "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(report, open(OUT, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("\nPASS: unattended Sheets access works." if report["ok"] else "\nFAIL: values did not match; see above.")


if __name__ == "__main__":
    main()
