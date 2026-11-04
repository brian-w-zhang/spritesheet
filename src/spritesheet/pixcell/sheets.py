"""Push a Pixcell Workbook to Google Sheets (service account, shared spreadsheet) and drive it.

Auth: ~/.config/spritesheet/sa-key.json; spreadsheet id: ~/.config/spritesheet/config.json -> sheet_ids[<key>].
Neither the key nor the id is ever printed or logged (errors are redacted).
Limits respected: ≤ 1 request/s, ≤ ~1.8 MB per request, retry with exponential backoff on 429/5xx.
"""
import json
import os
import time

from .. import tokenizer as T
from .workbook import Text, Workbook, a1, is_formula, rng

CFG_DIR = os.path.expanduser("~/.config/spritesheet")
KEY = os.path.join(CFG_DIR, "sa-key.json")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
API = "https://sheets.googleapis.com/v4/spreadsheets"
MAX_BYTES = 1_800_000
PIXEL = 12


def available() -> bool:
    return os.path.exists(KEY) and os.path.exists(os.path.join(CFG_DIR, "config.json"))


class SheetsError(RuntimeError):
    pass


class Client:
    def __init__(self, key="v32", min_interval=1.05):
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(KEY, scopes=SCOPES)
        self.s = AuthorizedSession(creds)
        self._sid = json.load(open(os.path.join(CFG_DIR, "config.json")))["sheet_ids"][key]
        self.min_interval = min_interval
        self._last = 0.0
        self.requests = 0
        self.retries = 0
        self.bytes_sent = 0

    # ---- transport
    def _redact(self, s: str) -> str:
        return str(s).replace(self._sid, "<spreadsheet-id>")

    def call(self, method, path, params=None, body=None, timeout=900):
        url = f"{API}/{self._sid}{path}"
        data = json.dumps(body) if body is not None else None
        for attempt in range(9):
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            self.requests += 1
            try:
                r = self.s.request(method, url, params=params, data=data, timeout=timeout,
                                   headers={"Content-Type": "application/json"} if data else None)
            except Exception as ex:  # network error: retry
                err = f"{type(ex).__name__}"
                code = None
            else:
                code = r.status_code
                if code < 300:
                    if data:
                        self.bytes_sent += len(data)
                    return r.json() if r.text else {}
                err = r.text[:500]
            if code is not None and code not in (429, 500, 502, 503, 504):
                raise SheetsError(f"HTTP {code} on {method} {path}: {self._redact(err)}")
            self.retries += 1
            time.sleep(min(64, 2 ** attempt))
        raise SheetsError(f"giving up on {method} {path} after retries: {self._redact(err)}")

    def batch_update(self, requests):
        return self.call("POST", ":batchUpdate", body={"requests": requests})

    def values_batch_update(self, data, option):
        return self.call("POST", "/values:batchUpdate", body={"valueInputOption": option, "data": data})

    def values_batch_get(self, ranges, render="UNFORMATTED_VALUE"):
        return self.call("GET", "/values:batchGet", params={"ranges": ranges, "valueRenderOption": render})

    def meta(self):
        return self.call("GET", "", params={"fields": "properties.title,sheets.properties"})


# ---------------------------------------------------------------- payload building
def _json_value(v):
    if v is None:
        return None
    if is_formula(v):
        return v
    if isinstance(v, str):
        return str(v)
    return float(v)


def _value_ranges(wb: Workbook):
    """Yield (option, sheet, r0, c0, rows) rectangles. Text cells go RAW, numbers/formulas USER_ENTERED."""
    for name, cells in wb.sheets.items():
        if not cells:
            continue
        nr, nc = wb.dims(name)
        texts = {k: v for k, v in cells.items() if isinstance(v, str) and not is_formula(v)}
        if len(texts) == len(cells):
            opt_groups = [("RAW", cells)]
        else:
            opt_groups = [("USER_ENTERED", {k: v for k, v in cells.items() if k not in texts})]
            if texts:
                opt_groups.append(("RAW", texts))
        for opt, cs in opt_groups:
            if opt == "RAW" and len(cs) != len(cells):
                # sparse text cells: one small range per vertical run in a column
                bycol = {}
                for (r, c), v in cs.items():
                    bycol.setdefault(c, []).append((r, v))
                for c, items in bycol.items():
                    items.sort()
                    run = [items[0]]
                    for it in items[1:] + [None]:
                        if it is not None and it[0] == run[-1][0] + 1:
                            run.append(it)
                            continue
                        yield opt, name, run[0][0], c, [[_json_value(v)] for _, v in run]
                        if it is not None:
                            run = [it]
                continue
            r_min = min(r for r, _ in cs)
            c_min = min(c for _, c in cs)
            c_max = max(c for _, c in cs)
            rmax = max(r for r, _ in cs)
            grid = [[None] * (c_max - c_min + 1) for _ in range(rmax - r_min + 1)]
            for (r, c), v in cs.items():
                grid[r - r_min][c - c_min] = _json_value(v)
            yield opt, name, r_min, c_min, grid


def _chunks(wb: Workbook, max_bytes=MAX_BYTES):
    """Pack value ranges into requests [(option, [ValueRange,...])] of ≤ max_bytes JSON each."""
    pending = {"RAW": [], "USER_ENTERED": []}
    size = {"RAW": 0, "USER_ENTERED": 0}
    order = []
    for opt, sheet, r0, c0, rows in _value_ranges(wb):
        i = 0
        while i < len(rows):
            # take as many rows as fit
            j, acc = i, 0
            while j < len(rows):
                rs = len(json.dumps(rows[j])) + 1
                if acc + rs > max_bytes * 0.9 and j > i:
                    break
                acc += rs
                j += 1
            part = rows[i:j]
            vr = {"range": f"'{sheet}'!{rng(r0 + i, c0, r0 + j - 1, c0 + max(len(x) for x in part) - 1)}",
                  "majorDimension": "ROWS", "values": part}
            if size[opt] + acc > max_bytes and pending[opt]:
                order.append((opt, pending[opt]))
                pending[opt], size[opt] = [], 0
            pending[opt].append(vr)
            size[opt] += acc + 100
            i = j
    for opt in ("RAW", "USER_ENTERED"):
        if pending[opt]:
            order.append((opt, pending[opt]))
    # RAW (vocab, text) first so MATCH sees the words; then formulas/values in workbook order
    return [x for x in order if x[0] == "RAW"] + [x for x in order if x[0] == "USER_ENTERED"]


# ---------------------------------------------------------------- push
def push(wb: Workbook, client: Client | None = None, log=print) -> dict:
    """Replace every tab of the shared spreadsheet with the workbook's tabs, write all cells, format Prompt."""
    cl = client or Client()
    t0 = time.time()
    meta = cl.meta()
    old = [s["properties"] for s in meta.get("sheets", [])]
    tmp_title = f"_pixcell_tmp_{int(time.time())}"
    rep = cl.batch_update([{"addSheet": {"properties": {"title": tmp_title, "index": 0}}}])
    tmp_id = rep["replies"][0]["addSheet"]["properties"]["sheetId"]
    if old:
        cl.batch_update([{"deleteSheet": {"sheetId": p["sheetId"]}} for p in old])
    adds = []
    for idx, name in enumerate(wb.sheets):
        nr, nc = wb.dims(name)
        if name == "Prompt":
            d = wb.meta["display"]
            nr, nc = max(nr, d["r0"] + d["n"] + 1), max(nc, d["c0"] + d["n"] + 1)
        adds.append({"addSheet": {"properties": {"title": name, "index": idx,
                                                 "gridProperties": {"rowCount": max(nr, 1), "columnCount": max(nc, 1),
                                                                    "hideGridlines": name == "Prompt"}}}})
    rep = cl.batch_update(adds + [{"deleteSheet": {"sheetId": tmp_id}}])
    ids = {r["addSheet"]["properties"]["title"]: r["addSheet"]["properties"]["sheetId"]
           for r in rep["replies"] if "addSheet" in r}
    t_tabs = time.time() - t0
    log(f"tabs: {len(ids)} created in {t_tabs:.1f}s")
    chunks = _chunks(wb)
    t1 = time.time()
    for k, (opt, data) in enumerate(chunks):
        cl.values_batch_update(data, opt)
        log(f"  values {k + 1}/{len(chunks)} ({opt}, {len(data)} ranges) t={time.time() - t1:.1f}s")
    t_values = time.time() - t1
    fmt = format_prompt(cl, wb, ids["Prompt"])
    return {"tabs": len(ids), "value_requests": len(chunks), "tabs_s": t_tabs, "values_s": t_values,
            "upload_s": time.time() - t0, "requests": cl.requests, "retries": cl.retries,
            "bytes_sent": cl.bytes_sent, "format": fmt}


def format_prompt(cl: Client, wb: Workbook, gid: int, pixel=PIXEL) -> dict:
    d = wb.meta["display"]
    reqs = []
    for dim, start in (("ROWS", d["r0"] - 1), ("COLUMNS", d["c0"] - 1)):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": gid, "dimension": dim, "startIndex": start, "endIndex": start + d["n"]},
            "properties": {"pixelSize": pixel}, "fields": "pixelSize"}})
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": "COLUMNS", "startIndex": 1,
                                                         "endIndex": 2},
                                               "properties": {"pixelSize": 320}, "fields": "pixelSize"}})
    reqs.append({"updateSheetProperties": {"properties": {"sheetId": gid, "gridProperties": {"hideGridlines": True}},
                                           "fields": "gridProperties.hideGridlines"}})
    # B5 "Generate" as a native checkbox (data validation BOOLEAN; still a plain TRUE/FALSE cell)
    reqs.append({"setDataValidation": {"range": {"sheetId": gid, "startRowIndex": 4, "endRowIndex": 5,
                                                 "startColumnIndex": 1, "endColumnIndex": 2},
                                       "rule": {"condition": {"type": "BOOLEAN"}, "showCustomUi": True}}})
    pr = wb.meta.get("progress")
    if pr:  # square stage-bar cells (same pixel size as the display)
        reqs.append({"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": "ROWS",
                                                             "startIndex": pr["r0"] - 1, "endIndex": pr["r0"]},
                                                   "properties": {"pixelSize": pixel}, "fields": "pixelSize"}})
    cl.batch_update(reqs)
    # read back the achieved sizes
    r0, c0, n = d["r0"], d["c0"], d["n"]
    got = cl.call("GET", "", params={"ranges": f"'Prompt'!{rng(r0, c0, r0 + 1, c0 + 1)}",
                                     "fields": "sheets.data(rowMetadata.pixelSize,columnMetadata.pixelSize)"})
    try:
        dd = got["sheets"][0]["data"][0]
        rows = [m.get("pixelSize") for m in dd.get("rowMetadata", [])]
        cols = [m.get("pixelSize") for m in dd.get("columnMetadata", [])]
    except (KeyError, IndexError):
        rows = cols = None
    return {"requested_px": pixel, "row_px": rows, "col_px": cols}


# ---------------------------------------------------------------- drive
def set_prompt_and_read(prompt, seed, client: Client | None = None, wb: Workbook | None = None, timeout=600,
                        poll=1.0) -> dict:
    """Write Prompt!B2/B3, poll Pixels until it is consistent with the new input and stops changing.

    Consistent = Tokens!A1 equals the Python clean(prompt) and Latent s0 equals the expected value (so the
    sheet has recalculated with these inputs); stable = two consecutive identical Pixels reads.
    """
    cl = client or Client()
    N = (wb.meta["N"] if wb else 32)
    t0 = time.time()
    cl.values_batch_update([{"range": "'Prompt'!B2:B3", "values": [[str(prompt)], [float(seed)]]}], "RAW")
    t_write = time.time() - t0
    want_clean = T.clean(prompt)
    want_s0 = int(abs(float(seed))) % (2147483647 - 1) + 1
    ranges = [f"'Pixels'!{rng(1, 1, N, N)}", "'Tokens'!A1", "'Latent'!A7:D7"]
    last, reads, first_ok = None, 0, None
    while True:
        got = cl.values_batch_get(ranges)
        reads += 1
        vr = got.get("valueRanges", [])
        px = vr[0].get("values", []) if vr else []
        clean = (vr[1].get("values", [[""]])[0] or [""])[0] if len(vr) > 1 else None
        lat = vr[2].get("values", [[]])[0] if len(vr) > 2 and vr[2].get("values") else []
        ok_inputs = (clean == want_clean or (want_clean == "" and clean in ("", None))) and lat and lat[0] == want_s0
        if ok_inputs and first_ok is None:
            first_ok = time.time() - t0
        if ok_inputs and px == last and len(px) == N:
            break
        last = px if ok_inputs else None
        if time.time() - t0 > timeout:
            raise SheetsError(f"timeout after {timeout}s waiting for Pixels")
        time.sleep(poll)
    return {"classes": [[int(v) if isinstance(v, (int, float)) else v for v in row] for row in px],
            "style": lat[3] if len(lat) > 3 else None, "u": lat[2] if len(lat) > 2 else None,
            "elapsed_s": time.time() - t0, "write_s": t_write, "reads": reads,
            "first_consistent_s": first_ok}


def read_ranges(cl: Client, ranges, render="UNFORMATTED_VALUE"):
    return [vr.get("values", []) for vr in cl.values_batch_get(ranges, render).get("valueRanges", [])]
