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
from .emit import DISPLAY_BG, render_spec  # noqa: E402  (class 0 colour: light grey, white sprites stay visible)
PIXEL = 12
DELETE_BATCH = 4


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
    if isinstance(v, bool):  # e.g. the Generate checkbox: must stay a real TRUE/FALSE for the checkbox UI
        return v
    return float(v)


INERT = "#="  # inert-formula marker for push(inert=True); must never occur in real cell text


def _value_ranges(wb: Workbook, inert: bool = False):
    """Yield (option, sheet, r0, c0, rows) rectangles. Text cells go RAW, numbers/formulas USER_ENTERED.
    inert=True: everything goes RAW and formulas are written as text "#=..." (no recalculation during upload);
    push() then switches each tab on with one findReplace "#=" → "=" (D-48)."""
    for name, cells in wb.sheets.items():
        if not cells:
            continue
        if inert:
            r_min = min(r for r, _ in cells); c_min = min(c for _, c in cells)
            r_max = max(r for r, _ in cells); c_max = max(c for _, c in cells)
            grid = [[None] * (c_max - c_min + 1) for _ in range(r_max - r_min + 1)]
            for (r, c), v in cells.items():
                grid[r - r_min][c - c_min] = ("#" + v) if is_formula(v) else _json_value(v)
            yield "RAW", name, r_min, c_min, grid
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


def _chunks(wb: Workbook, max_bytes=MAX_BYTES, inert: bool = False):
    """Pack value ranges into requests [(option, [ValueRange,...])] of ≤ max_bytes JSON each."""
    pending = {"RAW": [], "USER_ENTERED": []}
    size = {"RAW": 0, "USER_ENTERED": 0}
    order = []
    for opt, sheet, r0, c0, rows in _value_ranges(wb, inert=inert):
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
def push(wb: Workbook, client: Client | None = None, log=print, key="v32", inert: bool = False, row_tabs=()) -> dict:
    """Replace every tab of the shared spreadsheet sheet_ids[key] (v32 = v1 demo, diffusion = v2) with the
    workbook's tabs, write all cells, format Prompt. A given client wins over key."""
    cl = client or Client(key)
    t0 = time.time()
    meta = cl.meta()
    old = [s["properties"] for s in meta.get("sheets", [])]
    tmp_title = f"_pixcell_tmp_{int(time.time())}"
    rep = cl.batch_update([{"addSheet": {"properties": {"title": tmp_title, "index": 0}}}])
    tmp_id = rep["replies"][0]["addSheet"]["properties"]["sheetId"]
    # Delete the old tabs downstream-first (right-most first) in small batches: one request deleting a ~1M-formula
    # workbook's 100+ tabs did not return within 25 min on the v2 spike; 4 tabs per request take ~20 s.
    old = sorted([p for p in old if p["sheetId"] != tmp_id], key=lambda p: -p.get("index", 0))
    for i in range(0, len(old), DELETE_BATCH):
        batch = old[i:i + DELETE_BATCH]
        try:
            cl.batch_update([{"deleteSheet": {"sheetId": p["sheetId"]}} for p in batch])
        except SheetsError as ex:   # stale tab list right after a failed push: delete what still exists
            if "No sheet with id" not in str(ex):
                raise
            live = {x["properties"]["sheetId"] for x in cl.meta().get("sheets", [])}
            rest = [p for p in batch if p["sheetId"] in live]
            if rest:
                cl.batch_update([{"deleteSheet": {"sheetId": p["sheetId"]}} for p in rest])
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
    chunks = _chunks(wb, inert=inert)
    t1 = time.time()
    for k, (opt, data) in enumerate(chunks):
        cl.values_batch_update(data, opt)
        log(f"  values {k + 1}/{len(chunks)} ({opt}, {len(data)} ranges) t={time.time() - t1:.1f}s")
    t_values = time.time() - t1
    t_activate = 0.0
    if inert:  # switch formulas on tab by tab, upstream first (workbook order), so each request computes one tab
        t_activate = activate(wb, cl, ids, log, row_tabs=row_tabs)
    fmt = format_prompt(cl, wb, ids["Prompt"])
    return {"tabs": len(ids), "value_requests": len(chunks), "tabs_s": t_tabs, "values_s": t_values,
            "inert": inert, "activate_s": t_activate,
            "upload_s": time.time() - t0, "requests": cl.requests, "retries": cl.retries,
            "bytes_sent": cl.bytes_sent, "format": fmt}


def activate(wb: Workbook, cl: Client, ids=None, log=print, start=None, tries=12, row_tabs=()) -> float:
    """findReplace "#=" → "=" per formula tab, upstream first. Idempotent (an active tab has nothing to replace), so it
    also resumes a push whose activation was cut off. A 404 during a heavy recalculation (the spreadsheet is briefly
    unavailable, D-45/D-53) is waited out and retried."""
    if ids is None:
        ids = {x["properties"]["title"]: x["properties"]["sheetId"] for x in cl.meta().get("sheets", [])}
    t2 = time.time()
    names = [n for n in wb.sheets if any(is_formula(v) for v in wb.sheets[n].values())]
    if start:
        names = names[names.index(start):]
    def fr(name, rng=None):
        req = {"find": INERT, "replacement": "=", "matchCase": True}
        if rng is None:
            req["sheetId"] = ids[name]
        else:
            req["range"] = {"sheetId": ids[name], "startRowIndex": rng[0], "endRowIndex": rng[1]}
        for k in range(tries):
            try:
                cl.batch_update([{"findReplace": req}])
                return
            except SheetsError as ex:
                if not any(c in str(ex) for c in ("HTTP 404", "HTTP 503", "giving up")) or k == tries - 1:
                    raise
                log(f"  activate {name} {rng or ''}: busy ({str(ex)[:40]}), retry {k + 1} in 60s")
                time.sleep(60)

    for name in names:
        ta = time.time()
        if name in row_tabs:   # a tab that closes a long causal loop: one row per request (D-53: whole tab = 404s)
            nrows = wb.dims(name)[0]
            for r in range(nrows):
                tr = time.time()
                fr(name, (r, r + 1))
                log(f"  activate {name} row {r + 1}/{nrows}: {time.time() - tr:.1f}s")
        else:
            fr(name)
        log(f"  activate {name}: {time.time() - ta:.1f}s")
    return time.time() - t2


def format_prompt(cl: Client, wb: Workbook, gid: int, pixel=PIXEL) -> dict:
    """Square pixel cells, gridlines hidden, digits hidden, one conditional-format colour rule per class over the
    display and (v2) every filmstrip thumbnail, incl. MASK in the "unknown" colour. Layout from emit.render_spec."""
    sp = render_spec(wb)
    reqs = []

    def dim(kind, a, b, px):
        return {"updateDimensionProperties": {"range": {"sheetId": gid, "dimension": kind, "startIndex": a - 1,
                                                        "endIndex": b}, "properties": {"pixelSize": px},
                                              "fields": "pixelSize"}}
    reqs += [dim("ROWS", a, b, pixel) for a, b in sp["pixel_rows"]]
    reqs += [dim("COLUMNS", a, b, pixel) for a, b in sp["pixel_cols"]]
    reqs += [dim("ROWS", r, r, 20) for r in sorted(set(sp["label_rows"]))]
    reqs.append(dim("COLUMNS", 2, 2, 320))
    reqs.append({"updateSheetProperties": {"properties": {"sheetId": gid, "gridProperties": {"hideGridlines": True}},
                                           "fields": "gridProperties.hideGridlines"}})

    # Seamless sprite: hide the class-id digits and colour each cell by conditional formatting (one rule per class).
    def grid(r0, c0, nr, nc):
        return {"sheetId": gid, "startRowIndex": r0 - 1, "endRowIndex": r0 - 1 + nr,
                "startColumnIndex": c0 - 1, "endColumnIndex": c0 - 1 + nc}

    def rgb(h):
        h = h.lstrip("#")
        return {"red": int(h[0:2], 16) / 255, "green": int(h[2:4], 16) / 255, "blue": int(h[4:6], 16) / 255}

    ranges = [grid(r0, c0, n, n) for r0, c0, n in sp["grids"]]
    hide = {"userEnteredFormat": {"numberFormat": {"type": "NUMBER", "pattern": ";;;"}}}
    for g in ranges:
        reqs.append({"repeatCell": {"range": g, "cell": hide, "fields": "userEnteredFormat.numberFormat"}})
    for i, (k, hx) in enumerate(sp["classes"]):
        reqs.append({"addConditionalFormatRule": {"index": i, "rule": {"ranges": ranges, "booleanRule": {
            "condition": {"type": "NUMBER_EQ", "values": [{"userEnteredValue": str(k)}]},
            "format": {"backgroundColor": rgb(hx)}}}}})
    pr = wb.meta.get("progress")
    if pr:
        bar = grid(pr["r0"], pr["c0"], 1, len(pr["stages"]))
        reqs.append({"repeatCell": {"range": bar, "cell": hide, "fields": "userEnteredFormat.numberFormat"}})
        for k, hx in ((1, "#e8705b"), (0, "#e6e3ea")):
            reqs.append({"addConditionalFormatRule": {"index": 0, "rule": {"ranges": [bar], "booleanRule": {
                "condition": {"type": "NUMBER_EQ", "values": [{"userEnteredValue": str(k)}]},
                "format": {"backgroundColor": rgb(hx)}}}}})
        reqs.append(dim("ROWS", pr["r0"], pr["r0"], pixel))
    cl.batch_update(reqs)
    # read back the achieved sizes (display corner, and the last thumbnail's corner when there is a filmstrip)
    out = {"requested_px": pixel, "grids": len(sp["grids"]), "cf_rules": len(sp["classes"])}
    for tag, (r0, c0, n) in (("display", sp["grids"][0]), ("last_grid", sp["grids"][-1])):
        got = cl.call("GET", "", params={"ranges": f"'Prompt'!{rng(r0, c0, r0 + 1, c0 + 1)}",
                                         "fields": "sheets.data(rowMetadata.pixelSize,columnMetadata.pixelSize)"})
        try:
            dd = got["sheets"][0]["data"][0]
            rows = [m.get("pixelSize") for m in dd.get("rowMetadata", [])]
            cols = [m.get("pixelSize") for m in dd.get("columnMetadata", [])]
        except (KeyError, IndexError):
            rows = cols = None
        if tag == "display":
            out.update({"row_px": rows, "col_px": cols})
        else:
            out.update({"last_grid_row_px": rows, "last_grid_col_px": cols})
    return out


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
    seed_rng = (wb.meta.get("seed_range") if wb else None) or "'Latent'!A7:D7"   # v3-P: 'Seed'!A1:B1 (A = s0)
    ranges = [f"'Pixels'!{rng(1, 1, N, N)}", "'Tokens'!A1", seed_rng]
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
