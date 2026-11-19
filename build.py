import os, json, math, time, datetime as dt, requests

KEY = os.environ["FD_API_KEY"]
BASE = "https://api.football-data.org/v4"
LEAGUES = {"PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
           "BL1": "Bundesliga", "FL1": "Ligue 1"}
_last = 0

def get(path):
    """Ambil data API, dengan jeda agar tidak melewati batas 10 permintaan/menit."""
    global _last
    wait = 6.5 - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(BASE + path, headers={"X-Auth-Token": KEY}, timeout=30)
    _last = time.time()
    r.raise_for_status()
    return r.json()

def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)

def shrink(ratio, n, k=5):
    """Tarik rasio ke 1.0 (rata-rata liga) bila sampel laga masih sedikit."""
    return (ratio * n + k) / (n + k)

def probs(lh, la):
    ph = pd = pa = o25 = btts = tot = 0.0
    best = (0, 0, 0.0)
    for i in range(9):
        for j in range(9):
            p = pois(i, lh) * pois(j, la)
            tot += p
            if i > j: ph += p
            elif i == j: pd += p
            else: pa += p
            if i + j > 2: o25 += p
            if i and j: btts += p
            if p > best[2]: best = (i, j, p)
    return ph/tot, pd/tot, pa/tot, o25/tot, btts/tot, best[:2]

def decide(ph, pd, pa, form_h, form_a):
    top = max(ph, pd, pa)
    fav_home = ph >= pa
    fav, opp = (form_h, form_a) if fav_home else (form_a, form_h)
    reasons = []
    if top >= 0.62: reasons.append("peluang unggulan tinggi")
    if pd >= 0.28: reasons.append("risiko seri tinggi")
    if fav >= opp: reasons.append("form unggulan sejalan dengan peluang")
    else: reasons.append("form unggulan lebih buruk dari lawan")
    if top >= 0.62 and pd < 0.25 and fav >= opp: label = "Prioritas"
    elif top < 0.45 or pd >= 0.30: label = "Hindari"
    else: label = "Pertimbangkan"
    return label, reasons

def analyse(code, name):
    st = get(f"/competitions/{code}/standings")["standings"]
    tab = {s["type"]: {t["team"]["id"]: t for t in s["table"]} for s in st}
    home = tab.get("HOME") or tab["TOTAL"]
    away = tab.get("AWAY") or tab["TOTAL"]
    lh = sum(t["goalsFor"] for t in home.values()) / max(1, sum(t["playedGames"] for t in home.values())) or 1.5
    la = sum(t["goalsFor"] for t in away.values()) / max(1, sum(t["playedGames"] for t in away.values())) or 1.2

    ms = get(f"/competitions/{code}/matches")["matches"]
    pts = {}
    for m in sorted((m for m in ms if m["status"] == "FINISHED"), key=lambda m: m["utcDate"]):
        ft = m["score"]["fullTime"]
        if ft["home"] is None: continue
        h, a = m["homeTeam"]["id"], m["awayTeam"]["id"]
        ph_, pa_ = (3, 0) if ft["home"] > ft["away"] else (1, 1) if ft["home"] == ft["away"] else (0, 3)
        pts.setdefault(h, []).append(ph_); pts.setdefault(a, []).append(pa_)
    form = lambda t: sum(pts.get(t, [])[-5:])

    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for m in ms:
        if m["status"] not in ("SCHEDULED", "TIMED"): continue
        t = dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        if not (now <= t <= now + dt.timedelta(days=8)): continue
        hid, aid = m["homeTeam"]["id"], m["awayTeam"]["id"]
        if hid not in home or aid not in away: continue
        H, A = home[hid], away[aid]
        nh, na = max(1, H["playedGames"]), max(1, A["playedGames"])
        att_h = shrink((H["goalsFor"] / nh) / lh, nh)
        def_h = shrink((H["goalsAgainst"] / nh) / la, nh)
        att_a = shrink((A["goalsFor"] / na) / la, na)
        def_a = shrink((A["goalsAgainst"] / na) / lh, na)
        xh, xa = lh * att_h * def_a, la * att_a * def_h
        ph, pd, pa, o25, btts, sc = probs(xh, xa)
        label, why = decide(ph, pd, pa, form(hid), form(aid))
        out.append({"lg": name, "t": m["utcDate"], "h": m["homeTeam"]["shortName"] or m["homeTeam"]["name"],
                    "a": m["awayTeam"]["shortName"] or m["awayTeam"]["name"],
                    "ph": ph, "pd": pd, "pa": pa, "o25": o25, "btts": btts, "sc": sc,
                    "xh": xh, "xa": xa, "fh": form(hid), "fa": form(aid),
                    "label": label, "why": why})
    return out

data = []
for code, name in LEAGUES.items():
    try:
        data += analyse(code, name)
    except Exception as e:
        print("Gagal", code, e)
data.sort(key=lambda x: x["t"])

TEMPLATE = open("template.html", encoding="utf-8").read()
updated = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%d %b %Y %H:%M WIB")
html = TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")).replace("__UPDATED__", updated)
open("index.html", "w", encoding="utf-8").write(html)
print("Selesai:", len(data), "pertandingan")
