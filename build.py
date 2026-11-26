import os, sys, json, math, time, datetime as dt, requests

KEY = os.environ.get("FD_API_KEY", "")
BASE = "https://api.football-data.org/v4"
LEAGUES = {"PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
           "BL1": "Bundesliga", "FL1": "Ligue 1", "CL": "Liga Champions",
           "ELC": "Championship", "DED": "Eredivisie", "PPL": "Primeira Liga",
           "BSA": "Brasileirão"}
HIST = "history.json"
HOME_ADV = 60      # keunggulan kandang (poin Elo)
ELO_W = 0.4        # bobot Elo saat digabung dengan model gol
CORNER_BASE = (5.4, 4.6)   # asumsi rata-rata korner kandang/tandang (heuristik, bukan data)
_last = 0.0


def get(path):
    """Ambil data API dengan jeda agar tidak melewati batas 10 permintaan/menit."""
    global _last
    for attempt in range(2):
        wait = 6.5 - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        r = requests.get(BASE + path, headers={"X-Auth-Token": KEY}, timeout=30)
        _last = time.time()
        if r.status_code == 429 and attempt == 0:
            time.sleep(35)
            continue
        r.raise_for_status()
        return r.json()


def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def shrink(ratio, n, k=5):
    """Tarik rasio ke 1.0 (rata-rata liga) bila sampel laga masih sedikit."""
    return (ratio * n + k) / (n + k)


def grid(lh, la, n=10):
    pa = [pois(i, lh) for i in range(n)]
    pb = [pois(j, la) for j in range(n)]
    g = [[x * y for y in pb] for x in pa]
    t = sum(map(sum, g))
    return [[x / t for x in r] for r in g]


def summarize(g):
    ph = pd = pa = o25 = btts = 0.0
    best = (0, 0, 0.0)
    for i, r in enumerate(g):
        for j, p in enumerate(r):
            if i > j: ph += p
            elif i == j: pd += p
            else: pa += p
            if i + j > 2: o25 += p
            if i and j: btts += p
            if p > best[2]: best = (i, j, p)
    return ph, pd, pa, o25, btts, best[:2]


def fit_supremacy(lh, la, target_ph):
    """Geser selisih gol harapan agar peluang menang kandang sama dengan hasil gabungan Elo."""
    lo, hi = -1.5, 1.5
    for _ in range(30):
        mid = (lo + hi) / 2
        p = summarize(grid(max(0.05, lh + mid), max(0.05, la - mid)))[0]
        if p < target_ph: lo = mid
        else: hi = mid
    s = (lo + hi) / 2
    return max(0.05, lh + s), max(0.05, la - s)


def cover(g, side, line):
    """Peluang menang, push (taruhan kembali), kalah untuk handicap pada sisi tertentu."""
    w = pu = l = 0.0
    for i, r in enumerate(g):
        for j, p in enumerate(r):
            v = (i - j if side == "h" else j - i) + line
            if v > 1e-9: w += p
            elif v < -1e-9: l += p
            else: pu += p
    return w, pu, l


# ---------- Model korner (heuristik buatan sendiri, tanpa data korner asli) ----------
def nb_pmf(mu, vf, n=40):
    """Distribusi binomial negatif dengan varians = vf * rata-rata (korner lebih melebar dari Poisson)."""
    r = mu / (vf - 1.0)
    p = r / (r + mu)
    out = [math.exp(math.lgamma(x + r) - math.lgamma(r) - math.lgamma(x + 1)
                    + r * math.log(p) + x * math.log(1 - p)) for x in range(n)]
    t = sum(out)
    return [x / t for x in out]


def tail(pmf, line):
    return sum(p for x, p in enumerate(pmf) if x > line)


def lean_pick(rows, need=0.60):
    """Pilih garis paling dekat 50% yang masih punya kecondongan minimal `need` (garis ekstrem tidak informatif)."""
    ok = [(max(o, 1 - o), ln, o) for ln, o in rows if max(o, 1 - o) >= need]
    if not ok:
        return None
    lean, ln, o = min(ok)
    return {"line": ln, "dir": "over" if o >= 0.5 else "under", "p": round(lean, 4)}


def corners(xh, xa, lh, la):
    """Korner tiap tim = rata-rata umum x (gol harapan relatif terhadap liga) ^ 0.45."""
    clamp = lambda r: max(0.55, min(1.8, r))
    ch = CORNER_BASE[0] * clamp(xh / lh) ** 0.45
    ca = CORNER_BASE[1] * clamp(xa / la) ** 0.45
    ph, pa = nb_pmf(ch, 1.25, 30), nb_pmf(ca, 1.25, 30)
    pt = [0.0] * 60
    mx = {"h": 0.0, "d": 0.0, "a": 0.0}
    for i, a in enumerate(ph):
        for j, b in enumerate(pa):
            pt[i + j] += a * b
            mx["h" if i > j else "a" if j > i else "d"] += a * b
    r4 = lambda x: round(x, 4)
    ov = [[ln, r4(tail(pt, ln))] for ln in (8.5, 9.5, 10.5, 11.5)]
    rec = lean_pick(ov)
    return {"ch": round(ch, 2), "ca": round(ca, 2), "tot": round(ch + ca, 2), "ov": ov,
            "th": [[x, r4(tail(ph, x))] for x in (3.5, 4.5, 5.5, 6.5)],
            "ta": [[x, r4(tail(pa, x))] for x in (2.5, 3.5, 4.5, 5.5)],
            "mx": {k: r4(v) for k, v in mx.items()}, "rec": rec}


def markets(g, ph, pa):
    """Peluang selisih gol (garis setengah, tanpa hasil imbang) dan total gol dari matriks skor."""
    ah = [[L] + [round(x, 4) for x in cover(g, "h", L)] for L in (-1.5, -0.5, 0.5, 1.5)]
    over = lambda line: sum(p for i, r in enumerate(g) for j, p in enumerate(r) if i + j > line)
    ou = [[ln, round(over(ln), 4), round(1 - over(ln), 4)] for ln in (1.5, 2.5, 3.5)]
    etg = sum((i + j) * p for i, r in enumerate(g) for j, p in enumerate(r))
    fav = "h" if ph >= pa else "a"
    rec_ah = None
    for L in (-2.5, -1.5, -0.5, 0.5, 1.5):
        w, pu, l = cover(g, fav, L)
        if w + l > 0 and w / (w + l) >= 0.60:
            rec_ah = {"side": fav, "line": L, "p": round(w / (w + l), 4), "push": round(pu, 4)}
            break
    rec_ou = lean_pick([[ln, o] for ln, o, _ in ou])
    return {"ah": ah, "ou": ou, "etg": round(etg, 2), "rec": {"ah": rec_ah, "ou": rec_ou}}


def finished(ms):
    out = []
    for m in ms:
        if m["status"] != "FINISHED":
            continue
        ft = m["score"]["fullTime"]
        if ft["home"] is None or ft["away"] is None:
            continue
        out.append(m)
    return sorted(out, key=lambda m: m["utcDate"])


def elo_step(R, h, a, gh, ga):
    rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
    e = 1 / (1 + 10 ** (-(rh + HOME_ADV - ra) / 400))
    s = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
    d = 20 * (math.log(abs(gh - ga) + 1) + 1) * (s - e)
    R[h], R[a] = rh + d, ra - d


def elo_table(ms):
    R = {}
    for m in finished(ms):
        ft = m["score"]["fullTime"]
        elo_step(R, m["homeTeam"]["id"], m["awayTeam"]["id"], ft["home"], ft["away"])
    return R


def blend(ph, pd, pa, rh, ra):
    """Geser peluang menang/kalah ke arah ekspektasi Elo."""
    e = 1 / (1 + 10 ** (-(rh + HOME_ADV - ra) / 400))
    delta = ELO_W * (e - (ph + 0.5 * pd))
    ph2, pa2 = max(0.01, ph + delta), max(0.01, pa - delta)
    s = ph2 + pd + pa2
    return ph2 / s, pd / s, pa2 / s


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


def nm(team):
    return team.get("shortName") or team["name"]


def backtest(ms):
    """Uji mundur: tiap laga selesai diprediksi HANYA dengan data sebelum laga itu, lalu dicocokkan dengan hasil."""
    R, st, pts, rows = {}, {}, {}, []
    L = {"h": 0.0, "a": 0.0, "n": 0}
    new = lambda: {"hf": 0, "ha": 0, "hn": 0, "af": 0, "aa": 0, "an": 0}
    for m in finished(ms):
        h, a = m["homeTeam"]["id"], m["awayTeam"]["id"]
        gh, ga = m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]
        S, T = st.setdefault(h, new()), st.setdefault(a, new())
        if L["n"] >= 30 and S["hn"] >= 3 and T["an"] >= 3:
            lh, la = (L["h"] / L["n"]) or 1.5, (L["a"] / L["n"]) or 1.2
            att_h = shrink((S["hf"] / S["hn"]) / lh, S["hn"])
            def_h = shrink((S["ha"] / S["hn"]) / la, S["hn"])
            att_a = shrink((T["af"] / T["an"]) / la, T["an"])
            def_a = shrink((T["aa"] / T["an"]) / lh, T["an"])
            xh, xa = lh * att_h * def_a, la * att_a * def_h
            ph, pd, pa = summarize(grid(xh, xa))[:3]
            ph, pd, pa = blend(ph, pd, pa, R.get(h, 1500.0), R.get(a, 1500.0))
            xh, xa = fit_supremacy(xh, xa, ph)
            ph, pd, pa, o25, _, _ = summarize(grid(xh, xa))
            label, _ = decide(ph, pd, pa, sum(pts.get(h, [])[-5:]), sum(pts.get(a, [])[-5:]))
            rows.append({"ph": ph, "pd": pd, "pa": pa, "po": o25, "label": label,
                         "res": "h" if gh > ga else "d" if gh == ga else "a", "tg": gh + ga,
                         "h": nm(m["homeTeam"]), "a": nm(m["awayTeam"]), "sc": f"{gh}-{ga}", "t": m["utcDate"]})
        S["hf"] += gh; S["ha"] += ga; S["hn"] += 1
        T["af"] += ga; T["aa"] += gh; T["an"] += 1
        L["h"] += gh; L["a"] += ga; L["n"] += 1
        elo_step(R, h, a, gh, ga)
        p1, p2 = (3, 0) if gh > ga else (1, 1) if gh == ga else (0, 3)
        pts.setdefault(h, []).append(p1)
        pts.setdefault(a, []).append(p2)
    return rows


def accuracy(rows):
    if not rows:
        return {"n": 0}
    hit, brier, ou_hit = 0, 0.0, 0
    by, bk = {}, {}
    for r in rows:
        p = {"h": r["ph"], "d": r["pd"], "a": r["pa"]}
        pick = max(p, key=p.get)
        ok = pick == r["res"]
        hit += ok
        brier += sum((p[k] - (1.0 if k == r["res"] else 0.0)) ** 2 for k in "hda")
        b = by.setdefault(r["label"], [0, 0]); b[0] += 1; b[1] += ok
        key = "hi" if p[pick] >= 0.65 else "mid" if p[pick] >= 0.5 else "lo"
        c = bk.setdefault(key, [0, 0, 0.0]); c[0] += 1; c[1] += ok; c[2] += p[pick]
        ou_hit += (r["po"] >= 0.5) == (r["tg"] > 2)
    recent = []
    for r in sorted(rows, key=lambda r: r["t"], reverse=True)[:12]:
        p = {"h": r["ph"], "d": r["pd"], "a": r["pa"]}
        pick = max(p, key=p.get)
        recent.append({"h": r["h"], "a": r["a"], "sc": r["sc"], "pick": pick, "res": r["res"],
                       "ok": pick == r["res"], "t": r["t"]})
    n = len(rows)
    return {"n": n, "hit": hit / n, "brier": brier / n, "base": 2 / 3,
            "by": {k: {"n": v[0], "hit": v[1] / v[0]} for k, v in by.items()},
            "bk": {k: {"n": v[0], "hit": v[1] / v[0], "avg": v[2] / v[0]} for k, v in bk.items()},
            "ou": {"n": n, "hit": ou_hit / n}, "recent": recent}


def analyse(code, name):
    st = get(f"/competitions/{code}/standings")["standings"]
    tab = {s["type"]: {t["team"]["id"]: t for t in s["table"]} for s in st}
    total = tab.get("TOTAL") or next(iter(tab.values()))
    home = tab.get("HOME") or total
    away = tab.get("AWAY") or total
    gp = lambda tb: max(1, sum(t["playedGames"] for t in tb.values()))
    lh = sum(t["goalsFor"] for t in home.values()) / gp(home) or 1.5
    la = sum(t["goalsFor"] for t in away.values()) / gp(away) or 1.2

    ms = get(f"/competitions/{code}/matches")["matches"]
    elo = elo_table(ms)
    bt = backtest(ms)
    pts, done = {}, {}
    for m in finished(ms):
        gh, ga = m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]
        h, a = m["homeTeam"]["id"], m["awayTeam"]["id"]
        p1, p2 = (3, 0) if gh > ga else (1, 1) if gh == ga else (0, 3)
        pts.setdefault(h, []).append(p1)
        pts.setdefault(a, []).append(p2)
        done[str(m["id"])] = ("h" if gh > ga else "d" if gh == ga else "a", f"{gh}-{ga}")
    form = lambda t: sum(pts.get(t, [])[-5:])

    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for m in ms:
        if m["status"] not in ("SCHEDULED", "TIMED"):
            continue
        t = dt.datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        if not (now <= t <= now + dt.timedelta(days=8)):
            continue
        hid, aid = m["homeTeam"]["id"], m["awayTeam"]["id"]
        if hid not in home or aid not in away:
            continue
        H, A = home[hid], away[aid]
        nh, na = max(1, H["playedGames"]), max(1, A["playedGames"])
        att_h = shrink((H["goalsFor"] / nh) / lh, nh)
        def_h = shrink((H["goalsAgainst"] / nh) / la, nh)
        att_a = shrink((A["goalsFor"] / na) / la, na)
        def_a = shrink((A["goalsAgainst"] / na) / lh, na)
        xh, xa = lh * att_h * def_a, la * att_a * def_h
        ph, pd, pa = summarize(grid(xh, xa))[:3]
        rh, ra = elo.get(hid, 1500.0), elo.get(aid, 1500.0)
        ph, pd, pa = blend(ph, pd, pa, rh, ra)
        xh, xa = fit_supremacy(xh, xa, ph)
        g = grid(xh, xa)
        ph, pd, pa, o25, btts, sc = summarize(g)
        mk = markets(g, ph, pa)
        label, why = decide(ph, pd, pa, form(hid), form(aid))
        why.append(f"rating Elo {round(rh)} vs {round(ra)}")
        out.append({"id": m["id"], "lg": name, "t": m["utcDate"],
                    "h": nm(m["homeTeam"]), "a": nm(m["awayTeam"]),
                    "ph": ph, "pd": pd, "pa": pa, "o25": o25, "btts": btts, "sc": sc,
                    "xh": xh, "xa": xa, "fh": form(hid), "fa": form(aid),
                    "eh": round(rh), "ea": round(ra), "label": label, "why": why,
                    "cr": corners(xh, xa, lh, la), **mk})
    return out, done, bt


def load_hist():
    try:
        with open(HIST, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def update_hist(hist, upcoming, done, now):
    """Arsip prediksi live (dibekukan saat kickoff). Dipertahankan agar riwayat tetap tersimpan."""
    for u in upcoming:
        kick = dt.datetime.fromisoformat(u["t"].replace("Z", "+00:00"))
        if now < kick:
            hist[str(u["id"])] = {k: u[k] for k in ("t", "lg", "h", "a", "ph", "pd", "pa", "label")}
    for k, h in hist.items():
        if "res" not in h and k in done:
            h["res"], h["sc"] = done[k]


def main():
    hist, data, done, bt, errs = load_hist(), [], {}, [], 0
    for code, name in LEAGUES.items():
        try:
            up, dn, b = analyse(code, name)
            data += up
            done.update(dn)
            bt += b
        except Exception as e:
            errs += 1
            print("Gagal", code, e)
    if errs == len(LEAGUES):
        sys.exit(1)  # jangan timpa halaman lama dengan halaman kosong
    now = dt.datetime.now(dt.timezone.utc)
    update_hist(hist, data, done, now)
    with open(HIST, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False)
    data.sort(key=lambda x: x["t"])
    acc = accuracy(bt)

    tpl = open("template.html", encoding="utf-8").read()
    upd = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%d %b %Y %H:%M WIB")
    esc = lambda o: json.dumps(o).replace("</", "<\\/")
    html = tpl.replace("__DATA__", esc(data)).replace("__ACC__", esc(acc)).replace("__UPDATED__", upd)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Selesai:", len(data), "pertandingan,", acc.get("n", 0), "laga uji mundur,", errs, "liga gagal")


if __name__ == "__main__":
    main()
