import os, sys, json, math, time, datetime as dt, requests

KEY = os.environ.get("FD_API_KEY", "")
BASE = "https://api.football-data.org/v4"
LEAGUES = {"PL": "Premier League", "PD": "La Liga", "SA": "Serie A",
           "BL1": "Bundesliga", "FL1": "Ligue 1", "CL": "Liga Champions"}
HIST = "history.json"
HOME_ADV = 60      # keunggulan kandang dalam poin Elo
ELO_W = 0.4        # bobot Elo saat digabung dengan model gol
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
    g = [[pois(i, lh) * pois(j, la) for j in range(n)] for i in range(n)]
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


def markets(g, ph, pa):
    """Handicap Asia (garis penuh dan setengah) dan Over/Under dari matriks skor."""
    ah = [[L] + [round(x, 4) for x in cover(g, "h", L)] for L in (-1.5, -1, -0.5, 0, 0.5, 1, 1.5)]
    over = lambda line: sum(p for i, r in enumerate(g) for j, p in enumerate(r) if i + j > line)
    ou = [[ln, round(over(ln), 4), round(1 - over(ln), 4)] for ln in (1.5, 2.5, 3.5)]
    etg = sum((i + j) * p for i, r in enumerate(g) for j, p in enumerate(r))
    # Rekomendasi handicap: garis tersulit untuk tim unggulan dengan peluang tembus >= 60%
    fav = "h" if ph >= pa else "a"
    rec_ah = None
    for L in (-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5):
        w, pu, l = cover(g, fav, L)
        if w + l > 0 and w / (w + l) >= 0.60:
            rec_ah = {"side": fav, "line": L, "p": round(w / (w + l), 4), "push": round(pu, 4)}
            break
    # Kecenderungan Over/Under: garis dengan kecondongan terkuat, minimal 58%
    ln, ov = max(((ln, o) for ln, o, _ in ou), key=lambda x: abs(x[1] - 0.5))
    lean = max(ov, 1 - ov)
    rec_ou = {"line": ln, "dir": "over" if ov >= 0.5 else "under", "p": round(lean, 4)} if lean >= 0.58 else None
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


def elo_table(ms):
    """Rating Elo dari hasil laga musim ini (selisih gol memperbesar perubahan)."""
    R = {}
    for m in finished(ms):
        h, a = m["homeTeam"]["id"], m["awayTeam"]["id"]
        gh, ga = m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]
        rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
        e = 1 / (1 + 10 ** (-(rh + HOME_ADV - ra) / 400))
        s = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        k = 20 * (math.log(abs(gh - ga) + 1) + 1)
        d = k * (s - e)
        R[h], R[a] = rh + d, ra - d
    return R


def blend(ph, pd, pa, rh, ra):
    """Geser peluang menang/kalah ke arah ekspektasi Elo, peluang seri tetap."""
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
                    "h": m["homeTeam"].get("shortName") or m["homeTeam"]["name"],
                    "a": m["awayTeam"].get("shortName") or m["awayTeam"]["name"],
                    "ph": ph, "pd": pd, "pa": pa, "o25": o25, "btts": btts, "sc": sc,
                    "xh": xh, "xa": xa, "fh": form(hid), "fa": form(aid),
                    "eh": round(rh), "ea": round(ra), "label": label, "why": why, **mk})
    return out, done


def load_hist():
    try:
        with open(HIST, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def update_hist(hist, upcoming, done, now):
    # Simpan prediksi hanya selama laga belum dimulai, supaya tidak "mengintip" hasil.
    for u in upcoming:
        kick = dt.datetime.fromisoformat(u["t"].replace("Z", "+00:00"))
        if now < kick:
            hist[str(u["id"])] = {k: u[k] for k in ("t", "lg", "h", "a", "ph", "pd", "pa", "label")}
    for k, h in hist.items():
        if "res" not in h and k in done:
            h["res"], h["sc"] = done[k]


def accuracy(hist):
    rows = [h for h in hist.values() if "res" in h]
    if not rows:
        return {"n": 0}
    hit, brier = 0, 0.0
    by, bk, recent = {}, {}, []
    for h in rows:
        p = {"h": h["ph"], "d": h["pd"], "a": h["pa"]}
        pick = max(p, key=p.get)
        ok = pick == h["res"]
        hit += ok
        brier += sum((p[k] - (1.0 if k == h["res"] else 0.0)) ** 2 for k in "hda")
        b = by.setdefault(h["label"], [0, 0]); b[0] += 1; b[1] += ok
        key = "hi" if p[pick] >= 0.65 else "mid" if p[pick] >= 0.5 else "lo"
        c = bk.setdefault(key, [0, 0, 0.0]); c[0] += 1; c[1] += ok; c[2] += p[pick]
    for h in sorted(rows, key=lambda h: h["t"], reverse=True)[:12]:
        p = {"h": h["ph"], "d": h["pd"], "a": h["pa"]}
        pick = max(p, key=p.get)
        recent.append({"h": h["h"], "a": h["a"], "sc": h.get("sc", ""), "pick": pick,
                       "res": h["res"], "ok": pick == h["res"], "t": h["t"]})
    n = len(rows)
    return {"n": n, "hit": hit / n, "brier": brier / n, "base": 2 / 3,
            "by": {k: {"n": v[0], "hit": v[1] / v[0]} for k, v in by.items()},
            "bk": {k: {"n": v[0], "hit": v[1] / v[0], "avg": v[2] / v[0]} for k, v in bk.items()},
            "recent": recent}


def main():
    hist, data, done, errs = load_hist(), [], {}, 0
    for code, name in LEAGUES.items():
        try:
            up, dn = analyse(code, name)
            data += up
            done.update(dn)
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
    acc = accuracy(hist)

    tpl = open("template.html", encoding="utf-8").read()
    upd = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%d %b %Y %H:%M WIB")
    esc = lambda o: json.dumps(o).replace("</", "<\\/")
    html = tpl.replace("__DATA__", esc(data)).replace("__ACC__", esc(acc)).replace("__UPDATED__", upd)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Selesai:", len(data), "pertandingan,", acc.get("n", 0), "hasil terlacak")


if __name__ == "__main__":
    main()
