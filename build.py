import os, sys, re, json, math, time, base64, datetime as dt, requests

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



# ---------- Statistik tim, pola, dan insight ----------
def team_stats(ms):
    seq = {}
    for m in finished(ms):
        h, a = m["homeTeam"]["id"], m["awayTeam"]["id"]
        gh, ga = m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]
        seq.setdefault(h, []).append(("H", gh, ga))
        seq.setdefault(a, []).append(("A", ga, gh))
    pt = {"W": 3, "D": 1, "L": 0}
    S = {}
    for t, L in seq.items():
        rs = ["W" if gf > ga else "D" if gf == ga else "L" for _, gf, ga in L]

        def split(v):
            X = [x for x in L if x[0] == v]
            w = sum(1 for _, gf, ga in X if gf > ga)
            d = sum(1 for _, gf, ga in X if gf == ga)
            return {"n": len(X), "w": w, "d": d, "l": len(X) - w - d,
                    "gf": sum(x[1] for x in X), "ga": sum(x[2] for x in X)}

        def streak(cond):
            c = 0
            for _, gf, ga in reversed(L):
                if not cond(gf, ga):
                    break
                c += 1
            return c

        S[t] = {"n": len(L), "w": rs.count("W"), "d": rs.count("D"), "l": rs.count("L"),
                "gf": sum(x[1] for x in L), "ga": sum(x[2] for x in L),
                "cs": sum(1 for _, gf, ga in L if ga == 0), "fts": sum(1 for _, gf, ga in L if gf == 0),
                "btts": sum(1 for _, gf, ga in L if gf > 0 and ga > 0),
                "o25": sum(1 for _, gf, ga in L if gf + ga > 2),
                "form": "".join(rs[-10:]), "p5": sum(pt[r] for r in rs[-5:]),
                "home": split("H"), "away": split("A"),
                "st": {"win": streak(lambda f, g: f > g), "unb": streak(lambda f, g: f >= g),
                       "los": streak(lambda f, g: f < g), "wl": streak(lambda f, g: f <= g)}}
    return S


def patterns(name, s):
    P, st, n = [], s["st"], s["n"]
    if n < 4:
        return P
    if st["win"] >= 3: P.append(f"{name} menang {st['win']} laga beruntun.")
    elif st["unb"] >= 5: P.append(f"{name} tidak terkalahkan dalam {st['unb']} laga terakhir.")
    if st["los"] >= 3: P.append(f"{name} kalah {st['los']} laga beruntun.")
    elif st["wl"] >= 5: P.append(f"{name} belum menang dalam {st['wl']} laga terakhir.")
    h, a = s["home"], s["away"]
    if h["n"] >= 4 and a["n"] >= 4:
        hp, ap = (3 * h["w"] + h["d"]) / h["n"], (3 * a["w"] + a["d"]) / a["n"]
        if hp - ap >= 1.0: P.append(f"{name} jauh lebih kuat di kandang ({hp:.1f} poin per laga) dibanding tandang ({ap:.1f}).")
        elif ap - hp >= 0.5: P.append(f"{name} justru lebih baik di tandang ({ap:.1f} poin per laga) dibanding kandang ({hp:.1f}).")
    r = lambda k: s[k] / n
    if r("o25") >= 0.65: P.append(f"{round(r('o25') * 100)}% laga {name} berakhir dengan 3 gol atau lebih.")
    elif r("o25") <= 0.30: P.append(f"Hanya {round(r('o25') * 100)}% laga {name} berakhir dengan 3 gol atau lebih, cenderung rapat.")
    if r("btts") >= 0.65: P.append(f"Kedua tim mencetak gol di {round(r('btts') * 100)}% laga {name}.")
    if r("cs") >= 0.40: P.append(f"{name} menjaga gawang bersih di {round(r('cs') * 100)}% laga.")
    if r("fts") >= 0.30: P.append(f"{name} gagal mencetak gol di {round(r('fts') * 100)}% laga.")
    if n >= 8:
        sea, f5 = (3 * s["w"] + s["d"]) / n, s["p5"] / 5
        if f5 - sea >= 0.8: P.append(f"Performa {name} sedang naik: {f5:.1f} poin per laga dalam 5 laga terakhir, di atas rata-rata musim ({sea:.1f}).")
        elif sea - f5 >= 0.8: P.append(f"Performa {name} sedang turun: {f5:.1f} poin per laga dalam 5 laga terakhir, di bawah rata-rata musim ({sea:.1f}).")
    return P


def match_insight(hn, an, hs, as_, xh, xa, p, elo, etg):
    out = []
    if hs and as_ and hs["home"]["n"] >= 3 and as_["away"]["n"] >= 3:
        h, a = hs["home"], as_["away"]
        out.append(f"{hn} di kandang: {h['w']}M {h['d']}S {h['l']}K, mencetak {h['gf'] / h['n']:.1f} dan kebobolan {h['ga'] / h['n']:.1f} gol per laga. "
                   f"{an} di tandang: {a['w']}M {a['d']}S {a['l']}K, mencetak {a['gf'] / a['n']:.1f} dan kebobolan {a['ga'] / a['n']:.1f} gol per laga.")
        out.append(f"Rata-rata total gol di kandang {hn} adalah {(h['gf'] + h['ga']) / h['n']:.1f} dan di tandang {an} adalah "
                   f"{(a['gf'] + a['ga']) / a['n']:.1f}. Model memperkirakan {etg:.1f} gol di laga ini.")
    pats = []
    for s, n in ((hs, hn), (as_, an)):
        if s:
            pats += patterns(n, s)[:2]
    out += pats[:4]
    top = max(p)
    lean = f"{hn} menang" if p[0] == top else f"{an} menang" if p[2] == top else "hasil seri"
    out.append(f"Model condong ke {lean} ({round(top * 100)}%): selisih gol harapan {xh - xa:+.1f} dan selisih rating Elo {round(elo[0] - elo[1]):+d} untuk tuan rumah.")
    if (hs and hs["n"] < 8) or (as_ and as_["n"] < 8):
        out.append("Catatan: data musim ini masih sedikit untuk salah satu tim, jadi pola belum stabil.")
    return out


def comp_summary(ms):
    F = finished(ms)
    n = len(F)
    if not n:
        return None
    g = [(m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]) for m in F]
    return {"n": n, "avg": round(sum(a + b for a, b in g) / n, 2),
            "hw": sum(a > b for a, b in g) / n, "dr": sum(a == b for a, b in g) / n, "aw": sum(a < b for a, b in g) / n,
            "o25": sum(a + b > 2 for a, b in g) / n, "btts": sum(a > 0 and b > 0 for a, b in g) / n}


def comp_insights(S, names):
    T = [(t, s) for t, s in S.items() if s["n"] >= 5]
    if len(T) < 4:
        return []
    out = []
    t, s = max(T, key=lambda x: x[1]["gf"] / x[1]["n"])
    out.append(f"Serangan terbaik: {names[t]} mencetak {s['gf'] / s['n']:.2f} gol per laga.")
    t, s = min(T, key=lambda x: x[1]["ga"] / x[1]["n"])
    out.append(f"Pertahanan terbaik: {names[t]} hanya kebobolan {s['ga'] / s['n']:.2f} gol per laga.")
    t, s = max(T, key=lambda x: x[1]["st"]["unb"])
    if s["st"]["unb"] >= 4: out.append(f"Rentetan tak terkalahkan terpanjang: {names[t]}, {s['st']['unb']} laga.")
    t, s = max(T, key=lambda x: x[1]["st"]["win"])
    if s["st"]["win"] >= 3: out.append(f"Rentetan kemenangan terpanjang: {names[t]}, {s['st']['win']} laga beruntun.")
    hm = [(t, s) for t, s in T if s["home"]["n"] >= 3]
    if hm:
        t, s = max(hm, key=lambda x: (3 * x[1]["home"]["w"] + x[1]["home"]["d"]) / x[1]["home"]["n"])
        h = s["home"]
        out.append(f"Paling kuat di kandang: {names[t]} dengan {(3 * h['w'] + h['d']) / h['n']:.1f} poin per laga.")
    t, s = max(T, key=lambda x: x[1]["o25"] / x[1]["n"])
    out.append(f"Laga paling banyak gol: {names[t]}, {round(s['o25'] / s['n'] * 100)}% laganya berakhir 3 gol atau lebih.")
    t, s = max(T, key=lambda x: x[1]["p5"])
    out.append(f"Form terbaik: {names[t]} meraih {s['p5']} dari 15 poin di 5 laga terakhir.")
    return out


def llm_summary(name, facts):
    """Opsional: Claude menulis ringkasan dari fakta yang sudah dihitung. Aktif hanya bila INSIGHT_LLM=1."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or os.environ.get("INSIGHT_LLM") != "1" or not facts:
        return None
    prompt = ("Kamu analis statistik sepakbola. Dari fakta tentang " + name + " berikut, tulis 4 poin singkat dalam bahasa Indonesia "
              "yang menjelaskan pola permainan dan artinya. Hanya pakai fakta yang diberikan, jangan mengarang angka, "
              "jangan menyarankan bertaruh.\n\n" + "\n".join(facts))
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": os.environ.get("INSIGHT_MODEL", "claude-sonnet-5-5"), "max_tokens": 500,
                                "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    except Exception as e:
        print("Ringkasan AI gagal", name, e)
        return None


def standings_out(st):
    out = {}
    for s in st:
        rows = out.setdefault(s["type"], [])
        for t in s["table"]:
            rows.append({"pos": t["position"], "id": str(t["team"]["id"]), "name": nm(t["team"]),
                         "m": t["playedGames"], "w": t["won"], "d": t["draw"], "l": t["lost"],
                         "gf": t["goalsFor"], "ga": t["goalsAgainst"],
                         "gd": t.get("goalDifference", t["goalsFor"] - t["goalsAgainst"]), "pts": t["points"]})
    return out


# ---------- Supabase ----------
def sb_headers():
    k = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if k.startswith("sb_"):  # kunci format baru (sb_secret_...) dikirim lewat header apikey saja
        return {"apikey": k, "Content-Type": "application/json"}
    return {"apikey": k, "Authorization": "Bearer " + k, "Content-Type": "application/json"}


def key_role(k):
    """Baca peran dari kunci lama (JWT): 'anon' atau 'service_role'. None jika bukan JWT."""
    try:
        p = k.split(".")[1]
        p += "=" * (-len(p) % 4)
        return json.loads(base64.urlsafe_b64decode(p)).get("role")
    except Exception:
        return None


def check_keys():
    k = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    u = os.environ.get("SUPABASE_URL", "").strip()
    if k.startswith("sb_publishable_") or key_role(k) == "anon":
        sys.exit("GAGAL: SUPABASE_SERVICE_KEY berisi kunci anon/publishable. Isi dengan kunci service_role "
                 "(Supabase > Project Settings > API > tab Legacy API keys > service_role).")
    if u and not u.startswith("https://"):
        sys.exit("GAGAL: SUPABASE_URL harus diawali https:// (contoh https://abcd.supabase.co)")


def sb_ready():
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


def sb_put(key, data):
    url = os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/snapshots?on_conflict=key"
    body = json.dumps({"key": key, "data": data, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    r = requests.post(url, headers={**sb_headers(), "Prefer": "resolution=merge-duplicates"}, data=body, timeout=90)
    if not r.ok:
        print("Supabase menolak:", r.status_code, r.text[:300])
    r.raise_for_status()


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
    try:
        scorers = [{"p": x["player"]["name"], "t": nm(x["team"]), "g": x.get("goals") or 0,
                    "a": x.get("assists") or 0, "n": x.get("playedMatches") or 0}
                   for x in get(f"/competitions/{code}/scorers?limit=10")["scorers"]]
    except Exception as e:
        print("Pencetak gol tidak tersedia", code, e)
        scorers = []
    elo = elo_table(ms)
    bt = backtest(ms)
    S = team_stats(ms)
    names = {}
    for tb in tab.values():
        for tid, t in tb.items():
            names[tid] = nm(t["team"])
    for m in ms:
        for k in ("homeTeam", "awayTeam"):
            names.setdefault(m[k]["id"], nm(m[k]))
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
        ins = match_insight(names[hid], names[aid], S.get(hid), S.get(aid), xh, xa, (ph, pd, pa), (rh, ra), mk["etg"])
        out.append({"id": m["id"], "lg": name, "t": m["utcDate"],
                    "h": nm(m["homeTeam"]), "a": nm(m["awayTeam"]),
                    "ph": ph, "pd": pd, "pa": pa, "o25": o25, "btts": btts, "sc": sc,
                    "xh": xh, "xa": xa, "fh": form(hid), "fa": form(aid),
                    "eh": round(rh), "ea": round(ra), "label": label, "why": why,
                    "cr": corners(xh, xa, lh, la), "ins": ins,
                    "hid": hid, "aid": aid, **mk})
    ci = comp_insights(S, names)
    comp = {"table": standings_out(st), "sum": comp_summary(ms), "ins": ci, "scorers": scorers,
            "teams": {str(t): {"name": names.get(t, "?"), **s, "pat": patterns(names.get(t, "?"), s)} for t, s in S.items()}}
    ai = llm_summary(name, ci + [f"Rata-rata gol per laga: {comp['sum']['avg']}"] if comp["sum"] else ci)
    if ai:
        comp["ai"] = ai
    return out, done, bt, comp


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
    hist, data, done, bt, comps, errs = load_hist(), [], {}, [], {}, 0
    for code, name in LEAGUES.items():
        try:
            up, dn, b, comp = analyse(code, name)
            data += up
            done.update(dn)
            bt += b
            comps[name] = comp
        except Exception as e:
            errs += 1
            print("Gagal", code, e)
    if errs == len(LEAGUES):
        sys.exit(1)  # jangan menimpa data lama dengan data kosong
    now = dt.datetime.now(dt.timezone.utc)
    update_hist(hist, data, done, now)
    with open(HIST, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False)
    data.sort(key=lambda x: x["t"])
    upd = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).strftime("%d %b %Y %H:%M WIB")
    comps["_order"] = [n for n in LEAGUES.values() if n in comps]
    snaps = {"matches": {"d": data, "acc": accuracy(bt), "upd": upd}, "competitions": comps}
    if sb_ready():
        check_keys()
        for k, v in snaps.items():
            sb_put(k, v)
        print("Data terkirim ke Supabase:", ", ".join(snaps))
    else:
        with open("snapshot.json", "w", encoding="utf-8") as f:
            json.dump(snaps, f, ensure_ascii=False)
        print("PERINGATAN: SUPABASE_URL/SUPABASE_SERVICE_KEY belum diatur; data ditulis ke snapshot.json saja")

    tpl = open("template.html", encoding="utf-8").read()
    html = tpl.replace("__SB_URL__", os.environ.get("SUPABASE_URL", "")).replace("__SB_KEY__", os.environ.get("SUPABASE_ANON_KEY", ""))
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Selesai:", len(data), "pertandingan,", len(comps) - 1, "kompetisi,", accuracy(bt).get("n", 0), "laga uji mundur,", errs, "liga gagal")


if __name__ == "__main__":
    main()
