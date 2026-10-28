import os, json, urllib.request, sqlite3, datetime as dt
from flask import Flask, g, jsonify, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
DB, SLOT_MIN = "fuel.db", 10
RULES = [
    "قاعدة الأولوية: سيارات الإسعاف تأخذ الدور الأول دائماً في أي كازية قبل باقي السيارات.",
    "قاعدة منع التكرار: كل رقم سيارة يسجل مرة واحدة فقط، ولا يمكن تسجيله باسم شخص آخر.",
    "كيف يتم الحجز: أدخل رقم السيارة واسمك ونوع الوقود والكمية بالليتر، فيحجز النظام الكازية والموعد المناسب.",
]

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close(_):
    if "db" in g: g.db.close()

def init():
    c = sqlite3.connect(DB)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS stations(id INTEGER PRIMARY KEY, name TEXT, city TEXT, fuel TEXT, liters REAL, opens TEXT);
    CREATE TABLE IF NOT EXISTS vehicles(id INTEGER PRIMARY KEY, plate TEXT UNIQUE, owner TEXT, kind TEXT,
      fuel TEXT, liters REAL, station_id INTEGER, slot TEXT);""")
    c.commit(); c.close()

def reschedule(sid):
    d = db(); st = d.execute("select * from stations where id=?", (sid,)).fetchone()
    h, m = map(int, st["opens"].split(":"))
    t = dt.datetime.combine(dt.date.today(), dt.time(h, m))
    t = max(t, dt.datetime.now().replace(second=0, microsecond=0))
    used = 0
    rows = d.execute("select * from vehicles where station_id=? order by kind='ambulance' desc, id", (sid,)).fetchall()
    for v in rows:  # الإسعاف أولاً، ثم حسب ترتيب التسجيل
        if used + v["liters"] > st["liters"]:
            d.execute("update vehicles set station_id=NULL, slot=NULL where id=?", (v["id"],))
        else:
            d.execute("update vehicles set slot=? where id=?", (t.strftime("%H:%M"), v["id"]))
            used += v["liters"]; t += dt.timedelta(minutes=SLOT_MIN)
    d.commit()

def assign(vid):
    d = db(); v = d.execute("select * from vehicles where id=?", (vid,)).fetchone()
    best, best_free = None, -1
    for s in d.execute("select * from stations where fuel=?", (v["fuel"],)):
        # الإسعاف يتجاهل حجز غيره (يمكنه إزاحتهم)، والبقية تحترم الكمية المحجوزة
        cond = "and kind='ambulance'" if v["kind"] == "ambulance" else ""
        used = d.execute(f"select coalesce(sum(liters),0) from vehicles where station_id=? {cond}", (s["id"],)).fetchone()[0]
        if s["liters"] - used >= v["liters"] and s["liters"] - used > best_free:
            best, best_free = s["id"], s["liters"] - used
    if best:
        d.execute("update vehicles set station_id=? where id=?", (best, vid)); d.commit(); reschedule(best)

def rebalance():
    for v in db().execute("select id from vehicles where station_id is null order by kind='ambulance' desc, id").fetchall():
        assign(v["id"])

def state():
    d = db(); out = []
    for s in d.execute("select * from stations"):
        r = d.execute("select coalesce(sum(liters),0) from vehicles where station_id=?", (s["id"],)).fetchone()[0]
        out.append({**dict(s), "reserved": r, "free": max(s["liters"] - r, 0)})
    vs = [dict(r) for r in d.execute("""select v.*, s.name station from vehicles v left join stations s on s.id=v.station_id
                                         order by v.kind='ambulance' desc, v.id""")]
    return {"stations": out, "vehicles": vs}

@app.get("/")
def index(): return render_template("index.html")

@app.get("/api/state")
def api_state(): return jsonify(state())

@app.post("/api/stations")
def add_station():
    j = request.json; d = db()
    try: liters = float(j["liters"])
    except (KeyError, ValueError): return jsonify(error="كمية الوقود غير صحيحة"), 400
    if not j.get("name") or liters < 0: return jsonify(error="أدخل اسم الكازية وكمية صحيحة"), 400
    d.execute("insert into stations(name,city,fuel,liters,opens) values(?,?,?,?,?)",
              (j["name"].strip(), j.get("city", "").strip(), j.get("fuel", "بنزين 95"), liters, j.get("opens", "08:00")))
    d.commit(); rebalance(); return jsonify(ok=True)

@app.post("/api/vehicles")
def add_vehicle():
    j = request.json; d = db()
    plate = "".join((j.get("plate") or "").split()).upper()
    owner = (j.get("owner") or "").strip()
    try: liters = float(j["liters"])
    except (KeyError, ValueError): return jsonify(error="كمية الوقود غير صحيحة"), 400
    if not plate or not owner or liters <= 0: return jsonify(error="أكمل جميع الحقول"), 400
    old = d.execute("select owner from vehicles where plate=?", (plate,)).fetchone()
    if old: return jsonify(error=f"السيارة {plate} مسجلة مسبقاً باسم {old['owner']} — لا يمكن التسجيل مرتين"), 409
    cur = d.execute("insert into vehicles(plate,owner,kind,fuel,liters) values(?,?,?,?,?)",
                    (plate, owner, "ambulance" if j.get("ambulance") else "normal", j.get("fuel", "بنزين 95"), liters))
    d.commit(); assign(cur.lastrowid)
    v = d.execute("select v.*, s.name station from vehicles v left join stations s on s.id=v.station_id where v.id=?",
                  (cur.lastrowid,)).fetchone()
    return jsonify(vehicle=dict(v))

# ---------------- RAG: استرجاع + توليد ----------------
def knowledge():
    docs = list(RULES)
    for s in state()["stations"]:
        docs.append(f"كازية {s['name']} في {s['city']}: وقود {s['fuel']}، المتوفر {s['free']:.0f} لتر من أصل {s['liters']:.0f}، "
                    f"تبدأ التعبئة الساعة {s['opens']}.")
    for v in state()["vehicles"]:
        docs.append(f"السيارة {v['plate']} باسم {v['owner']} ({'إسعاف' if v['kind']=='ambulance' else 'عادية'}) تحتاج {v['liters']:.0f} لتر "
                    + (f"موعدها في {v['station']} الساعة {v['slot']}." if v["slot"] else "وهي حالياً في قائمة الانتظار."))
    return docs

def retrieve(q, k=4):
    docs = knowledge()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(docs + [q])
    sims = cosine_similarity(vec.transform([q]), vec.transform(docs))[0]
    return [docs[i] for i in sims.argsort()[::-1][:k] if sims[i] > 0]

@app.post("/api/ask")
def ask():
    j = request.json
    q = (j.get("q") or "").strip()
    hist = [h for h in (j.get("history") or [])[-6:] if h.get("role") in ("user", "assistant")]
    prev = " ".join(h["content"] for h in hist if h["role"] == "user")[-200:]
    ctx = retrieve(q + " " + prev)  # نضيف سؤالاً سابقاً ليفهم الأسئلة المتتابعة
    if not ctx:
        return jsonify(answer="لم أجد معلومات تخص سؤالك في بيانات الكازيات الحالية.", sources=[])
    system = ("أنت مساعد لتنظيم تعبئة الوقود. أجب بالعربية وباختصار وبأسلوب ودّي، "
              "وبالاعتماد على المعلومات التالية فقط. إن لم تكفِ قل ذلك.\n\nالمعلومات:\n" + "\n".join(ctx))
    msgs = [{"role": h["role"], "content": h["content"]} for h in hist] + [{"role": "user", "content": q}]
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            r = anthropic.Anthropic().messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                                                      max_tokens=500, system=system, messages=msgs)
            return jsonify(answer=r.content[0].text, sources=ctx)
        body = json.dumps({"model": os.getenv("OLLAMA_MODEL", "qwen2.5:3b"), "stream": False,
                           "messages": [{"role": "system", "content": system}] + msgs}).encode()
        req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return jsonify(answer=json.load(r)["message"]["content"], sources=ctx)
    except Exception:
        return jsonify(answer="أقرب المعلومات لسؤالك:\n" + "\n".join("• " + c for c in ctx), sources=ctx)

if __name__ == "__main__":
    init(); app.run(debug=True)