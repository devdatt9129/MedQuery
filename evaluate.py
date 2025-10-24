#!/usr/bin/env python3
import os, json, re, time, uuid, csv
from pathlib import Path
from typing import List, Dict, Tuple
import requests

# ---------- Config ----------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Adjust if your file lives elsewhere
GOLD_PATH = Path("./backend/data/golden_source.json")
SESSION_ID = f"eval-{uuid.uuid4().hex[:8]}"
TIMEOUT = 60  # seconds per request
OUT_JSON = Path("eval_results.json")
OUT_CSV = Path("eval_report.csv")

# ---------- Text utils ----------
WS = re.compile(r"\s+")

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = WS.sub(" ", s)
    if len(s) >= 2 and s[0] in {'"', '“', '”'} and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s

def tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(s)) if t]

# ---------- Metrics ----------
def precision_recall_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    pt, gt = set(tokens(pred)), set(tokens(gold))
    if not pt or not gt:
        return 0.0, 0.0, 0.0
    common = pt & gt
    if not common:
        return 0.0, 0.0, 0.0
    prec = len(common) / len(pt)
    rec  = len(common) / len(gt)
    f1   = 2 * prec * rec / (prec + rec)
    return round(prec, 4), round(rec, 4), round(f1, 4)

def citation_supported(pred: str, gold: str, citations: List[Dict]) -> int:
    """Support heuristic: ≥30% overlap or substring match in any snippet."""
    gold_toks = set(tokens(gold))
    if not citations:
        return 0
    for c in citations:
        snippet = c.get("snippet") or ""
        sn_toks = set(tokens(snippet))
        inter = gold_toks & sn_toks
        if gold_toks and (len(inter) / max(len(gold_toks), 1)) >= 0.3:
            return 1
        if normalize(pred) and normalize(pred) in normalize(snippet):
            return 1
    return 0

# ---------- I/O ----------
def load_golden(path: Path) -> List[Dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("golden_source.json must be a list of QA items.")
    return data

def call_backend(question: str, session_id: str) -> Tuple[str, List[Dict], int, str]:
    url = f"{BACKEND_URL.rstrip('/')}/ask"
    try:
        r = requests.post(url,
                          json={"question": question, "session_id": session_id},
                          timeout=TIMEOUT)
        if r.status_code != 200:
            return "", [], r.status_code, f"HTTP {r.status_code}: {r.text[:300]}"
        data = r.json()
        ans = data.get("answer", "") or data.get("result", "") or ""
        cits = data.get("citations", []) or []
        return ans, cits, r.status_code, ""
    except Exception as e:
        return "", [], 0, str(e)

# ---------- Main ----------
def main():
    print("== MedQuery Evaluation ==")
    print(f"- Backend: {BACKEND_URL}")
    print(f"- Session: {SESSION_ID}")
    print(f"- Golden:  {GOLD_PATH.resolve()}")

    qa = load_golden(GOLD_PATH)
    rows: List[Dict] = []
    n = len(qa)

    for i, item in enumerate(qa, 1):
        q = item["question"]
        gold = item["gold_answer"]
        qtype = item.get("type", "factual")
        print(f"[{i}/{n}] Q: {q}")

        t_start = time.time()
        pred, cits, code, err = call_backend(q, SESSION_ID)
        latency = int((time.time() - t_start) * 1000)

        if err:
            print(f"   -> ERROR: {err}")
            prec = rec = f1s = 0.0
            sup = 0
        else:
            print(f"   -> A: {pred[:80]}{'...' if len(pred) > 80 else ''}  ({latency} ms)")
            prec, rec, f1s = precision_recall_f1(pred, gold)
            sup = citation_supported(pred, gold, cits)

        rows.append({
            "id": item.get("id"),
            "type": qtype,
            "question": q,
            "gold": gold,
            "pred": pred,
            "precision": prec,
            "recall": rec,
            "f1": f1s,
            "supported": sup,
            "status": code,
            "error": err,
            "latency_ms": latency,
            "citations": cits,
            "source": item.get("source", "")
        })

    # Save JSON
    OUT_JSON.write_text(json.dumps(rows, indent=2))

    # Save CSV (only declared columns)
    csv_fields = [
        "id","type","precision","recall","f1","supported",
        "latency_ms","status","question","gold","pred","source"
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in csv_fields})

    # Aggregates
    ok_rows = [r for r in rows if not r["error"] and r["status"] == 200]
    ok = len(ok_rows)
    total = len(rows)
    def avg(k): return sum(r[k] for r in ok_rows) / ok if ok else 0.0

    prec_avg, rec_avg, f1_avg, sup_avg, lat_avg = (
        avg("precision"), avg("recall"), avg("f1"), avg("supported"), avg("latency_ms")
    )

    # By-type breakdown
    by_type: Dict[str, List[Dict]] = {}
    for r in ok_rows:
        by_type.setdefault(r["type"], []).append(r)

    type_lines = []
    for t, group in by_type.items():
        def a(k): return sum(x[k] for x in group) / len(group)
        type_lines.append(
            f"  - {t:12s} Precision {a('precision'):.2f}  Recall {a('recall'):.2f}  "
            f"F1 {a('f1'):.2f}  Support {a('supported'):.2f}  n={len(group)}"
        )

    print("\n== Summary ==")
    print(f"Items: {total} | OK: {ok} | Errors: {total - ok}")
    print(f"Precision: {prec_avg:.2f} | Recall: {rec_avg:.2f} | F1: {f1_avg:.2f} | "
          f"Support: {sup_avg:.2f} | Avg Latency: {lat_avg:.0f} ms")
    if type_lines:
        print("By type:\n" + "\n".join(type_lines))
    print(f"\nWrote: {OUT_JSON.resolve()}\nWrote: {OUT_CSV.resolve()}")
    print("Done.")

if __name__ == "__main__":
    main()