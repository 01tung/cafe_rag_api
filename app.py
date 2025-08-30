# app.py — Qdrant Cloud + 動態權重 + 必要條件/部分符合邏輯
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os, json
from typing import Dict, Any, Optional, Tuple, List
from flask_cors import CORS

# 讀 .env
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
if not (OPENAI_API_KEY and QDRANT_URL and QDRANT_API_KEY):
    raise RuntimeError("缺少 OPENAI_API_KEY / QDRANT_URL / QDRANT_API_KEY")

# ---- Imports ----
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVS
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})  # 開發階段先全開，之後可改白名單

# ---- 常數 ----
COLLECTION = "cafes_v1"
SCORING_KEYS = ["Quiet", "Tasty", "Cheap", "Seat", "Wifi", "Music"]

# ---- 建立向量庫連線（查詢用）----
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 用官方 SDK 連 Qdrant Cloud
qclient = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 相容不同版的 LangChain Qdrant 介面：優先用 embeddings 參數，失敗再退回 embedding_function
try:
    db = QdrantVS(client=qclient, collection_name=COLLECTION, embeddings=embedding)
except TypeError:
    # 有些版本參數名是 embedding_function
    db = QdrantVS(client=qclient, collection_name=COLLECTION, embedding_function=embedding)

# ---- LLM（穩定輸出）----
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=OPENAI_API_KEY)

# ======= Few-shot =======
FEW_SHOT: List[Dict[str, Any]] = [
  {
    "user": "想找台北大安區 安靜 有插座 適合久坐",
    "required": {"City":"台北", "Socket":"有插座", "Limited_time":"不限時",
                 "Name":None, "Address":None, "Open_time":None, "Standing_desk":None},
    "weights":  {"Quiet":0.35,"Seat":0.25,"Wifi":0.15,"Tasty":0.10,"Cheap":0.10,"Music":0.05},
    "rationale":"重視安靜與久坐，Quiet/Seat 高；需穩定網路，Wifi 次高。"
  },
  {
    "user": "想聊天聚會、甜點好吃，地點不限",
    "required": {"City":None,"Socket":None,"Limited_time":None,"Name":None,"Address":None,"Open_time":None,"Standing_desk":None},
    "weights":  {"Tasty":0.35,"Seat":0.20,"Music":0.15,"Cheap":0.15,"Wifi":0.10,"Quiet":0.05},
    "rationale":"以口味與氛圍為主，Tasty 高、Music 稍高；Quiet 低。"
  },
  {
    "user": "要辦公用、網路穩、最好不限時",
    "required": {"Limited_time":"不限時","City":None,"Socket":None,"Name":None,"Address":None,"Open_time":None,"Standing_desk":None},
    "weights":  {"Wifi":0.30,"Quiet":0.25,"Seat":0.20,"Tasty":0.10,"Cheap":0.10,"Music":0.05},
    "rationale":"工作情境看中 Wifi/Quiet/Seat。"
  },
]

def examples_block() -> str:
    lines = []
    for ex in FEW_SHOT:
        lines.append("使用者需求：" + ex["user"])
        lines.append("示例輸出：")
        lines.append(json.dumps({
            "required": ex["required"],
            "weights": ex["weights"],
            "rationale": ex["rationale"]
        }, ensure_ascii=False))
        lines.append("")
    return "\n".join(lines)

# ======= 工具：標準化＋計分 =======
def normalize_1_to_5(x: Optional[float]) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v < 1: v = 1
    if v > 5: v = 5
    return v / 5.0

def compute_score(meta: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    norm = {k: normalize_1_to_5(meta.get(k)) for k in SCORING_KEYS}
    weighted = {k: norm[k] * float(weights.get(k, 0.0)) for k in SCORING_KEYS}
    score = sum(weighted.values())
    return {
        "metrics":   {k: meta.get(k, None) for k in SCORING_KEYS},
        "normalized":norm,
        "weights":   {k: float(weights.get(k, 0.0)) for k in SCORING_KEYS},
        "weighted":  {k: round(weighted[k], 6) for k in SCORING_KEYS},
        "Score":     round(score, 6),
    }

def metadata_match(meta: Dict[str, Any], req: Dict[str, Optional[str]]) -> bool:
    def contains(a: str, b: str) -> bool:
        return (a or "").strip() and (b or "").strip() and (b.strip() in a.strip())
    for k in ["Name", "Address", "City", "Open_time"]:
        if req.get(k) and not contains(str(meta.get(k,"")), str(req[k])):
            return False
    for k in ["Limited_time", "Socket", "Standing_desk"]:
        if req.get(k) and str(meta.get(k,"")).strip() != str(req[k]).strip():
            return False
    return True

def parse_require_and_weights(user_question: str) -> Tuple[Dict[str, Optional[str]], Dict[str, float], str]:
    schema_hint = """
只輸出 JSON：
{
  "required": {
    "Name": null | string, "City": null | string, "Address": null | string, "Open_time": null | string,
    "Limited_time": null | "不限時" | "限時" | "視情況而定",
    "Socket": null | "有插座" | "無插座" | "部分座位有插座",
    "Standing_desk": null | "有站立座位" | "無站立座位" | "可能有站立座位"
  },
  "weights": {"Quiet": number,"Tasty": number,"Cheap": number,"Seat": number,"Wifi": number,"Music": number},
  "rationale": string
}
要求：
- 六個權重 >=0 且總和=1（小數 3 位內）
- 未提到的因子仍須分配較低權重
- 僅輸出 JSON
""".strip()
    prompt = f"""你是咖啡廳推薦系統的權重規劃員。依使用者需求，同時輸出「必要條件」與「六項權重（總和=1）」。
範例：
{examples_block()}

{schema_hint}

使用者需求：{user_question}
"""
    resp = llm.predict(prompt)
    try:
        obj = json.loads(resp)
    except Exception:
        obj = {
            "required": {k: None for k in ["Name","City","Address","Open_time","Limited_time","Socket","Standing_desk"]},
            "weights":  {"Quiet":0.20,"Tasty":0.20,"Cheap":0.20,"Seat":0.20,"Wifi":0.10,"Music":0.10},
            "rationale":"回退：JSON 解析失敗，使用預設權重。"
        }
    required = obj.get("required", {})
    weights  = obj.get("weights", {})
    rationale = obj.get("rationale","")

    for k in ["Name","City","Address","Open_time","Limited_time","Socket","Standing_desk"]:
        required.setdefault(k, None)
    for k in SCORING_KEYS:
        try: weights[k] = float(weights.get(k,0))
        except: weights[k] = 0.0
    s = sum(max(0.0,v) for v in weights.values())
    if s <= 0:
        weights = {"Quiet":0.20,"Tasty":0.20,"Cheap":0.20,"Seat":0.20,"Wifi":0.10,"Music":0.10}
        rationale = (rationale + "（回退：權重總和<=0，改用預設）").strip()
    else:
        weights = {k: max(0.0,v)/s for k,v in weights.items()}
    weights = {k: round(v,3) for k,v in weights.items()}
    diff = round(1.0 - sum(weights.values()), 3)
    if abs(diff) >= 0.001:
        kmax = max(weights, key=lambda k: weights[k])
        weights[kmax] = round(weights[kmax] + diff, 3)
    return required, weights, rationale

def format_formula_string(weights: Dict[str, float]) -> str:
    def pct(x): return f"{round(100*x)}%"
    return (
        f"Score = {weights['Quiet']:.3f}·Quiet' + {weights['Tasty']:.3f}·Tasty' + "
        f"{weights['Cheap']:.3f}·Cheap' + {weights['Seat']:.3f}·Seat' + "
        f"{weights['Wifi']:.3f}·Wifi' + {weights['Music']:.3f}·Music'\n"
        f"（權重：Quiet {pct(weights['Quiet'])}, Tasty {pct(weights['Tasty'])}, "
        f"Cheap {pct(weights['Cheap'])}, Seat {pct(weights['Seat'])}, "
        f"Wifi {pct(weights['Wifi'])}, Music {pct(weights['Music'])}）"
    )

# ======= API =======
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}
    user_question = (data.get("question") or "").strip()
    debug = bool(data.get("debug", True))
    if not user_question:
        return jsonify({"error":"Missing question"}), 400

    required, weights, rationale = parse_require_and_weights(user_question)

    all_docs = db.similarity_search(user_question, k=50)
    strict = [d for d in all_docs if metadata_match(d.metadata, required)]
    if len(strict) >= 5:
        candidates, partial = strict[:10], False
    else:
        need = 10 - len(strict)
        rest = [d for d in all_docs if d not in strict][:need]
        candidates, partial = strict + rest, True

    if not candidates:
        return jsonify({"required": required, "weights": weights,
                        "answer": "很抱歉，找不到合適的候選資料。"})

    scored: List[Dict[str, Any]] = []
    for doc in candidates:
        meta = doc.metadata
        detail = compute_score(meta, weights)
        scored.append({
            "Name": meta.get("Name","未知名稱"),
            "City": meta.get("City"),
            "Address": meta.get("Address"),
            "Open_time": meta.get("Open_time"),
            "Limited_time": meta.get("Limited_time"),
            "Socket": meta.get("Socket"),
            "Standing_desk": meta.get("Standing_desk"),
            "Mrt": meta.get("Mrt"),
            "Url": meta.get("Url"),
            "score_detail": detail,
        })
    scored.sort(key=lambda x: x["score_detail"]["Score"], reverse=True)
    top5 = scored[:5]

    header = ("⚠️ 找不到完全符合所有必要條件的 5 間；以下為盡量符合的推薦：\n"
              if partial else "✅ 根據您的需求之推薦：\n")
    header += "\n【本次推薦公式】\n" + format_formula_string(weights) + "\n"
    if rationale: header += f"【權重理由】{rationale}\n"

    blocks = []
    for i, row in enumerate(top5, 1):
        d = row["score_detail"]
        reasons = []
        for k in ["Quiet","Seat","Wifi","Tasty","Cheap"]:
            if d["normalized"][k] >= 0.8:
                reasons.append(f"{k}表現佳")
        reason_text = "、".join(reasons) if reasons else "整體表現均衡"
        block = [
            f"第 {i} 名：{row['Name']}（總分 {d['Score']:.3f}）",
            f"地點：{row.get('City','')} | {row.get('Address','')}",
            f"營業：{row.get('Open_time','')} | 限時：{row.get('Limited_time','')} | 插座：{row.get('Socket','')} | 站立：{row.get('Standing_desk','')}",
            f"推薦理由：{reason_text}"
        ]
        if debug:
            block.append("")
            block.append("加權分數明細：")
            block.append("Factor | Raw | Normalized | Weight | Weighted")
            for k in SCORING_KEYS:
                raw = d["metrics"].get(k, None); norm = d["normalized"][k]
                w = d["weights"][k]; ww = d["weighted"][k]
                block.append(f"{k} | {raw} | {norm:.3f} | {w:.3f} | {ww:.3f}")
            block.append(f"Score：{d['Score']:.3f}")
        blocks.append("\n".join(block))

    answer = header + "\n\n".join(blocks)
    return jsonify({"required": required, "weights": weights, "top5": top5, "answer": answer})

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "5000"))  # Render 會注入 PORT
    app.run(host="0.0.0.0", port=port, debug=False)

