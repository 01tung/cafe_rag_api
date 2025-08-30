# build_rag.py — upload cafes to Qdrant Cloud (fixed: use url/api_key instead of client)
import os
import json
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.vectorstores import Qdrant as QdrantVS

# ---------- env ----------
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
COLLECTION = "cafes_v1"  # 與 app.py 保持一致

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("缺少 QDRANT_URL 或 QDRANT_API_KEY，請確認 .env 已設定")

# ---------- helpers ----------
def load_json_any(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 支援：1) {"data":[...]} 2) 直接是 list
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        return raw["data"]
    if isinstance(raw, list):
        return raw
    raise ValueError("無法在 JSON 找到資料列（期待 {'data':[...]} 或 直接就是 [...]）")

def s(item: Dict[str, Any], key: str, default: str = "無資料") -> str:
    v = item.get(key, None)
    return default if v in (None, "") else str(v).strip()

def f(item: Dict[str, Any], key: str) -> Optional[float]:
    v = item.get(key, None)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None

def make_page(meta: Dict[str, Any]) -> str:
    # 簡單中文文本，利於語意檢索
    lines = [
        f"咖啡廳名稱：{meta.get('Name','')}",
        f"城市：{meta.get('City','')}",
        f"地址：{meta.get('Address','')}",
        f"營業時間：{meta.get('Open_time','')}",
        f"限時：{meta.get('Limited_time','')}",
        f"插座：{meta.get('Socket','')}",
        f"站立座位：{meta.get('Standing_desk','')}",
        f"安靜：{meta.get('Quiet','')}",
        f"美味：{meta.get('Tasty','')}",
        f"便宜：{meta.get('Cheap','')}",
        f"座位：{meta.get('Seat','')}",
        f"WiFi：{meta.get('Wifi','')}",
        f"音樂：{meta.get('Music','')}",
        f"捷運：{meta.get('Mrt','')}",
        f"網址：{meta.get('Url','')}",
    ]
    return "\n".join(lines)

# ---------- main ----------
def main():
    data = load_json_any("cafes.json")
    print(f"載入 cafes.json：{len(data)} 筆")

    docs: List[Document] = []
    skipped = 0
    for i, item in enumerate(data, 1):
        name = s(item, "name") if "name" in item else s(item, "Name")
        if name == "無資料":
            skipped += 1
            continue

        meta: Dict[str, Any] = {
            "ID": s(item, "id", f"item_{i}"),
            "Name": name,
            "City": s(item, "city") if "city" in item else s(item, "City"),
            "Address": s(item, "address") if "address" in item else s(item, "Address"),
            "Open_time": s(item, "open_time") if "open_time" in item else s(item, "Open_time"),
            "Limited_time": s(item, "limited_time") if "limited_time" in item else s(item, "Limited_time"),
            "Socket": s(item, "socket") if "socket" in item else s(item, "Socket"),
            "Standing_desk": s(item, "standing_desk") if "standing_desk" in item else s(item, "Standing_desk"),
            "Mrt": s(item, "mrt") if "mrt" in item else s(item, "Mrt"),
            "Url": s(item, "url") if "url" in item else s(item, "Url"),
        }
        # 六個打分欄位（1~5；可為 None）
        meta["Quiet"] = f(item, "quiet") if "quiet" in item else f(item, "Quiet")
        meta["Tasty"] = f(item, "tasty") if "tasty" in item else f(item, "Tasty")
        meta["Cheap"] = f(item, "cheap") if "cheap" in item else f(item, "Cheap")
        meta["Seat"]  = f(item, "seat")  if "seat"  in item else f(item, "Seat")
        meta["Wifi"]  = f(item, "wifi")  if "wifi"  in item else f(item, "Wifi")
        meta["Music"] = f(item, "music") if "music" in item else f(item, "Music")

        docs.append(Document(page_content=make_page(meta), metadata=meta))
        if i % 100 == 0:
            print(f"  轉換進度 {i}/{len(data)}")

    print(f"轉換完成，共 {len(docs)} 筆；跳過 {skipped} 筆")

    # 1) 先用官方 qdrant_client 管理 collection（明確指定 384 維 + COSINE）
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    # 2) 再用 LangChain 封裝上傳（**改為使用 url / api_key，不再傳 client=**）
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    QdrantVS.from_documents(
        documents=docs,
        embedding=embedding,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION,
    )

    print(f"\n✅ 已上傳到 Qdrant：{COLLECTION}，共 {len(docs)} 筆")

if __name__ == "__main__":
    main()
