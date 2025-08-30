import requests
import json

def print_detail(rank, cafe):
    name = cafe.get("Name", "未知")
    meta = cafe
    d = cafe["score_detail"]

    print(f"\n第 {rank} 名：{name}（總分 {d['Score']:.3f}）")
    print(f"地點：{meta.get('City','')} | {meta.get('Address','')}")
    print(f"營業：{meta.get('Open_time','')} | 限時：{meta.get('Limited_time','')} | 插座：{meta.get('Socket','')} | 站立：{meta.get('Standing_desk','')}")
    print("加權分數明細：")
    print("Factor        Raw   Normalized   Weight   Weighted")
    for k in ["Quiet","Tasty","Cheap","Seat","Wifi","Music"]:
        raw = d["metrics"].get(k, None)
        norm = d["normalized"][k]
        w = d["weights"][k]
        ww = d["weighted"][k]
        print(f"{k:<12} {str(raw):<5} {norm:>10.3f}   {w:>6.2f}   {ww:>8.3f}")

def main():
    question = input("請輸入您的問題：").strip()
    debug_in = input("是否顯示加權明細？(y/n, 預設 y): ").strip().lower()
    debug = (debug_in != "n")

    url = "http://127.0.0.1:5000/ask"
    payload = {"question": question, "debug": debug}
    resp = requests.post(url, json=payload, timeout=60)

    print("\n✅ 查詢完成")
    print("Status Code:", resp.status_code)

    try:
        data = resp.json()
    except Exception:
        print("\n⚠️ 回傳非 JSON：\n", resp.text[:500])
        return

    # 先印可直接顯示的答案
    print("\n──────── 回覆（可直接顯示） ────────")
    print(data.get("answer","<無>"))

    # 再印結構化結果（方便檢查）
    if "required" in data:
        print("\n──────── 解析到的必要條件 ────────")
        print(json.dumps(data["required"], ensure_ascii=False, indent=2))

    if "top3" in data:
        print("\n──────── 前 3 名（結構化＋明細） ────────")
        for i, cafe in enumerate(data["top3"], 1):
            print_detail(i, cafe)

if __name__ == "__main__":
    main()


    