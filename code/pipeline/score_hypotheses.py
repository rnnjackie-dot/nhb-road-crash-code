"""
对 LLMKG (_hypotheses.json) 与 LLM (_direct.json) 两类假设
调用 DeepSeek API 进行新颖性 & 创新性评分，（如果可以，希望可以多使用几个模型进行调用）
然后在两组合并后做 z-score 标准化，输出汇总结果。
"""
import json
import sys
import time
from pathlib import Path
import os

import numpy as np
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL    = "deepseek-chat"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

HYP_DIR = Path(os.environ.get("HYP_DIR", "hypotheses"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "data"))

# ── 1. 加载假设 ───────────────────────────────────────────────

def load_llmkg_hypotheses() -> list[dict]:
    """读取 *_hypotheses.json，来源标记 LLMKG"""
    records = []
    for fp in sorted(HYP_DIR.glob("*_hypotheses.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        accident_type = data.get("accident_type") or fp.stem.replace("_hypotheses", "")
        for hyp in data.get("hypotheses", []):
            records.append({
                "source": "LLMKG",
                "accident_type": accident_type,
                "file": fp.name,
                "id": hyp.get("hypothesis_id") or hyp.get("id", ""),
                "statement": hyp.get("statement", ""),
            })
    return records


def load_llm_hypotheses() -> list[dict]:
    """读取 *_direct.json，来源标记 LLM"""
    records = []
    for fp in sorted(HYP_DIR.glob("*_direct.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        accident_type = data.get("accident_type") or fp.stem.replace("_direct", "")
        for hyp in data.get("hypotheses", []):
            records.append({
                "source": "LLM",
                "accident_type": accident_type,
                "file": fp.name,
                "id": hyp.get("id", ""),
                "statement": hyp.get("statement", ""),
            })
    return records


# ── 2. DeepSeek 评分 ──────────────────────────────────────────

SCORE_PROMPT = """\
你是一位交通安全领域的学术专家，请对以下研究假设进行评分。

【假设陈述】
{statement}

请从两个维度给出 1-10 的整数评分（1=最低，10=最高）：
1. 新颖性（novelty）：该假设提出的视角或变量组合在交通安全文献中的独特程度。
2. 创新性（innovativeness）：该假设对现有理论或实践的突破与贡献程度。

只输出如下 JSON，不要任何额外说明：
{{"novelty": <1-10>, "innovativeness": <1-10>}}
"""


def score_one(statement: str, retries: int = 3) -> dict:
    prompt = SCORE_PROMPT.format(statement=statement)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            text = resp.choices[0].message.content.strip()
            # 提取 JSON
            start = text.find("{")
            end   = text.rfind("}") + 1
            scores = json.loads(text[start:end])
            novelty      = int(scores["novelty"])
            innovativeness = int(scores["innovativeness"])
            assert 1 <= novelty <= 10 and 1 <= innovativeness <= 10
            return {"novelty": novelty, "innovativeness": innovativeness}
        except Exception as e:
            print(f"  [警告] 评分失败（第{attempt+1}次）: {e}")
            time.sleep(2)
    return {"novelty": None, "innovativeness": None}


def score_all(records: list[dict]) -> list[dict]:
    for i, rec in enumerate(records, 1):
        print(f"  [{i}/{len(records)}] {rec['source']} | {rec['accident_type']} | {rec['id']}")
        scores = score_one(rec["statement"])
        rec["novelty_raw"]        = scores["novelty"]
        rec["innovativeness_raw"] = scores["innovativeness"]
        time.sleep(0.5)   # 避免限速
    return records


# ── 3. Z-score 标准化（两组合并后计算） ───────────────────────

def zscore_normalize(records: list[dict]) -> list[dict]:
    """
    对 novelty_raw / innovativeness_raw 在全部记录（LLMKG+LLM）上
    分别做 z-score 标准化，结果写入 novelty_z / innovativeness_z。
    """
    for dim in ("novelty", "innovativeness"):
        raw_key = f"{dim}_raw"
        z_key   = f"{dim}_z"
        vals = np.array(
            [r[raw_key] for r in records if r[raw_key] is not None],
            dtype=float,
        )
        mean = vals.mean()
        std  = vals.std(ddof=0)   # 总体标准差
        print(f"[{dim}] 均值={mean:.3f}  标准差={std:.3f}")
        for rec in records:
            raw = rec[raw_key]
            if raw is None or std == 0:
                rec[z_key] = None
            else:
                rec[z_key] = round((raw - mean) / std, 4)
    return records


# ── 4. 汇总统计 ───────────────────────────────────────────────

def group_summary(records: list[dict]) -> dict:
    result = {}
    for source in ("LLMKG", "LLM"):
        grp = [r for r in records if r["source"] == source]
        for dim in ("novelty", "innovativeness"):
            raws = [r[f"{dim}_raw"] for r in grp if r[f"{dim}_raw"] is not None]
            zs   = [r[f"{dim}_z"]   for r in grp if r[f"{dim}_z"]   is not None]
            result.setdefault(source, {})[dim] = {
                "mean_raw": round(float(np.mean(raws)), 4) if raws else None,
                "std_raw":  round(float(np.std(raws, ddof=0)), 4) if raws else None,
                "mean_z":   round(float(np.mean(zs)),  4) if zs  else None,
                "std_z":    round(float(np.std(zs, ddof=0)),  4) if zs  else None,
            }
    return result


# ── 5. 主流程 ─────────────────────────────────────────────────

def main():
    print("=== 加载假设 ===")
    llmkg = load_llmkg_hypotheses()
    llm   = load_llm_hypotheses()
    print(f"LLMKG 假设数: {len(llmkg)}")
    print(f"LLM   假设数: {len(llm)}")
    all_records = llmkg + llm

    print("\n=== DeepSeek 评分 ===")
    print("--- LLMKG ---")
    score_all(llmkg)
    print("--- LLM ---")
    score_all(llm)

    print("\n=== Z-score 标准化（全局） ===")
    zscore_normalize(all_records)

    print("\n=== 分组统计 ===")
    summary = group_summary(all_records)
    for src, dims in summary.items():
        print(f"\n[{src}]")
        for dim, stats in dims.items():
            print(f"  {dim}: raw均值={stats['mean_raw']} raw标准差={stats['std_raw']} "
                  f"| z均值={stats['mean_z']} z标准差={stats['std_z']}")

    # ── 保存结果 ──
    out_detail = OUT_DIR / "hypothesis_scores.json"
    out_summary = OUT_DIR / "hypothesis_scores_summary.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_detail.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_summary.write_text(
        json.dumps({"group_summary": summary, "all_records_count": len(all_records)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== 完成 ===")
    print(f"明细: {out_detail}")
    print(f"汇总: {out_summary}")


if __name__ == "__main__":
    main()
