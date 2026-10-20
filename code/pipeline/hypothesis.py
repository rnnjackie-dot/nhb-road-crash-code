"""
交通事故知识图谱 — 假设生成模块
对每种事故类型读取对应案例output，调用DeepSeek生成研究假设
评估维度：新颖性 / 实用性 / 可验证性
"""

import sys
import json
import time
from pathlib import Path
import os
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
HYPOTHESIS_DIR = Path(os.environ.get("HYPOTHESIS_DIR", "hypotheses"))
HYPOTHESIS_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def load_cases_by_type() -> dict:
    """按事故类型分组加载案例数据"""
    groups = {}
    for f in sorted(OUTPUT_DIR.glob("OCR-[0-9][0-9][0-9].json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        t = data["meta"]["accident_type"]
        groups.setdefault(t, []).append(data)
    return groups


def summarize_case(case: dict) -> dict:
    """提取案例的关键信���用于假设生成"""
    meta = case.get("meta", {})
    nodes = case.get("nodes", {})

    traits = [
        {
            "value": n["properties"].get("value"),
            "dimension": n["properties"].get("dimension"),
            "theory_reference": n["properties"].get("theory_reference", []),
        }
        for n in nodes.get("traits", [])
    ]
    tendencies = [
        {
            "label": n["properties"].get("tendency_label"),
            "mechanism": n["properties"].get("psychological_mechanism"),
            "theory_reference": n["properties"].get("theory_reference", []),
            "behavior_chain": n["properties"].get("behavior_chain", []),
        }
        for n in nodes.get("behavioral_tendencies", [])
    ]
    combinations = [
        {
            "label": n["properties"].get("combination_label"),
            "risk_level": n["properties"].get("risk_level"),
            "trait_values": n["properties"].get("trait_values", []),
            "theory_reference": n["properties"].get("theory_reference", []),
        }
        for n in nodes.get("trait_combinations", [])
    ]
    causal_factors = [
        {
            "label": n["properties"].get("factor_label"),
            "level": n["properties"].get("factor_level"),
            "description": n["properties"].get("description"),
        }
        for n in nodes.get("causal_factors", [])
    ]

    return {
        "case_id": meta.get("case_id"),
        "testimony_summary": meta.get("testimony_summary"),
        "primary_cause": meta.get("primary_cause_hypothesis"),
        "traits": traits,
        "tendencies": tendencies,
        "combinations": combinations,
        "causal_factors": causal_factors,
    }


def build_hypothesis_prompt(accident_type: str, cases_summary: list[dict]) -> str:
    cases_json = json.dumps(cases_summary, ensure_ascii=False, indent=2)
    return f"""你是一名交通心理学研究专家，擅长从案例数据中生成高质量研究假设。

以下是【{accident_type}】事故类型的结构化知识图谱数据，来自真实案例提取：

{cases_json}

━━━━━━━━━━━━━━━━��━━━━━━━━━━━━
【任务：生成研究假设】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请基于上述数据，生成5条研究假设。每条假设须满足：

1. 新颖性：不是已有文献的简单重复，需从数据中发现新的变量关系或交互效应
2. 实用性：对交通安全干预、驾驶员培训或政策制定有直接指导价值
3. 可验证性：可通过问卷、实验或田野研究加以检验

每条假设包含：
- hypothesis_id: H{accident_type[:2]}-01 ~ 05
- statement: 假设陈述（"当X时，Y会导致Z"格式，≤50字）
- rationale: 从数据中的依据（引用具体trait/tendency/theory_reference，≤80字）
- testability: 可验证性描述（建议研究方法，≤40字）
- variable_x: 自变量
- variable_y: 因变量或调节变量
- related_theories: 关联理论列表
- intervention_suggestion: 基于此假设的干预建议（≤40字）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【输出格式】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "accident_type": "{accident_type}",
  "hypotheses": [
    {{
      "hypothesis_id": "H-01",
      "statement": "假设陈述",
      "rationale": "数据依据",
      "testability": "建议研究方法",
      "variable_x": "自变量",
      "variable_y": "因变量",
      "related_theories": ["理论名称"],
      "intervention_suggestion": "干预建议"
    }}
  ],
  "cross_case_patterns": "两个案例间共同规律的总结（≤60字）",
  "unique_insights": "该事故类型区别于其他类型的独特心理机制（≤60字）",
  "priority_hypothesis": "H-01~05中最具研究价值的假设编号及理由（≤40字）"
}}

仅输出合法JSON，不得输出任何JSON之外的内容。
"""


def call_deepseek(prompt: str, label: str) -> dict | None:
    print(f"  → 调用 DeepSeek API: {label}")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 稍高温度增加假设多样性
            max_tokens=8192,
        )
        raw = resp.choices[0].message.content.strip()
        # 去掉markdown代码块
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [FAIL] JSON解析失败 ({label}): {e}")
        (HYPOTHESIS_DIR / f"{label}_raw.txt").write_text(raw, encoding="utf-8")
        return None
    except Exception as e:
        print(f"  [FAIL] API调用失败 ({label}): {e}")
        return None


def main():
    print("=== 交通事故假设生成系统 ===\n")

    groups = load_cases_by_type()
    print(f"共 {len(groups)} 种事故类型：{list(groups.keys())}\n")

    all_hypotheses = {}

    for i, (accident_type, cases) in enumerate(groups.items()):
        label = accident_type.replace("/", "_")
        out_file = HYPOTHESIS_DIR / f"{label}_hypotheses.json"

        if out_file.exists():
            print(f"[{i+1}/{len(groups)}] {accident_type} 已存在，跳过")
            all_hypotheses[accident_type] = json.loads(out_file.read_text(encoding="utf-8"))
            continue

        print(f"[{i+1}/{len(groups)}] 生成假设：{accident_type}（{len(cases)} 个案例）")

        cases_summary = [summarize_case(c) for c in cases]
        prompt = build_hypothesis_prompt(accident_type, cases_summary)
        result = call_deepseek(prompt, label)

        if result:
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            h_count = len(result.get("hypotheses", []))
            priority = result.get("priority_hypothesis", "")
            print(f"  [OK] 生成 {h_count} 条假设，优先假设：{priority}")
            all_hypotheses[accident_type] = result
        else:
            print(f"  [FAIL] {accident_type} 假设生成失败")

        if i < len(groups) - 1:
            time.sleep(1)

    # 生成跨类型综合报告
    print("\n生成跨类型综合假设报告...")
    summary_file = HYPOTHESIS_DIR / "all_hypotheses_summary.json"
    summary = {
        "total_accident_types": len(all_hypotheses),
        "total_hypotheses": sum(len(v.get("hypotheses", [])) for v in all_hypotheses.values()),
        "by_type": {
            t: {
                "hypothesis_count": len(v.get("hypotheses", [])),
                "priority": v.get("priority_hypothesis"),
                "cross_case_patterns": v.get("cross_case_patterns"),
                "unique_insights": v.get("unique_insights"),
                "top_hypotheses": [
                    {
                        "id": h["hypothesis_id"],
                        "statement": h["statement"],
                        "testability": h.get("testability"),
                        "intervention": h.get("intervention_suggestion"),
                    }
                    for h in v.get("hypotheses", [])[:2]
                ],
            }
            for t, v in all_hypotheses.items()
        },
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 完成 ===")
    print(f"假设目录: {HYPOTHESIS_DIR}")
    print(f"总假设数: {summary['total_hypotheses']}")
    print(f"综合报告: {summary_file}")


if __name__ == "__main__":
    main()
