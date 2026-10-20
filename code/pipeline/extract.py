"""
交通事故知识图谱提取系统
- 读取 OCR 结果，调用 DeepSeek API 提取结构化信息
- 增量构建知识图谱 JSON
"""

import sys
import json
import re
import time
from pathlib import Path
import os
sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI
import docx

# ── 配置 ──────────────────────────────────────────────
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

THEORY_DOCX = Path(os.environ.get("THEORY_DOCX", "theory.docx"))
OCR_DIR = Path(os.environ.get("OCR_DIR", "ocr_results"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
KG_FILE = OUTPUT_DIR / "knowledge_graph.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ── 读取理论文档 ──────────────────────────────────────
def read_docx(path: str) -> list:
    """读取 .docx 文件，返回段落文本列表"""
    doc = docx.Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def parse_theories(paragraphs: list) -> str:
    """将段落列表拼接为理论字符串，注入 prompt"""
    return "\n".join(paragraphs)


# ── 读取文档 ──────────────────────────────────────────
def parse_cases_from_ocr(ocr_dir: Path) -> list:
    """读取 ocr_results/*.txt，每个文件为一个案例"""
    cases = []
    for i, txt_file in enumerate(sorted(ocr_dir.glob("*.txt")), 1):
        text = txt_file.read_text(encoding="utf-8").strip()
        if len(text) < 100:
            continue
        cases.append({
            "case_id": f"OCR-{i:03d}",
            "accident_type": "待推断",
            "driver_name": "（已脱敏）",
            "testimony": text,
            "source": txt_file.stem,
        })
    return cases


# ── 知识图谱状态管理 ──────────────────────────────────
def load_kg() -> dict:
    if KG_FILE.exists():
        return json.loads(KG_FILE.read_text(encoding="utf-8"))
    return {
        "nodes": {
            "traits": [],
            "trait_combinations": [],
            "behavioral_tendencies": [],
            "behaviors": [],
            "causal_factors": [],
        },
        "relationships": [],
        "cases": [],
    }


def save_kg(kg: dict):
    KG_FILE.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")


def get_existing_snapshot(kg: dict) -> dict:
    """提取已有节点的快照字符串，用于注入prompt"""
    def fmt(nodes, key):
        return [f"{n['properties'].get('merge_key', '')} | {n['properties'].get(key, '')}"
                for n in nodes]

    return {
        "traits": fmt(kg["nodes"]["traits"], "value"),
        "combos": fmt(kg["nodes"]["trait_combinations"], "combination_label"),
        "tendencies": fmt(kg["nodes"]["behavioral_tendencies"], "tendency_label"),
        "behaviors": fmt(kg["nodes"]["behaviors"], "behavior_label"),
        "causal_factors": fmt(kg["nodes"]["causal_factors"], "factor_label"),
    }


# ── Prompt 构建 ───────────────────────────────────────
def build_prompt(case: dict, snapshot: dict, theories_str: str = "") -> str:
    def list_to_str(lst): return "\n".join(f"  - {x}" for x in lst) if lst else "  （暂无）"

    case_id = case["case_id"]
    accident_type = case["accident_type"]
    testimony = case["testimony"]

    existing_traits_str = list_to_str(snapshot["traits"])
    existing_combos_str = list_to_str(snapshot["combos"])
    existing_tendencies_str = list_to_str(snapshot["tendencies"])
    existing_behaviors_str = list_to_str(snapshot["behaviors"])
    existing_causal_factors_str = list_to_str(snapshot["causal_factors"])

    return f"""你是一名交通事故行为心理分析专家，精通知识图谱结构化提取。
请对以下【交通肇事口供】进行结构化分析，输出可存档的JSON知识图谱数据。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【交通心理学理论参考库 — 节点标注依据】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

以下为50条交通心理学理论，提取节点时须在 theory_reference 字段标注最匹配的理论编号和名称（可多个）：

{theories_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【增量学习约束 — 最高优先级】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本系统为增量知识图谱，规则如下：
  ① 字面匹配已有节点 → 直接复用，is_new: false
  ② 语义相似度 > 0.85 → 复用已有节点，is_new: false
  ③ 确实不存在 → 新建，is_new: true

当前知识库快照（高频节点优先匹配）：

■ 已有 Trait 节点：
{existing_traits_str}

■ 已有 TraitCombination 节点：
{existing_combos_str}

■ 已有 BehavioralTendency 节点：
{existing_tendencies_str}

■ 已有 Behavior 节点：
{existing_behaviors_str}

■ 已有 CausalFactor 节点：
{existing_causal_factors_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【口供原文】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

案例编号：{case_id}
事故类型：{accident_type}

{testimony}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【分析任务】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 任务A：驾驶员个体特征（Trait节点）

按以下七个维度逐一提取，有则输出，无则跳过。

### A1 驾驶前身体状态
提取项：疲劳程度、睡眠情况、健康状况、用药情况
语义映射：
  - "打瞌睡"/"眼皮重"/"撑不住" → 急性疲劳发作
  - "昨晚没睡"/"熬夜" → 睡眠剥夺
  - "开了很久" → 连续驾驶疲劳

### A2 驾驶前心理与情绪状态
提取项：情绪类型、情绪强度、情绪对驾驶决策的影响方向
情绪类型标签（优先使用）：
  焦虑型、愤怒型、急躁型、低落型、兴奋型、平稳型

### A3 风险感知与认知偏差
提取项：事前风险评估方式、认知偏差类型
认知偏差标签（优先使用）：
  - 乐观偏差、控制幻觉、习惯化偏差、时间折扣偏差、可得性启发
语义映射：
  - "经常走这条路" → 习惯化偏差
  - "就这一次"/"不会那么巧" → 乐观偏差
  - "感觉没事"/"能行" → 控制幻觉
  - "赶时间/赶回去" → 时间折扣偏差

### A4 驾驶动机与行程压力
提取项：本次驾驶的目的、时间压力来源、外部催促因素
压力来源标签（优先使用）：
  职业期限压力、人际催促压力、经济激励压力、自我设定时限

### A5 速度与车辆控制行为意图
提取项：主观速度判断、超速动机、车辆操控意图
语义映射：
  - "没注意速度" → 无意识超速
  - "习惯开快" → 习惯性超速
  - "路上没人/很空" → 情境判断型超速

### A6 注意力分配状态
提取项：注意力分散来源、专注程度描述
分心类型标签（优先使用）：
  - 手机操作型、车内社交型、思维漫游型、导航操作型、视觉分心型

### A7 人口与经验特征
提取项：年龄段、性别（可推断时）、驾龄、职业驾驶背景

━━━━━━━━━
Trait节点格式：
{{
  "neo4j_label": "Trait",
  "merge_key": "{{category}}:{{value}}",
  "properties": {{
    "dimension": "A1~A7",
    "category": "特征类别",
    "value": "特征值（≤10字）",
    "theory_reference": ["理论编号: 理论名称"],
    "source_snippet": "口供原文片段（≤20字）",
    "confidence": 0.0~1.0,
    "is_new": true/false,
    "matched_existing_id": "已有节点ID或null"
  }}
}}
━━━━━━━━━

## 任务B：特征组合（TraitCombination节点）

识别具有联合解释力的特征共现组合（至少2个Trait）。

━━━━━━━━━
TraitCombination节点格式：
{{
  "neo4j_label": "TraitCombination",
  "merge_key": "{{combination_label}}",
  "properties": {{
    "combination_label": "组合名称（≤12字）",
    "trait_values": ["特征值1", "特征值2"],
    "accident_type_scope": ["适用事故类型"],
    "risk_level": "low/medium/high/critical",
    "theory_reference": ["理论编号: 理论名称"],
    "source_snippet": "口供原文片段（≤25字）",
    "confidence": 0.0~1.0,
    "is_new": true/false,
    "matched_existing_id": "已有节点ID或null"
  }}
}}
━━━━━━━━━

## 任务C：行为倾向（BehavioralTendency节点）

━━━━━━━━━
BehavioralTendency节点格式：
{{
  "neo4j_label": "BehavioralTendency",
  "merge_key": "{{tendency_label}}",
  "properties": {{
    "tendency_label": "倾向名称（≤10字）",
    "psychological_mechanism": "核心心理机制（1句话）",
    "theory_reference": ["理论编号: 理论名称"],
    "behavior_chain": [
      "Step1: 触发情境",
      "Step2: 认知状态",
      "Step3: 情绪/动机",
      "Step4: 驾驶决策",
      "Step5: 行为与结果"
    ],
    "accident_type_scope": ["适用事故类型"],
    "source_snippet": "口供原文片段（≤25字）",
    "confidence": 0.0~1.0,
    "is_new": true/false,
    "matched_existing_id": "已有节点ID或null"
  }}
}}
━━━━━━━━━

## 任务D：具体行为事件（Behavior节点）

behavior_type 标准值：
  pre_drive_decision / speed_management / attention_lapse /
  vehicle_control_error / hazard_response_failure / post_accident_behavior

━━━━━━━━━
Behavior节点格式：
{{
  "neo4j_label": "Behavior",
  "merge_key": "{{behavior_label}}",
  "properties": {{
    "behavior_label": "行为标签（动宾结构≤8字）",
    "behavior_type": "标准类型值",
    "triggered_by_traits": ["Trait的value值"],
    "triggered_by_tendencies": ["BehavioralTendency的tendency_label值"],
    "outcome": "直接后果（≤20字）",
    "counterfactual": "yes/likely/unclear",
    "source_snippet": "口供原文片段（≤20字）",
    "confidence": 0.0~1.0,
    "is_new": true/false,
    "matched_existing_id": "已有节点ID或null"
  }}
}}
━━━━━━━━━

## 任务E：结构性致因因素（CausalFactor节点）

factor_level 标准值：
  road_design / regulatory / occupational / cultural / vehicle

━━━━━━━━━
CausalFactor节点格式：
{{
  "neo4j_label": "CausalFactor",
  "merge_key": "{{factor_label}}",
  "properties": {{
    "factor_label": "致因标签（≤12字）",
    "factor_level": "标准层级值",
    "description": "该因素如何促成事故（1~2句话）",
    "accident_type_scope": ["适用事故类型"],
    "source_snippet": "口供原文片段或推断依据（≤20字）",
    "confidence": 0.0~1.0,
    "is_new": true/false,
    "matched_existing_id": "已有节点ID或null"
  }}
}}
━━━━━━━━━

## 任务F：关系边（Relationships）

可用关系类型：
  HAS_TRAIT / COMBINED_WITH / TRIGGERS / MANIFESTS_AS /
  LEADS_TO / CONTRIBUTES_TO / AMPLIFIES

━━━━━━━━━
Relationship格式：
{{
  "rel_type": "标准关系类型",
  "from_label": "起始节点neo4j_label",
  "from_merge_key": "起始节点merge_key值",
  "to_label": "终止节点neo4j_label",
  "to_merge_key": "终止节点merge_key值",
  "properties": {{
    "weight": 0.0~1.0,
    "evidence": "口供原文支持（≤15字）",
    "is_inferred": true/false
  }}
}}
━━━━━━━━━

## 任务G：增量更新记录

{{
  "new_nodes_summary": {{
    "traits": ["新增Trait的value列表"],
    "combinations": ["新增TraitCombination的combination_label列表"],
    "tendencies": ["新增BehavioralTendency的tendency_label列表"],
    "behaviors": ["新增Behavior的behavior_label列表"],
    "causal_factors": ["新增CausalFactor的factor_label列表"]
  }},
  "validated_patterns": [
    {{"node_merge_key": "被验证节点的merge_key", "validation_strength": "strong/moderate/weak"}}
  ],
  "suggested_merges": [
    {{"node_a_merge_key": "节点A", "node_b_merge_key": "节点B", "similarity_estimate": 0.0, "merge_recommendation": "建议合并/建议保留区分/需人工判断"}}
  ],
  "knowledge_gap": "现有知识库未能覆盖的新现象（无则填null）"
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【完整输出结构】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "meta": {{
    "case_id": "{case_id}",
    "accident_type": "{accident_type}",
    "testimony_summary": "一句话概括核心事实（≤30字）",
    "primary_cause_hypothesis": "最可能的主因（≤20字）",
    "confidence_overall": 0.0~1.0
  }},
  "nodes": {{
    "traits": [ ...任务A... ],
    "trait_combinations": [ ...任务B... ],
    "behavioral_tendencies": [ ...任务C... ],
    "behaviors": [ ...任务D... ],
    "causal_factors": [ ...任务E... ]
  }},
  "relationships": [ ...任务F... ],
  "incremental_update": {{ ...任务G... }}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【全局约束】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. confidence < 0.5 的节点不得输出
2. source_snippet 必须来自口供原文，不得编造
3. is_new 为布尔值，matched_existing_id 为字符串或 null
4. merge_key 在同类型节点内全局唯一
5. 节点标签风格须与已有知识库一致（中文简洁标签）
6. behavior_chain 必须为有序数组，不得合并步骤
7. 仅输出合法JSON，不得输出任何JSON之外的内容
"""

# ── API 调用 ──────────────────────────────────────────
def call_deepseek(prompt: str, case_id: str) -> dict | None:
    print(f"  → 调用 DeepSeek API: {case_id}")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=8192,
        )
        raw = resp.choices[0].message.content.strip()
        # 去掉可能的 markdown 代码块包裹
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [FAIL] JSON解析失败 ({case_id}): {e}")
        # 保存原始响应供调试
        (OUTPUT_DIR / f"{case_id}_raw.txt").write_text(raw, encoding="utf-8")
        return None
    except Exception as e:
        print(f"  [FAIL] API调用失败 ({case_id}): {e}")
        return None


# ── 增量合并 ──────────────────────────────────────────
def merge_nodes(kg: dict, result: dict, case_id: str):
    """将单个案例的提取结果合并进全局知识图谱"""
    node_map = {
        "traits": "traits",
        "trait_combinations": "trait_combinations",
        "behavioral_tendencies": "behavioral_tendencies",
        "behaviors": "behaviors",
        "causal_factors": "causal_factors",
    }

    for result_key, kg_key in node_map.items():
        for node in result.get("nodes", {}).get(result_key, []):
            mk = node.get("merge_key") or node.get("properties", {}).get("merge_key")
            if not mk:
                continue
            # 检查是否已存在
            existing = next(
                (n for n in kg["nodes"][kg_key]
                 if (n.get("merge_key") or n.get("properties", {}).get("merge_key")) == mk),
                None
            )
            if existing:
                # 更新 case_ids
                existing.setdefault("case_ids", [])
                if case_id not in existing["case_ids"]:
                    existing["case_ids"].append(case_id)
            else:
                node["case_ids"] = [case_id]
                kg["nodes"][kg_key].append(node)

    # 合并关系（去重）
    existing_rels = {
        (r["rel_type"], r["from_merge_key"], r["to_merge_key"])
        for r in kg["relationships"]
    }
    for rel in result.get("relationships", []):
        key = (rel.get("rel_type"), rel.get("from_merge_key"), rel.get("to_merge_key"))
        if key not in existing_rels:
            rel["case_ids"] = [case_id]
            kg["relationships"].append(rel)
            existing_rels.add(key)

    # 记录案例元信息
    kg["cases"].append({
        "case_id": case_id,
        "meta": result.get("meta", {}),
        "incremental_update": result.get("incremental_update", {}),
    })


# ── 主流程 ────────────────────────────────────────────
def main():
    print("=== 交通事故知识图谱提取系统 ===\n")

    # 读取理论库
    print("读取理论库...")
    theory_paragraphs = read_docx(THEORY_DOCX)
    theories_str = parse_theories(theory_paragraphs)
    print(f"理论库加载完成，共 {theories_str.count('理论')} 条\n")

    # 读取案例
    print("读取 OCR 案例...")
    cases = parse_cases_from_ocr(OCR_DIR)
    print(f"共读取 {len(cases)} 个案例\n")

    # 加载已有知识图谱
    kg = load_kg()
    print(f"已有知识图谱：{sum(len(v) for v in kg['nodes'].values())} 个节点，{len(kg['relationships'])} 条关系\n")

    # 已处理的案例
    processed_ids = {c["case_id"] for c in kg.get("cases", [])}

    for i, case in enumerate(cases):
        case_id = case["case_id"]

        if case_id in processed_ids:
            print(f"[{i+1}/{len(cases)}] {case_id} 已处理，跳过")
            continue

        print(f"[{i+1}/{len(cases)}] 处理 {case_id} ({case['accident_type']}案例 - {case['driver_name']})")

        snapshot = get_existing_snapshot(kg)
        prompt = build_prompt(case, snapshot, theories_str)

        result = call_deepseek(prompt, case_id)

        if result:
            # 保存单案例结果
            case_file = OUTPUT_DIR / f"{case_id}.json"
            case_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [OK] 已保存 {case_file.name}")

            merge_nodes(kg, result, case_id)
            save_kg(kg)
            print(f"  [OK] 知识图谱已更新：{sum(len(v) for v in kg['nodes'].values())} 节点，{len(kg['relationships'])} 关系")
        else:
            print(f"  [FAIL] {case_id} 提取失败，跳过")

        # 避免频率限制
        if i < len(cases) - 1:
            time.sleep(1)

    print(f"\n=== 完成 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"知识图谱: {KG_FILE}")
    print(f"总节点数: {sum(len(v) for v in kg['nodes'].values())}")
    print(f"总关系数: {len(kg['relationships'])}")
    print(f"已处理案例: {len(kg['cases'])}")


if __name__ == "__main__":
    main()
