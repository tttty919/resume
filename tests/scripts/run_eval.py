"""
评估脚本：每次改 Prompt 后跑一遍，对比 LLM 输出 vs 人工标注标准答案。

用法：
    python tests/scripts/run_eval.py

前提：
    - 服务已启动在 http://127.0.0.1:8766
    - tests/golden/resumes/ 下有测试简历
    - tests/golden/labels/ 下有对应的人工标注标准答案
"""

import json
import os
import sys
import time
from pathlib import Path

# Fix Windows GBK encoding issue
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import requests

BASE_URL = "http://127.0.0.1:8766"
API_KEY = os.environ.get("F1_API_KEY", "")
GOLDEN_DIR = Path(__file__).parent.parent / "golden"

# ── 测试 JD ──
TEST_JD = """## AI产品经理 — 岗位描述

### 岗位职责
1. 负责AI创新产品的需求分析、功能设计与迭代规划，独立撰写PRD
2. 深度理解AI大模型能力边界，能将技术能力转化为产品方案
3. 协调算法、工程、设计等多团队资源，推动产品从概念到上线
4. 建立产品数据指标体系，通过数据分析驱动产品优化决策

### 任职要求
1. 本科及以上学历，2年以上互联网/AI产品经验
2. 熟悉AI/LLM领域，对大模型应用有实际落地经验者优先
3. 具备优秀的数据分析能力，熟练使用SQL、Python等数据分析工具
4. 熟悉Axure/Figma等原型工具，能独立输出高质量PRD和交互原型
5. 具备较强的逻辑思维和系统性思考能力，有自驱力、好奇心
6. 计算机/AI相关专业背景或具备基础编程能力者优先"""


def call_api(method: str, path: str, **kwargs) -> dict:
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, **kwargs, timeout=120)
    return resp.json()


def eval_extraction():
    """Step 1: 评估简历提取质量"""
    print("\n" + "=" * 60)
    print("STEP 1: 简历信息提取评估")
    print("=" * 60)

    resume_dir = GOLDEN_DIR / "resumes"
    label_dir = GOLDEN_DIR / "labels"

    results = []
    for resume_file in sorted(resume_dir.iterdir()):
        if resume_file.suffix == ".pdf":
            continue  # PDF 需要单独处理

        name = resume_file.stem.replace("简历_", "").replace("_AI产品经理", "").replace("_AI产品", "")
        label_file = label_dir / f"{name}_expected.json"

        if not label_file.exists():
            print(f"  ⚠ {name}: 无标注文件，跳过")
            continue

        expected = json.loads(label_file.read_text(encoding="utf-8"))
        resume_text = resume_file.read_text(encoding="utf-8")

        print(f"\n--- {name} ---")

        # 调用简历提取 API
        try:
            resp = call_api("POST", "/api/extract-resume-text",
                json={"resume": resume_text, "api_key": API_KEY})
        except Exception as e:
            print(f"  ✗ API 调用失败: {e}")
            continue

        if not resp.get("success"):
            print(f"  ✗ 提取失败: {resp.get('message')}")
            continue

        parsed = resp.get("data", {})
        actual_basic = parsed.get("basic_info", {})
        actual_skills = parsed.get("skills", [])

        # ── 逐字段对比 ──
        expected_basic = expected.get("basic_info", {})
        basic_errors = []
        for field in ["name", "school", "degree"]:
            exp = expected_basic.get(field, "")
            act = actual_basic.get(field, "")
            if exp and act and exp != act:
                basic_errors.append(f"{field}: 期望'{exp}' → 实际'{act}'")

        # 技能对比
        exp_skills = set(s.lower() for s in expected.get("skills", []))
        act_skills = set(s.lower() for s in actual_skills)
        skill_precision = len(exp_skills & act_skills) / len(act_skills) if act_skills else 0
        skill_recall = len(exp_skills & act_skills) / len(exp_skills) if exp_skills else 0
        skill_f1 = 2 * skill_precision * skill_recall / (skill_precision + skill_recall) if (skill_precision + skill_recall) > 0 else 0

        # 工作经历数量
        exp_work = expected.get("work_experiences_count", 0)
        act_work = len(parsed.get("work_experiences", []))
        work_ok = abs(exp_work - act_work) <= 1  # 允许 ±1

        print(f"  姓名: {actual_basic.get('name', '?')} {'✓' if not any('name' in e for e in basic_errors) else '✗'}")
        print(f"  学校: {actual_basic.get('school', '?')} | 专业: {actual_basic.get('major', '?')} | 学历: {actual_basic.get('degree', '?')}")
        print(f"  技能: Precision={skill_precision:.0%} Recall={skill_recall:.0%} F1={skill_f1:.0%}")
        print(f"  工作经历: 期望{exp_work}段 实际{act_work}段 {'✓' if work_ok else '✗'}")
        if basic_errors:
            for e in basic_errors:
                print(f"  ⚠ {e}")

        results.append({
            "name": name,
            "skill_f1": skill_f1,
            "skill_precision": skill_precision,
            "skill_recall": skill_recall,
            "work_count_ok": work_ok,
            "basic_errors": basic_errors,
            "skills_missing": list(exp_skills - act_skills),
            "skills_extra": list(act_skills - exp_skills),
        })

    # ── 汇总 ──
    if results:
        avg_f1 = sum(r["skill_f1"] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"简历提取汇总: 平均 Skill F1 = {avg_f1:.1%}  ({len(results)} 份简历)")
        for r in results:
            print(f"  {r['name']}: F1={r['skill_f1']:.1%} P={r['skill_precision']:.1%} R={r['skill_recall']:.1%} WorkOK={r['work_count_ok']}")
        print(f"{'='*60}")
    return results


def eval_jd_parsing():
    """Step 2: JD 解析质量（人工审查）"""
    print("\n" + "=" * 60)
    print("STEP 2: JD 解析评估")
    print("=" * 60)

    try:
        resp = call_api("POST", "/api/parse-jd",
            json={"jd": TEST_JD, "api_key": API_KEY})
    except Exception as e:
        print(f"  ✗ API 调用失败: {e}")
        return None

    if not resp.get("success"):
        print(f"  ✗ 解析失败: {resp.get('message')}")
        return None

    reqs = resp.get("data", {}).get("requirements", [])
    notes = resp.get("data", {}).get("parsing_notes", "")

    print(f"  提取到 {len(reqs)} 条要求:")
    for r in reqs:
        print(f"    {r['id']}: [{r['type']}/{r['importance']}] {r['name']}")
        print(f"      {r['description']}")
        print(f"      keywords: {r.get('keywords', [])}")

    print(f"\n  parsing_notes: {notes}")

    # ── 人工审查清单 ──
    print(f"\n  ⚡ 人工审查要点:")
    print(f"    1. 检查是否有 OR 条件被错误拆分（如 'Python或Java' 不应拆成两条）")
    print(f"    2. 职责编号是否完整覆盖（原始JD有4条职责）")
    print(f"    3. type/importance 标注是否合理")
    print(f"    4. keywords 是否来自 JD 原文（非编造）")

    return reqs


def eval_matching():
    """Step 3: 匹配质量（需要先跑完 extraction）"""
    print("\n" + "=" * 60)
    print("STEP 3: 匹配与排序评估")
    print("=" * 60)

    try:
        resp = call_api("POST", "/api/parse-jd",
            json={"jd": TEST_JD, "api_key": API_KEY})
        if not resp.get("success"):
            print(f"  无法获取 JD 要求: {resp.get('message')}")
            return
        requirements = resp.get("data", {}).get("requirements", [])
    except Exception as e:
        print(f"  ✗ JD 解析失败: {e}")
        return

    label_dir = GOLDEN_DIR / "labels"
    expected_tiers = {}
    for label_file in sorted(label_dir.iterdir()):
        label = json.loads(label_file.read_text(encoding="utf-8"))
        expected_tiers[label_file.stem.replace("_expected", "")] = label.get("expected_job_match_tier", "unknown")

    print(f"\n  期望排序（人工标注）:")
    for name, tier in sorted(expected_tiers.items(), key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x[1], 99)):
        print(f"    {name}: {tier}")

    print(f"\n  ⚡ 匹配评估暂需手动验证。运行完整 Agent 流程后对比排序结果。")
    return expected_tiers


if __name__ == "__main__":
    print("F1 招聘筛选 — Prompt 评估脚本")
    print(f"服务地址: {BASE_URL}")
    print(f"API Key: {'已设置' if API_KEY else '未设置（使用服务端默认）'}")

    extraction_results = eval_extraction()
    jd_results = eval_jd_parsing()
    eval_matching()

    print(f"\n{'='*60}")
    print("评估完成。")
    print("提示：每次改 Prompt 后重新跑此脚本，对比数值变化。")
    print(f"{'='*60}")
