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


def eval_or_matching():
    """Step 3: OR 条件匹配专项测试"""
    print("\n" + "=" * 60)
    print("STEP 3: OR 条件匹配专项测试")
    print("=" * 60)

    # 找一个明确命中 Python 的候选人
    resume_file = GOLDEN_DIR / "resumes" / "简历_张明_AI产品经理.txt"
    resume_text = resume_file.read_text(encoding="utf-8")

    # 构造 OR 条件：Python或Ruby或Haskell
    test_reqs = [{
        "id": "req-or-001",
        "name": "编程语言（OR测试）",
        "description": "熟悉Python或Ruby或Haskell等至少一门编程语言（满足其一即可）",
        "type": "must",
        "importance": "high",
        "keywords": ["Python", "Ruby", "Haskell"]
    }]

    # 先提取简历结构化信息
    print("\n  1. 提取简历...")
    try:
        extract_resp = call_api("POST", "/api/extract-resume-text",
            json={"resume": resume_text, "api_key": API_KEY})
    except Exception as e:
        print(f"  ✗ 提取失败: {e}")
        return False

    if not extract_resp.get("success"):
        print(f"  ✗ 提取失败: {extract_resp.get('message')}")
        return False

    resume_data = extract_resp.get("data", {})
    print(f"  候选人: {resume_data.get('basic_info', {}).get('name', '?')}")
    skills = resume_data.get("skills", [])
    python_in_skills = any("python" in s.lower() for s in skills)
    print(f"  技能中{'有' if python_in_skills else '没有'} Python: {[s for s in skills if 'python' in s.lower()]}")

    # 调用匹配 API
    print("\n  2. 匹配 OR 条件...")
    try:
        match_resp = call_api("POST", "/api/match",
            json={
                "requirements": test_reqs,
                "resume": resume_data,
                "raw_resume_text": resume_text,
                "api_key": API_KEY
            })
    except Exception as e:
        print(f"  ✗ 匹配失败: {e}")
        return False

    if not match_resp.get("success"):
        print(f"  ✗ 匹配失败: {match_resp.get('message')}")
        return False

    matches = match_resp.get("data", {}).get("matches", [])
    if not matches:
        print("  ✗ 无匹配结果")
        return False

    m = matches[0]
    status = m.get("status", "?")
    confidence = m.get("confidence", 0)
    reasoning = m.get("reasoning", "")

    icon = "✓" if status == "satisfied" else "✗"
    print(f"\n  结果: {icon} status={status} confidence={confidence:.0%}")
    print(f"  reasoning: {reasoning}")

    # 判断：张明简历里 Python 精通，应该 satisfied
    if status == "satisfied":
        print(f"\n  ✅ OR 匹配正确！候选人命中 Python → satisfied")
        return True
    else:
        print(f"\n  ❌ OR 匹配错误！候选人精通 Python 但被判 {status}")
        print(f"     预期：OR 条件命中一个即 satisfied")
        return False


def eval_skill3_full():
    """Step 4: Skill 3 全量评测 — 5 份简历逐项匹配 + 对比人工标注"""
    print("\n" + "=" * 60)
    print("STEP 4: Skill 3 全量匹配评测（5 份简历）")
    print("=" * 60)

    # 1. JD 解析
    print("\n  [1/3] 解析 JD...")
    try:
        jd_resp = call_api("POST", "/api/parse-jd", json={"jd": TEST_JD, "api_key": API_KEY})
        if not jd_resp.get("success"):
            print(f"    ✗ JD 解析失败: {jd_resp.get('message')}")
            return None
        requirements = jd_resp.get("data", {}).get("requirements", [])
    except Exception as e:
        print(f"    ✗ JD 解析 API 调用失败: {e}")
        return None

    print(f"    JD 解析完成: {len(requirements)} 条要求")
    for r in requirements:
        has_or = "或" in r.get("description", "") or "或" in r.get("name", "")
        tag = " [OR]" if has_or else ""
        print(f"      {r['id']}: [{r['type']}/{r['importance']}] {r['name']}{tag}")

    # 2. 加载人工标注期望值
    label_dir = GOLDEN_DIR / "labels"
    expected_labels = {}
    for lf in sorted(label_dir.iterdir()):
        if lf.suffix == ".json":
            name = lf.stem.replace("_expected", "")
            expected_labels[name] = json.loads(lf.read_text(encoding="utf-8"))

    # 3. 逐份简历 → 提取 + 匹配
    print(f"\n  [2/3] 匹配 {len(expected_labels)} 份简历...")
    resume_dir = GOLDEN_DIR / "resumes"
    all_results = []

    for resume_file in sorted(resume_dir.iterdir()):
        if resume_file.suffix != ".txt":
            continue
        name_key = resume_file.stem.replace("简历_", "").replace("_AI产品经理", "").replace("_AI产品", "")
        expected = expected_labels.get(name_key)
        if not expected:
            print(f"    ⚠ {name_key}: 无标注，跳过")
            continue

        resume_text = resume_file.read_text(encoding="utf-8")
        print(f"\n    --- {name_key} (期望: {expected.get('expected_job_match_tier', '?')}) ---")

        # 提取
        try:
            ext_resp = call_api("POST", "/api/extract-resume-text",
                json={"resume": resume_text, "api_key": API_KEY})
        except Exception as e:
            print(f"      ✗ 提取失败: {e}")
            continue

        if not ext_resp.get("success"):
            print(f"      ✗ 提取失败: {ext_resp.get('message')}")
            continue

        resume_data = ext_resp.get("data", {})
        candidate_name = resume_data.get("basic_info", {}).get("name", "?")
        print(f"      姓名: {candidate_name}")

        # 匹配
        try:
            match_resp = call_api("POST", "/api/match",
                json={
                    "requirements": requirements,
                    "resume": resume_data,
                    "raw_resume_text": resume_text,
                    "api_key": API_KEY
                })
        except Exception as e:
            print(f"      ✗ 匹配失败: {e}")
            continue

        if not match_resp.get("success"):
            print(f"      ✗ 匹配失败: {match_resp.get('message')}")
            continue

        matches = match_resp.get("data", {}).get("matches", [])
        counts = {"satisfied": 0, "not_satisfied": 0, "cannot_judge": 0}
        or_checks = []  # OR 条件的匹配情况
        bad_cases = []  # 潜在 Bad Case

        for m in matches:
            status = m.get("status", "cannot_judge")
            counts[status] = counts.get(status, 0) + 1

            req_id = m.get("requirement_id", "")
            rid = m.get("requirement_name", "")
            desc = ""
            req_type = "bonus"
            for r in requirements:
                if r.get("id") == req_id:
                    desc = r.get("description", "")
                    req_type = r.get("type", "bonus")
                    break

            has_or = "或" in desc or "或" in rid
            confidence = m.get("confidence", 0)
            reasoning = m.get("reasoning", "")
            status_icon = {"satisfied": "v", "not_satisfied": "x", "cannot_judge": "?"}.get(status, "?")

            if has_or:
                or_checks.append({
                    "req_id": req_id,
                    "name": rid,
                    "description": desc,
                    "status": status,
                    "confidence": confidence,
                    "reasoning": reasoning[:120]
                })
                print(f"      {status_icon} {rid} [{status}] OR条件")

            # Bad Case 收集
            is_must = req_type == "must"
            if is_must and status == "not_satisfied":
                bad_cases.append({"kind": "must_ns", "req_name": rid, "status": status,
                    "confidence": confidence, "reasoning": reasoning, "description": desc})
            elif is_must and status == "cannot_judge" and confidence < 0.6:
                bad_cases.append({"kind": "must_cj", "req_name": rid, "status": status,
                    "confidence": confidence, "reasoning": reasoning, "description": desc})
            elif has_or and status == "not_satisfied":
                bad_cases.append({"kind": "or_strict", "req_name": rid, "status": status,
                    "confidence": confidence, "reasoning": reasoning, "description": desc})

        # 汇总
        must_total = len([r for r in requirements if r.get("type") == "must"])
        must_satisfied = sum(1 for m in matches if m.get("status") == "satisfied"
                             and any(r.get("id") == m.get("requirement_id") and r.get("type") == "must"
                                     for r in requirements))
        score_pct = must_satisfied / must_total * 100 if must_total else 0

        print(f"      汇总: {counts['satisfied']}满足 {counts['not_satisfied']}不满足 {counts['cannot_judge']}无法判断")
        print(f"      必须项满足率: {must_satisfied}/{must_total} ({score_pct:.0f}%)")

        if or_checks:
            print(f"      OR 条件详情:")
            for oc in or_checks:
                ok = "✅" if oc["status"] == "satisfied" else "❌"
                print(f"        {ok} {oc['name']}: {oc['status']} ({oc['confidence']:.0%}) — {oc['reasoning']}")

        all_results.append({
            "name": name_key,
            "candidate_name": candidate_name,
            "expected_tier": expected.get("expected_job_match_tier", "?"),
            "counts": counts,
            "must_satisfied": must_satisfied,
            "must_total": must_total,
            "score_pct": score_pct,
            "or_checks": or_checks,
            "bad_cases": bad_cases,
        })

    # 4. 排序 & 汇总
    print(f"\n{'='*60}")
    print(f"  [3/3] Skill 3 评测汇总")
    print(f"{'='*60}")
    print(f"  {'候选人':<10} {'期望':<8} {'must满足率':<12} {'满足':<6} {'不满足':<6} {'无法判':<6} {'排名':<6}")
    print(f"  {'-'*54}")

    tier_order = {"high": 0, "medium": 1, "low": 2}
    sorted_results = sorted(all_results, key=lambda x: -x["score_pct"])

    for i, r in enumerate(sorted_results):
        expected = r["expected_tier"]
        # 根据实际分数推断 tier
        if r["score_pct"] >= 70:
            actual_tier = "high"
        elif r["score_pct"] >= 40:
            actual_tier = "medium"
        else:
            actual_tier = "low"
        match_ok = "✓" if actual_tier == expected else "⚠"
        print(f"  {r['candidate_name']:<10} {expected:<8} {r['must_satisfied']}/{r['must_total']} ({r['score_pct']:.0f}%)     {r['counts']['satisfied']:<6} {r['counts']['not_satisfied']:<6} {r['counts']['cannot_judge']:<6} #{i+1:<5} {match_ok}")

    # 检查排序是否合理
    print(f"\n  排序验证:")
    top2 = [r["name"] for r in sorted_results[:2]]
    high_expected = [r["name"] for r in all_results if r["expected_tier"] == "high"]
    overlap = set(top2) & set(high_expected)
    if len(overlap) >= 1:
        print(f"    ✅ Top2 {top2} 至少包含 {len(overlap)} 个期望 high 候选人 ({list(overlap)})")
    else:
        print(f"    ❌ Top2 {top2} 不包含任何期望 high 候选人 ({high_expected})")

    # 5. Bad Case 分析
    print(f"\n{'='*60}")
    print(f"  Bad Case 分析")
    print(f"{'='*60}")

    total_bad = 0
    for r in all_results:
        bad_items = r.get("bad_cases", [])
        if bad_items:
            print(f"\n  ⚡ {r['candidate_name']} ({r['expected_tier']}期望):")
            for bc in bad_items:
                kind = bc["kind"]
                icon = {"must_ns": "🔴", "must_cj": "🟡", "or_strict": "🔶"}.get(kind, "⚪")
                print(f"    {icon} [{bc['req_name']}] status={bc['status']} conf={bc['confidence']:.0%}")
                print(f"       理由: {bc['reasoning'][:150]}")
                print(f"       需求: {bc['description'][:120]}")
                total_bad += 1

    if total_bad == 0:
        print("\n  ✅ 未发现明显 Bad Case")
    else:
        print(f"\n  共 {total_bad} 个潜在 Bad Case，需人工审查")

    return sorted_results


def eval_matching():
    """Step 4 别名（保留向后兼容）"""
    return eval_skill3_full()


if __name__ == "__main__":
    print("F1 招聘筛选 — Prompt 评估脚本")
    print(f"服务地址: {BASE_URL}")
    print(f"API Key: {'已设置' if API_KEY else '未设置（使用服务端默认）'}")

    extraction_results = eval_extraction()
    jd_results = eval_jd_parsing()
    or_test_passed = eval_or_matching()
    skill3_results = eval_skill3_full()

    print(f"\n{'='*60}")
    print("全量评估完成。")
    print(f"  Step 1 (简历提取): 平均 F1 = {sum(r['skill_f1'] for r in extraction_results)/len(extraction_results):.1%}" if extraction_results else "  Step 1: 跳过")
    print(f"  Step 2 (JD解析): {'完成' if jd_results else '跳过'}")
    print(f"  Step 3 (OR专项): {'PASS' if or_test_passed else 'FAIL'}")
    print(f"  Step 4 (Skill3全量): {'完成 ' + str(len(skill3_results)) + ' 人' if skill3_results else '跳过'}")
    print(f"\n提示：每次改 Prompt 后重新跑此脚本，对比数值变化。")
    print(f"{'='*60}")
