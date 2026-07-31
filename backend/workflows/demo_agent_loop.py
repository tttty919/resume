"""离线演示：用假工具箱 + 规则决策器跑通 agent_loop（无需 API Key）

目的：在不连大模型的情况下，验证第2步控制器的循环逻辑，并直观看到
5 份简历在同一个 JD 下各走各的路。

运行：
    cd resume && python -m backend.workflows.demo_agent_loop
"""

import asyncio

from backend.workflows.controller import agent_loop, rule_decide, MIN_CONFIDENCE

# ── 同一个 JD 的岗位要求 ──────────────────────────────────
REQUIREMENTS = [
    {"id": "r1", "name": "产品经理经验", "type": "must", "description": "3年以上产品经理经验", "keywords": ["产品经理", "3年"]},
    {"id": "r2", "name": "学历", "type": "must", "description": "本科及以上", "keywords": ["本科"]},
    {"id": "r3", "name": "AI产品0到1经验", "type": "must", "description": "有AI产品从0到1落地经验", "keywords": ["AI产品", "从0到1"]},
    {"id": "r4", "name": "AI编程工具", "type": "must", "description": "熟悉Cursor、Copilot等AI编程工具", "keywords": ["Cursor", "Copilot"]},
    {"id": "r5", "name": "大模型应用开发", "type": "nice", "description": "熟悉LLM应用开发", "keywords": ["LLM", "大模型"]},
]

S = "satisfied"; N = "not_satisfied"; C = "cannot_judge"


def _item(rid, name, status, conf, evidence=""):
    return {"requirement_id": rid, "requirement_name": name, "status": status,
            "confidence": conf, "evidence": evidence, "needs_human_review": False}


class FakeToolBox:
    """假工具箱：按预设剧本返回结果，替代真实 LLM/ChromaDB 调用。"""

    def __init__(self, script: dict):
        self.script = script  # {rid: {...行为...}}

    async def match_requirements(self, reqs, extra_synonyms=None):
        out = []
        for req in reqs:
            rid = req["id"]
            beh = self.script.get(rid, {})
            # 若带了同义词且该条定义了 rematch 结果 → 用重匹配结果
            if extra_synonyms and extra_synonyms.get(rid) and "rematch" in beh:
                st, cf, ev = beh["rematch"]
            else:
                st, cf, ev = beh.get("match", (C, 0.0, ""))
            out.append(_item(rid, req["name"], st, cf, ev))
        return out

    async def query_synonyms(self, term):
        for rid, beh in self.script.items():
            if beh.get("_name") and beh["_name"] in term:
                return beh.get("synonyms", [])
        return []

    async def deep_extract(self, req):
        return self.script.get(req["id"], {}).get("deep", {"found": False, "evidence": ""})

    async def verify_evidence(self, req, item):
        return self.script.get(req["id"], {}).get("verify", {"reliable": True, "confidence": item.get("confidence", 0.9)})

    async def ask_hr_question(self, req, item):
        return {"question": f"请补充候选人在「{req['name']}」方面的具体情况。"}

    async def analyze_risk(self, reqs, items):
        return {"summary": "（演示用占位风险分析）"}

    async def generate_recommendation(self, reqs, items, analysis):
        sat = sum(1 for i in items if i.get("status") == S)
        return {"scoring": {"counts": {"satisfied": sat, "total": len(items)}}}


# ── 5 份简历的剧本 ────────────────────────────────────────
CANDIDATES = {
    "A 简历很硬": {
        "r1": {"match": (S, 0.9, "8年产品经理")},
        "r2": {"match": (S, 0.95, "硕士")},
        "r3": {"match": (S, 0.9, "主导过AI助手0到1")},
        "r4": {"match": (S, 0.88, "长期使用Cursor")},
        "r5": {"match": (S, 0.85, "做过RAG应用")},
    },
    "B 用词不同(Windsurf)": {
        "r1": {"match": (S, 0.9, "6年产品")},
        "r2": {"match": (S, 0.95, "本科")},
        "r3": {"match": (S, 0.85, "AI产品原型3个")},
        # r4 首轮判不了；查同义词命中；重匹配成功
        "r4": {"_name": "AI编程工具", "match": (C, 0.0, ""),
               "synonyms": ["Windsurf", "AI编程工具"],
               "rematch": (S, 0.8, "Windsurf属于AI编程工具")},
        "r5": {"match": (S, 0.8, "LLM应用")},
    },
    "C 证据模糊": {
        "r1": {"match": (S, 0.9, "5年产品")},
        "r2": {"match": (S, 0.95, "本科")},
        # r3 命中但置信度低；校验发现不可靠；深挖仍偏弱 → 待面试
        "r3": {"match": (S, 0.5, "参与过AI相关项目"),
               "verify": {"reliable": False, "confidence": 0.4},
               "deep": {"found": True, "evidence": "一句项目经历", "confidence": 0.45}},
        "r4": {"match": (S, 0.85, "用过Copilot")},
        "r5": {"match": (S, 0.8, "了解大模型")},
    },
    "D 关键项缺失": {
        "r1": {"match": (S, 0.9, "4年产品")},
        "r2": {"match": (S, 0.95, "本科")},
        # r3/r4 完全没有；查同义词无果；深挖无果 → 达上限自动转HR
        "r3": {"_name": "AI产品0到1经验", "match": (C, 0.0, ""), "synonyms": [],
               "deep": {"found": False, "evidence": ""}},
        "r4": {"_name": "AI编程工具", "match": (C, 0.0, ""), "synonyms": [],
               "deep": {"found": False, "evidence": ""}},
        "r5": {"match": (N, 0.8, "无相关内容")},
    },
    "E 硬指标不符(大专)": {
        "r1": {"match": (S, 0.9, "5年产品")},
        "r2": {"match": (N, 0.9, "大专")},   # 确定性不满足，不启动智能环节
        "r3": {"match": (S, 0.85, "AI产品经验")},
        "r4": {"match": (S, 0.85, "用过Cursor")},
        "r5": {"match": (S, 0.8, "LLM")},
    },
}


async def run_one(title: str, script: dict):
    for rid, beh in script.items():
        beh.setdefault("_name", "")
    tb = FakeToolBox(script)
    result = await agent_loop(REQUIREMENTS, {}, "", {}, toolbox=tb, decide=rule_decide)

    print(f"\n{'='*70}\n候选人：{title}")
    print(f"  循环轮数: {result['iterations_used']}  停止原因: {result['stop_reason']}")
    print("  控制器动作链路:")
    for t in result["trace"]:
        rid = t["requirement_id"]
        print(f"    [{t['iteration']}] {t['tool']:<18} {rid:<4} {t['reason']}  → {t['detail']}")
    print("  最终每条结论:")
    for m in result["matches"]:
        flag = "  ⚠待HR" if m["requirement_id"] in result["pending_hr"] else ""
        print(f"    - {m['requirement_name']:<14} {m['status']:<13} conf={m.get('confidence')}{flag}")
    if result["hr_questions"]:
        print("  生成的HR复核问题:")
        for rid, q in result["hr_questions"].items():
            print(f"    - {rid}: {q}")


async def main():
    print(f"（置信度门槛 MIN_CONFIDENCE = {MIN_CONFIDENCE}）")
    for title, script in CANDIDATES.items():
        await run_one(title, script)


if __name__ == "__main__":
    asyncio.run(main())
