def experiment_plan(round_num: int = 1, **overrides):
    if round_num == 1:
        plan = {
            "kind": "baseline",
            "strategy": "establish measured baseline",
            "code_changes": "implement the baseline candidate",
            "hypothesis": "the candidate establishes correctness and baseline performance",
            "expected_effect": {
                "metric": "baseline",
                "direction": "establish_baseline",
                "mechanism": "measure the complete workload set",
            },
            "source": {"origin": "baseline"},
            "knowledge_uses": [],
        }
    else:
        plan = {
            "kind": "performance",
            "strategy": f"strategy {round_num}",
            "code_changes": f"changes {round_num}",
            "hypothesis": f"hypothesis {round_num}",
            "expected_effect": {
                "metric": "geo_mean",
                "direction": "increase",
                "mechanism": "reduce redundant work",
            },
            "source": {"origin": "coder"},
            "knowledge_uses": [],
        }
    plan.update(overrides)
    return plan


def round_conclusion(round_num: int = 1, **overrides):
    conclusion = {
        "round_num": round_num,
        "expectation_status": "baseline" if round_num == 1 else "met",
        "root_cause": f"cause {round_num}",
        "perf_gap_analysis": "" if round_num == 1 else f"observed result matched plan {round_num}",
        "next_suggestion": f"next {round_num}",
        "optimization_level": "L2_memory",
        "knowledge_assessments": [],
    }
    conclusion.update(overrides)
    return conclusion
