from __future__ import annotations

from typing import Any

from agents.a5_template import build_template_pipeline

PIPELINE = build_template_pipeline()


def answer_question(question: str) -> dict[str, Any]:
    """
    Multi-agent QA entry point.

    Output contract:
    {
      "answer": str,
      "safety_decision": "ALLOW" | "REJECT",
      "diagnosis": "SUCCESS" | "QUERY_ERROR" | "SCHEMA_MISMATCH" | "NO_DATA",
      "repair_attempted": bool,
      "repair_changed": bool,
      "explanation": str,
    }
    """
    nlu = PIPELINE["nlu"]
    security_agent = PIPELINE["security"]
    planner = PIPELINE["planner"]
    executor = PIPELINE["executor"]
    diagnosis_agent = PIPELINE["diagnosis"]
    repair_agent = PIPELINE["repair"]
    answer_extraction = PIPELINE["answer_extraction"]
    explanation_agent = PIPELINE["explanation"]

    # Stage 1: NL Understanding
    intent = nlu.run(question)

    # Stage 2: Security check
    security = security_agent.run(question, intent)
    if security["decision"] == "REJECT":
        diagnosis = {"label": "QUERY_ERROR", "reason": "Blocked by security policy."}
        explanation = explanation_agent.run(question, intent, security, diagnosis, [], False)
        return {
            "answer": "Request rejected by security policy.",
            "safety_decision": "REJECT",
            "diagnosis": diagnosis["label"],
            "repair_attempted": False,
            "repair_changed": False,
            "explanation": explanation,
        }

    # Stage 3: Query planning
    plan = planner.run(intent)

    # Stage 4: Query execution
    execution = executor.run(plan)

    # Stage 5: Diagnosis
    diagnosis = diagnosis_agent.run(execution)

    # Stage 6: Conditional repair (max 1 round)
    repair_attempted = False
    repair_changed = False
    if diagnosis["label"] in {"QUERY_ERROR", "SCHEMA_MISMATCH", "NO_DATA"}:
        repair_attempted = True
        repaired_plan = repair_agent.run(diagnosis, plan, intent)
        repair_changed = repaired_plan != plan
        execution = executor.run(repaired_plan)
        diagnosis = diagnosis_agent.run(execution)

    # Stage 7: Answer generation
    rows = execution.get("rows", [])
    if diagnosis["label"] == "SUCCESS":
        answer = answer_extraction.run(question, rows, intent.aspect)
    elif diagnosis["label"] == "NO_DATA":
        answer = "No matching regulation evidence found in KG."
    else:
        answer = "Query could not be resolved after repair attempt."

    # Stage 8: Explanation
    explanation = explanation_agent.run(question, intent, security, diagnosis, rows, repair_attempted)

    return {
        "answer": answer,
        "safety_decision": "ALLOW",
        "diagnosis": diagnosis["label"],
        "repair_attempted": repair_attempted,
        "repair_changed": repair_changed,
        "explanation": explanation,
    }


def run_multiagent_qa(question: str) -> dict[str, Any]:
    return answer_question(question)


def run_qa(question: str) -> dict[str, Any]:
    return answer_question(question)


if __name__ == "__main__":
    while True:
        q = input("Question (type exit to quit): ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        result = answer_question(q)
        print(f"Answer      : {result['answer']}")
        print(f"Safety      : {result['safety_decision']}")
        print(f"Diagnosis   : {result['diagnosis']}")
        print(f"Repair      : attempted={result['repair_attempted']}, changed={result['repair_changed']}")
        print(f"Explanation : {result['explanation']}")
        print()
