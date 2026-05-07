# Assignment 5: KG Multi-Agent QA System (A4 Extension)

Due Date: 4/24 - 5/7  
TA: 葛亭妤

---

## 0) Objective

This assignment extends Assignment 4. Your goals are:
1. Build a multi-agent QA system on top of your **A4 KG**
2. Add security validation (reject unsafe requests)
3. Add diagnosis and repair flow (not only direct answering)
4. Produce outputs that can be graded with a fixed TA pipeline

---

## Environment Setup (Before You Start)

### Prerequisites
- Python 3.11
- Docker Desktop (for Neo4j)
- Sufficient disk space for local model cache 

### Neo4j startup (example)
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### Python environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

### Recommended run order
```bash
python setup_data.py
python build_kg.py
python auto_test_a5.py
```

All commands should be run in the `assignment5/` folder.

---

## 1) Files Provided in This Assignment (and Purpose)

Starter pack files:

1. `assignment5.md`
   - Full assignment specification and grading rules

2. `query_system_multiagent_template.py`
   - Main pipeline template with fixed output contract
   - You must copy this into your own `query_system_multiagent.py`

3. `agents/a5_template.py`
   - Role boundary template for 7 agent responsibilities with TODO stubs

4. `auto_test_a5.py`
   - Fixed evaluator script used by TA

5. `test_data_a5.json`
   - Fixed benchmark dataset format (`normal` / `failure` / `unsafe`)

6. `setup_data.py`, `build_kg.py`, `source/`
   - A4 carry-over assets for KG construction
   - These are included to preserve A4→A5 continuity
   - You must replace/complete them with your own A4-finished version before final submission

---

## 2) What You May Modify vs. What You Should Not

### You may modify
- `query_system_multiagent_template.py`
- `agents/a5_template.py`
- Any new files you create (e.g., additional `agents/*.py`)
- `setup_data.py`, `build_kg.py` (to align with your own A4 implementation)

### You must create
- `query_system_multiagent.py`
  - Create by copying `query_system_multiagent_template.py`, then implement your logic

### You should not modify (test contract files)
- `auto_test_a5.py`

> Note: You may do local experiments, but final submission must remain compatible with TA's fixed test contract.

---

## 3) A4 → A5 Continuity Rules (Mandatory)

1. You must use your own A4 KG result (schema and build logic)
2. Runtime QA must be read-only on KG

---

## 4) Development Workflow (What You Need to Do)

1. Ensure your A4 pipeline builds a queryable KG
2. Copy file:
   - `query_system_multiagent_template.py` → `query_system_multiagent.py`
3. Ensure `setup_data.py` / `build_kg.py` are your own A4-complete versions
4. Implement your multi-agent flow in `agents/a5_template.py` or your own modules
5. Run `auto_test_a5.py` iteratively
6. Finalize submission package

---

## 5) Agent Capability Requirements

You may design modules freely, but your system capabilities must cover all 7 roles:

1. NL Understanding
2. Security / Policy
3. Query Planning
4. Query Execution
5. Diagnosis
6. Query Repair
7. Explanation

### Recommended flow style (Hybrid)
- Fixed front half: Understand → Security → Plan → Execute → Diagnose
- Dynamic back half: branch based on diagnosis result
- Maximum 1 repair round

---

## 6) Output Contract (Mandatory)

`query_system_multiagent.py` must expose at least one callable:
- `run_multiagent_qa(question)`
- `run_qa(question)`
- `answer_question(question)`

Return value must be a dict containing:
- `answer` (str)
- `safety_decision` (`ALLOW` / `REJECT`)
- `diagnosis` (`SUCCESS` / `QUERY_ERROR` / `SCHEMA_MISMATCH` / `NO_DATA`)
- `repair_attempted` (bool)
- `repair_changed` (bool)
- `explanation` (str)

---

## 7) Test Contract (Mandatory)

`auto_test_a5.py` must be able to:
1. Read `test_data_a5.json` in the specified format
2. Evaluate all three case types (`normal` / `failure` / `unsafe`)
3. Output fixed metrics and weighted 60-point system score
4. Generate `auto_test_a5_results.json` with:
   - per-case pass/fail
   - per-case model/system output
   - per-case contract coverage checks (e.g., missing `safety_decision`, `diagnosis`, `repair_attempted`, `repair_changed`, `explanation`)

### Hard policy
If TA cannot run your evaluator directly, or your output contract is incompatible, the corresponding grading component receives no credit.

---

## 8) Grading (Total 100)

### A. Report / Documentation: 40%
Your report must explain:
- How each agent is designed and implemented
- Why major design decisions were made
- What difficulties you encountered and how you addressed them
- Key findings/insights from debugging and evaluation

### B. System Performance: 60%
- Task Success Rate: 25%
- Security & Validation: 15%
- Error Detection Quality: 8%
- Query Regeneration: 6%
- Correct Resolution After Repair: 6%

`auto_test_a5.py` computes the 60-point system component.

---

## 9) Final Submission Checklist

Upload your github link with following files:

### Mandatory
1. `README.md`
   - Architecture diagram, agent responsibilities, pipeline, challenges, findings
2. `query_system_multiagent.py`
3. `agents/` (your implementation modules)
4. `auto_test_a5.py`
5. `requirements.txt`
6. `build_kg.py`
   - A4-compatible KG builder used for A4→A5 continuity validation


---

## 10) Practical Start Guide

1. Verify your A4 KG can be queried
2. Create `query_system_multiagent.py` from template
3. First make the full pipeline runnable
4. Then optimize score by using `auto_test_a5_results.json`

---

## Submission Details

Deadline: 2026/5/7  

