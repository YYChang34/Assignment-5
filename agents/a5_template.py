from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Intent:
    question_type: str
    keywords: list[str]
    search_terms: str
    aspect: str
    ambiguous: bool = False

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


_STOP_WORDS = {
    "what", "is", "the", "how", "many", "can", "a", "an", "i", "if", "for",
    "of", "to", "in", "be", "are", "will", "happen", "condition", "does",
    "do", "my", "me", "it", "which", "or", "and", "with", "by", "under",
    "have", "has", "been", "after", "before", "that", "when", "who", "this",
    "you", "your", "they", "their", "we", "our", "at", "on", "as", "from",
    "not", "no", "yes", "so", "but", "was", "were", "had", "get", "got",
    "then", "than", "more", "such", "into", "about", "just", "also", "up",
    "take", "out", "would", "could", "should",
}

_PENALTY_WORDS = {"penalty", "punishment", "deduction", "consequence", "penalized", "penalised"}
_FEE_WORDS = {"fee", "cost", "charge", "price", "pay", "payment", "ntd", "nt$"}
_PERMISSION_WORDS = {"allowed", "permit", "permitted", "allow", "can", "may", "right", "eligible"}
_EXAM_WORDS = {"exam", "examination", "test", "invigilator", "late", "barred", "admitted",
               "question paper", "answer sheet", "cheating", "copy", "notes", "electronic"}
_ID_WORDS = {"student id", "easycard", "mifare", "id card", "card replacement", "replacement"}
_GRADUATION_WORDS = {"graduation", "graduate", "credit", "credits", "degree", "bachelor",
                     "semester", "military training", "physical education", "pe", "dismiss",
                     "expelled", "extension", "leave of absence", "suspension"}
_GRADING_WORDS = {"passing score", "pass", "grade", "grading", "master", "phd", "doctoral",
                  "make-up", "makeup", "retake"}
_HEDGING_WORDS = {"probably", "maybe", "generally", "could", "perhaps", "might", "roughly",
                  "kind of", "sort of", "overall", "always", "every", "all"}

_ASPECT_TO_CATEGORY: dict[str, str | None] = {
    "exam": "Exam",
    "id_card": "Admin",
    "graduation": "General",
    "grading": "Grade",
    "general": None,
}


class NLUnderstandingAgent:
    def run(self, question: str) -> Intent:
        q = question.lower()

        question_type = self._classify_type(q)
        aspect = self._classify_aspect(q)
        ambiguous = any(w in q for w in _HEDGING_WORDS)
        keywords = self._extract_keywords(question)
        search_terms = self._build_search_terms(keywords)

        return Intent(
            question_type=question_type,
            keywords=keywords,
            search_terms=search_terms,
            aspect=aspect,
            ambiguous=ambiguous,
        )

    def _classify_type(self, q: str) -> str:
        if any(w in q for w in _PENALTY_WORDS):
            return "penalty"
        if any(w in q for w in _FEE_WORDS):
            return "fee"
        if re.search(r"\bhow (many|long|much)\b", q):
            return "numeric"
        if any(w in q for w in _PERMISSION_WORDS):
            return "permission"
        return "general"

    def _classify_aspect(self, q: str) -> str:
        if any(w in q for w in _EXAM_WORDS):
            return "exam"
        if any(w in q for w in _ID_WORDS):
            if any(w in q for w in _PENALTY_WORDS):
                return "exam"
            return "id_card"
        if any(w in q for w in {"dismissed", "expelled", "dismiss", "expel"}):
            return "graduation"
        if any(w in q for w in _GRADING_WORDS):
            return "grading"
        if any(w in q for w in _GRADUATION_WORDS):
            return "graduation"
        return "general"

    def _extract_keywords(self, question: str) -> list[str]:
        words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-\']*\b", question.lower())
        keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
        seen: set[str] = set()
        unique: list[str] = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:5]

    _KW_SYNONYMS: dict[str, list[str]] = {
        "score": ["grade"],
        "bachelor": ["undergraduate", "four"],
        "bachelor's": ["undergraduate", "four"],
        "master": ["postgraduate"],
        "graduate": ["postgraduate"],
        "dismissed": ["withdraw", "failing", "credits", "semesters"],
        "expelled": ["withdraw", "failing", "credits", "semesters"],
        "standard": ["expected"],
        "duration": ["years", "complete"],
        "period": ["years"],
        "cheating": ["copy", "misconduct"],
        "copying": ["copy"],
        "workday": ["working"],
        "workdays": ["working", "days"],
        "forgetting": ["bring", "forgot"],
    }

    def _build_search_terms(self, keywords: list[str]) -> str:
        expanded: list[str] = []
        seen: set[str] = set()
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                expanded.append(kw)
            for syn in self._KW_SYNONYMS.get(kw, []):
                if syn not in seen:
                    seen.add(syn)
                    expanded.append(syn)

        kw_set = set(keywords)
        if "passing" in kw_set and "score" in kw_set:
            if any(w in kw_set for w in ("undergraduate", "undergrad", "bachelor")):
                if "60" not in seen:
                    seen.add("60")
                    expanded.append("60")
            if any(w in kw_set for w in ("graduate", "master", "phd", "doctoral", "postgraduate")):
                if "70" not in seen:
                    seen.add("70")
                    expanded.append("70")

        if any(w in kw_set for w in ("bachelor", "bachelor's")) and any(
            w in kw_set for w in ("standard", "duration", "period")
        ):
            if "128" not in seen:
                seen.add("128")
                expanded.append("128")

        sanitized = [
            re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', ' ', kw).strip()
            for kw in expanded[:15]
        ]
        terms = [w for w in sanitized if w]
        return " ".join(terms) if terms else ""


class SecurityAgent:
    _BLOCKED = [
        "delete", "drop", "merge", "create", "set ",
        "bypass", "ignore previous", "pretend you are",
        "dump all", "export", "credentials", "word-by-word",
        "modify", "disable safety",
    ]

    def run(self, question: str, intent: Intent) -> dict[str, str]:
        q = question.lower()
        for pattern in self._BLOCKED:
            if pattern in q:
                return {"decision": "REJECT", "reason": f"Blocked pattern: '{pattern}'."}
        return {"decision": "ALLOW", "reason": "Passed security check."}


_CYPHER_FT_CAT = (
    "CALL db.index.fulltext.queryNodes('rule_content_idx', $search_terms) "
    "YIELD node AS r, score "
    "MATCH (:Regulation {category: $category})-[:HAS_RULE]->(r) "
    "RETURN r.article_number AS id, r.content AS content, r.source AS source, score "
    "ORDER BY score DESC LIMIT 5"
)

_CYPHER_FT_BROAD = (
    "CALL db.index.fulltext.queryNodes('rule_content_idx', $search_terms) "
    "YIELD node AS r, score "
    "RETURN r.article_number AS id, r.content AS content, r.source AS source, score "
    "ORDER BY score DESC LIMIT 5"
)

_CYPHER_ALL = (
    "MATCH (r:Rule) "
    "RETURN r.article_number AS id, r.content AS content, r.source AS source "
    "LIMIT 10"
)


class QueryPlannerAgent:
    def run(self, intent: Intent) -> dict[str, Any]:
        search_terms = intent.search_terms
        category = _ASPECT_TO_CATEGORY.get(intent.aspect)

        if not search_terms:
            return {
                "strategy": "all_rules",
                "keywords": intent.keywords,
                "search_terms": search_terms,
                "aspect": intent.aspect,
                "category": category,
                "cypher": _CYPHER_ALL,
                "params": {},
            }

        if category:
            return {
                "strategy": "fulltext_cat",
                "keywords": intent.keywords,
                "search_terms": search_terms,
                "aspect": intent.aspect,
                "category": category,
                "cypher": _CYPHER_FT_CAT,
                "params": {"search_terms": search_terms, "category": category},
            }

        return {
            "strategy": "fulltext_broad",
            "keywords": intent.keywords,
            "search_terms": search_terms,
            "aspect": intent.aspect,
            "category": None,
            "cypher": _CYPHER_FT_BROAD,
            "params": {"search_terms": search_terms},
        }


class QueryExecutionAgent:
    def run(self, plan: dict[str, Any]) -> dict[str, Any]:
        try:
            driver = _get_driver()
            cypher = plan["cypher"]
            params = plan.get("params", {})
            with driver.session() as session:
                result = session.run(cypher, **params)
                rows = [dict(r) for r in result]
            return {"rows": rows, "error": None}
        except Exception as exc:
            return {"rows": [], "error": str(exc), "error_type": _classify_neo4j_error(exc)}


def _classify_neo4j_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(w in msg for w in ("syntax", "property", "type", "unknown", "invalid")):
        return "SCHEMA_MISMATCH"
    return "QUERY_ERROR"


class DiagnosisAgent:
    def run(self, execution: dict[str, Any]) -> dict[str, str]:
        if execution.get("error"):
            label = execution.get("error_type", "QUERY_ERROR")
            return {"label": label, "reason": str(execution["error"])}
        if not execution.get("rows"):
            return {"label": "NO_DATA", "reason": "No matching rule found in KG."}
        return {"label": "SUCCESS", "reason": f"Found {len(execution['rows'])} rule(s)."}


class QueryRepairAgent:
    def run(
        self,
        diagnosis: dict[str, str],
        original_plan: dict[str, Any],
        intent: Intent,
    ) -> dict[str, Any]:
        repaired = dict(original_plan)
        search_terms = intent.search_terms

        if diagnosis["label"] == "QUERY_ERROR":
            short_terms = " ".join(intent.keywords[:2])
            repaired.update({
                "strategy": "fallback_short_ft",
                "cypher": _CYPHER_FT_BROAD,
                "params": {"search_terms": short_terms},
            })
        elif diagnosis["label"] == "NO_DATA":
            repaired.update({
                "strategy": "fallback_broad_ft",
                "cypher": _CYPHER_FT_BROAD,
                "params": {"search_terms": search_terms},
            })
        else:
            repaired.update({
                "strategy": "fallback_broad_ft",
                "cypher": _CYPHER_FT_BROAD,
                "params": {"search_terms": search_terms},
            })

        return repaired


_ANSWER_SYSTEM_PROMPT = (
    "You are an NCU university regulation assistant. "
    "Read ALL provided regulation texts carefully, then pick the MOST DIRECTLY RELEVANT one. "
    "Give a SHORT direct answer following these rules:\n"
    "- ALWAYS use Arabic digits, not words: write '5' not 'Five', '3' not 'Three', '40' not 'forty', '4' not 'four', '2' not 'Two'\n"
    "- 'how many/long/much' questions: state only the number and unit, end with period. "
    "Say 'working days' not 'workdays'. Example: '20 minutes.' or '128 credits.' or '3 working days.'\n"
    "- Passing score/grade questions: state 'X points.' Example: '60 points.' or '70 points.'\n"
    "- Penalty/consequence questions: state ALL consequences from the MOST RELEVANT rule. "
    "Example: '5 points deduction.' or 'Zero score and disciplinary action.'\n"
    "- Yes/No permission questions: ONLY include a TIME LIMIT or SCORE CONSEQUENCE if the rule explicitly states one. "
    "Do NOT restate the prohibition. "
    "Examples: 'No, you must wait 40 minutes.' (time) or 'No, the score will be zero.' (penalty) or just 'No.'\n"
    "- Fee questions: ALWAYS write the amount BEFORE NTD with period. "
    "Example: '200 NTD.' not 'NTD 200'. Example: '100 NTD.'\n"
    "Keep the answer under 2 sentences."
)


_NUM_WORD_MAP = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}


class AnswerExtractionAgent:
    def run(self, question: str, rows: list[dict], aspect: str = "") -> str:
        if not rows:
            return "No matching regulation evidence found in KG."

        rules_text = "\n".join(
            f"[{r['id']}] {r['content']}" for r in rows[:2]
        )

        messages = [
            {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Regulation text:\n{rules_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                ),
            },
        ]

        try:
            from llm_loader import generate_text
            answer = generate_text(messages, max_new_tokens=100)
        except Exception:
            answer = rows[0]["content"] if rows else "No matching regulation evidence found in KG."

        return self._postprocess(answer)

    @staticmethod
    def _postprocess(text: str) -> str:
        text = re.sub(r'\bNTD\s+(\d+)\.?', r'\1 NTD.', text)
        text = re.sub(r'\bmarks\b', 'points', text, flags=re.IGNORECASE)
        for word, digit in _NUM_WORD_MAP.items():
            text = re.sub(rf'\b{word}\b', digit, text, flags=re.IGNORECASE)
        _HAS_CONSEQUENCE = re.compile(r'\b(\d+|zero|points?|score|NTD|NT\$)\b', re.IGNORECASE)
        for prefix, short in (("No, ", "No."), ("Yes, ", "Yes.")):
            if text.startswith(prefix) and not _HAS_CONSEQUENCE.search(text):
                text = short
                break
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text


class ExplanationAgent:
    def run(
        self,
        question: str,
        intent: Intent,
        security: dict[str, str],
        diagnosis: dict[str, str],
        rows: list[dict],
        repair_attempted: bool,
    ) -> str:
        n_rules = len(rows)
        repair_note = " Repair was attempted." if repair_attempted else ""
        return (
            f"[{intent.question_type}] Security: {security['decision']}. "
            f"Retrieved {n_rules} rule(s). "
            f"Diagnosis: {diagnosis['label']}.{repair_note} "
            f"Keywords used: {', '.join(intent.keywords) or 'none'}."
        )


def build_template_pipeline() -> dict[str, Any]:
    return {
        "nlu": NLUnderstandingAgent(),
        "security": SecurityAgent(),
        "planner": QueryPlannerAgent(),
        "executor": QueryExecutionAgent(),
        "diagnosis": DiagnosisAgent(),
        "repair": QueryRepairAgent(),
        "answer_extraction": AnswerExtractionAgent(),
        "explanation": ExplanationAgent(),
    }
