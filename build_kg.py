"""
Build Knowledge Graph in Neo4j from SQLite database created by setup_data.py.

Schema:
  (:Regulation {reg_id, name, category})
  (:Rule {art_id, article_number, content, source, reg_id})
  (:Regulation)-[:HAS_RULE]->(:Rule)
"""
from __future__ import annotations

import os
import sqlite3

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

DB_PATH = "ncu_regulations.db"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def load_sqlite() -> tuple[list[dict], list[dict]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT reg_id, name, category FROM regulations")
    regulations = [dict(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT a.art_id, a.reg_id, a.article_number, a.content, r.name AS source
        FROM articles a
        JOIN regulations r ON a.reg_id = r.reg_id
    """)
    articles = [dict(r) for r in cur.fetchall()]

    conn.close()
    return regulations, articles


def build_kg(regulations: list[dict], articles: list[dict]) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("  Clearing existing graph...")
        session.run("MATCH (n) DETACH DELETE n")

        print("  Creating constraint on Rule.art_id...")
        session.run(
            "CREATE CONSTRAINT rule_art_id IF NOT EXISTS "
            "FOR (r:Rule) REQUIRE r.art_id IS UNIQUE"
        )

        print(f"  Creating {len(regulations)} Regulation nodes...")
        session.run(
            """
            UNWIND $regs AS r
            CREATE (:Regulation {reg_id: r.reg_id, name: r.name, category: r.category})
            """,
            regs=regulations,
        )

        print(f"  Creating {len(articles)} Rule nodes...")
        session.run(
            """
            UNWIND $rules AS r
            CREATE (:Rule {
                art_id: r.art_id,
                article_number: r.article_number,
                content: r.content,
                source: r.source,
                reg_id: r.reg_id
            })
            """,
            rules=articles,
        )

        print("  Creating HAS_RULE relationships...")
        session.run(
            """
            MATCH (reg:Regulation), (rule:Rule)
            WHERE reg.reg_id = rule.reg_id
            CREATE (reg)-[:HAS_RULE]->(rule)
            """
        )

        print("  Creating fulltext index on Rule.content...")
        session.run(
            "CREATE FULLTEXT INDEX rule_content_idx IF NOT EXISTS "
            "FOR (r:Rule) ON EACH [r.content]"
        )

        count = session.run("MATCH (r:Rule) RETURN count(r) AS c").single()["c"]
        print(f"  Done. Total Rule nodes: {count}")

    driver.close()


def main() -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"{DB_PATH} not found. Run setup_data.py first."
        )

    print("Loading SQLite data...")
    regulations, articles = load_sqlite()
    print(f"  Found {len(regulations)} regulations, {len(articles)} articles.")

    print("Building Neo4j KG...")
    build_kg(regulations, articles)
    print("KG build complete.")


if __name__ == "__main__":
    main()
