import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# CONFIG
# ============================================================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATA_DIR = Path(os.getenv("DATA_DIR", ".")).resolve()

FILES = {
    "Mathematics": "math_test.xlsx",
    "ComputerScience": "ComputerScience_test.xlsx",
    "ScienceEngineering": "ScienceEngineering_test.xlsx",
    "GenEd": "Gen_Ed_test.xlsx",
    "Elective": "Elective_test.xlsx",
    "ACS": "ACS_test.xlsx",
    "AI": "AI_test.xlsx",
    "BDA": "BDA_test.xlsx",
    "CSys": "CSys_test.xlsx",
    "SE": "SE_test.xlsx",
    "Minor": "Minor_test.xlsx",
}

REQUIRED_COLUMNS = [
    "Course Code",
    "Course Title",
    "Course Credits",
    "Prerequisites",
    "Category",
]


# ============================================================
# HELPERS
# ============================================================
def norm(value: object) -> str:
    if value is None:
        return "NONE"
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "null", "none", "n/a"}:
        return "NONE"
    return value


def normalize_code(code: object) -> str:
    return norm(code).upper().replace(" ", "").replace("^", "")


def read_excel_safe(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return []

    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].fillna("NONE")

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Course Code": normalize_code(row["Course Code"]),
                "Course Title": norm(row["Course Title"]),
                "Course Credits": norm(row["Course Credits"]),
                "Prerequisites": norm(row["Prerequisites"]),
                "Category": norm(row["Category"]),
            }
        )
    return rows


def extract_course_codes(text: str) -> list[str]:
    if not text or text == "NONE":
        return []
    cleaned = text.upper().replace(" ", "")
    return sorted(set(re.findall(r"[A-Z]{2,4}\d{4}", cleaned)))


# ============================================================
# GRAPH IMPORTER
# ============================================================
class GraphImporter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params) -> None:
        with self.driver.session() as session:
            session.run(query, **params)

    # -------------------------
    # Setup
    # -------------------------
    def clear_graph(self) -> None:
        self.run("MATCH (n) DETACH DELETE n")
        print("[OK] Graph cleared")

    def create_constraints(self) -> None:
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (n:Program) REQUIRE n.name IS UNIQUE
        """)
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (n:Course) REQUIRE n.code IS UNIQUE
        """)
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (n:Specialization) REQUIRE n.name IS UNIQUE
        """)
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (n:CSGroup) REQUIRE (n.name) IS UNIQUE
        """)
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (n:GenEdGroup) REQUIRE (n.name) IS UNIQUE
        """)
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (n:MinorType) REQUIRE (n.name) IS UNIQUE
        """)
        print("[OK] Constraints created")

    def create_base_structure(self) -> None:
        self.run("MERGE (:Program {name:'BSCS'})")

        self.run("""
            MATCH (p:Program {name:'BSCS'})
            MERGE (g:GenEd {name:'GenEd'})
            MERGE (p)-[:HAS_CATEGORY]->(g)
        """)

        self.run("""
            MATCH (p:Program {name:'BSCS'})
            MERGE (c:ComputerScience {name:'ComputerScience'})
            MERGE (p)-[:HAS_CATEGORY]->(c)
        """)

        self.run("""
            MATCH (p:Program {name:'BSCS'})
            MERGE (m:Mathematics {name:'Mathematics'})
            MERGE (p)-[:HAS_CATEGORY]->(m)
        """)

        self.run("""
            MATCH (p:Program {name:'BSCS'})
            MERGE (s:ScienceEngineering {name:'ScienceEngineering'})
            MERGE (p)-[:HAS_CATEGORY]->(s)
        """)

        self.run("""
            MATCH (p:Program {name:'BSCS'})
            MERGE (e:Electives {name:'Electives'})
            MERGE (p)-[:HAS_CATEGORY]->(e)
        """)

        self.run("""
            MATCH (p:Program {name:'BSCS'})
            MERGE (m:Minor {name:'Minor'})
            MERGE (p)-[:HAS_CATEGORY]->(m)
        """)

        print("[OK] Base structure created")

    def create_cs_structure(self) -> None:
        for group in ["Required", "Specializations", "ComputingElective"]:
            self.run("""
                MATCH (c:ComputerScience {name:'ComputerScience'})
                MERGE (g:CSGroup {name:$group})
                MERGE (c)-[:HAS_GROUP]->(g)
            """, group=group)

        for spec in ["ACS", "AI", "BDA", "CSys", "SE"]:
            self.run("""
                MATCH (g:CSGroup {name:'Specializations'})
                MERGE (s:Specialization {name:$spec})
                MERGE (g)-[:HAS_SPECIALIZATION]->(s)
            """, spec=spec)

            for subgroup in ["Required", "Optional"]:
                self.run("""
                    MATCH (s:Specialization {name:$spec})
                    MERGE (sg:SpecGroup {name:$subgroup, specialization:$spec})
                    MERGE (s)-[:HAS_GROUP]->(sg)
                """, spec=spec, subgroup=subgroup)

        print("[OK] CS structure created")

    def create_gened_structure(self) -> None:
        groups = [
            "DirectGenEd",
            "Arabic",
            "French",
            "Humanities",
            "Art",
            "HistoryPoliticalScience",
            "SocialSciences",
        ]

        for group in groups:
            self.run("""
                MATCH (c:GenEd {name:'GenEd'})
                MERGE (g:GenEdGroup {name:$group})
                MERGE (c)-[:HAS_GROUP]->(g)
            """, group=group)

        print("[OK] GenEd structure created")

    def create_minor_structure(self) -> None:
        minors = [
            "Business Administration",
            "General Engineering",
            "Communication Studies",
        ]

        for minor in minors:
            self.run("""
                MATCH (c:Minor {name:'Minor'})
                MERGE (m:MinorType {name:$minor})
                MERGE (c)-[:HAS_MINOR_TYPE]->(m)
            """, minor=minor)

        print("[OK] Minor structure created")

    # -------------------------
    # Generic course merge
    # -------------------------
    def merge_course(self, row: dict, source: str) -> None:
        self.run("""
            MERGE (c:Course {code:$code})
            SET c.title = $title,
                c.credits = $credits,
                c.prerequisites = $prereq,
                c.category = $category,
                c.source_file = $source
        """,
        code=row["Course Code"],
        title=row["Course Title"],
        credits=row["Course Credits"],
        prereq=row["Prerequisites"],
        category=row["Category"],
        source=source)

    # -------------------------
    # Imports by category
    # -------------------------
    def import_math(self, rows: list[dict]) -> None:
        for row in rows:
            self.merge_course(row, "math_test.xlsx")
            self.run("""
                MATCH (cat:Mathematics {name:'Mathematics'})
                MATCH (c:Course {code:$code})
                MERGE (cat)-[:HAS_COURSE]->(c)
            """, code=row["Course Code"])
        print("[OK] Mathematics imported")

    def import_science_engineering(self, rows: list[dict]) -> None:
        for row in rows:
            self.merge_course(row, "ScienceEngineering_test.xlsx")
            self.run("""
                MATCH (cat:ScienceEngineering {name:'ScienceEngineering'})
                MATCH (c:Course {code:$code})
                MERGE (cat)-[:HAS_COURSE]->(c)
            """, code=row["Course Code"])
        print("[OK] ScienceEngineering imported")

    def import_electives(self, rows: list[dict]) -> None:
        for row in rows:
            self.merge_course(row, "Elective_test.xlsx")
            self.run("""
                MATCH (cat:Electives {name:'Electives'})
                MATCH (c:Course {code:$code})
                MERGE (cat)-[:HAS_COURSE]->(c)
            """, code=row["Course Code"])
        print("[OK] Electives imported")

    def import_cs(self, rows: list[dict]) -> None:
        for row in rows:
            self.merge_course(row, "ComputerScience_test.xlsx")

            cat = row["Category"].strip().lower()
            if cat == "required":
                group = "Required"
            elif cat == "computing elective":
                group = "ComputingElective"
            else:
                group = "Specializations"

            self.run("""
                MATCH (g:CSGroup {name:$group})
                MATCH (c:Course {code:$code})
                MERGE (g)-[:HAS_COURSE]->(c)
            """, group=group, code=row["Course Code"])

        print("[OK] ComputerScience imported")

    # -------------------------
    # GenEd
    # -------------------------
    def detect_gened_group(self, row: dict) -> str:
        text = f"{row['Course Code']} {row['Course Title']} {row['Category']}".lower()
        text = text.replace(" ", "")

        if (
            "social" in text
            or "eco1300" in text
            or "geo1301" in text
            or "psy1301" in text
            or "soc1301" in text
            or "ssc1310" in text
        ):
            return "SocialSciences"
        if "ara" in text or "arabic" in text or "arb" in text:
            return "Arabic"
        if "frn" in text or "french" in text:
            return "French"
        if any(x in text for x in ["hum", "lit", "phi", "humanities"]):
            return "Humanities"
        if any(x in text for x in ["art", "eng2320", "hum2301", "com2327", "lit3370"]):
            return "Art"
        if any(x in text for x in ["his", "psc", "history", "political"]):
            return "HistoryPoliticalScience"
        return "DirectGenEd"

    def import_gened(self, rows: list[dict]) -> None:
        for row in rows:
            self.merge_course(row, "Gen_Ed_test.xlsx")
            group = self.detect_gened_group(row)

            self.run("""
                MATCH (g:GenEdGroup {name:$group})
                MATCH (c:Course {code:$code})
                MERGE (g)-[:HAS_COURSE]->(c)
            """, group=group, code=row["Course Code"])

        print("[OK] GenEd imported")

    # -------------------------
    # Specializations
    # -------------------------
    def import_specialization(self, spec: str, rows: list[dict]) -> None:
        for row in rows:
            self.merge_course(row, f"{spec}_test.xlsx")

            subgroup = row["Category"].strip().title()
            if subgroup not in {"Required", "Optional"}:
                subgroup = "Optional"

            self.run("""
                MATCH (sg:SpecGroup {name:$subgroup, specialization:$spec})
                MATCH (c:Course {code:$code})
                MERGE (sg)-[:HAS_COURSE]->(c)
            """, subgroup=subgroup, spec=spec, code=row["Course Code"])

        print(f"[OK] {spec} imported")

    # -------------------------
    # Minor
    # -------------------------
    def import_minor(self, rows: list[dict]) -> None:
        for row in rows:
            self.merge_course(row, "Minor_test.xlsx")
            self.run("""
                MATCH (m:MinorType {name:$minor})
                MATCH (c:Course {code:$code})
                MERGE (m)-[:HAS_COURSE]->(c)
            """, minor=row["Category"], code=row["Course Code"])

        print("[OK] Minor imported")

    # -------------------------
    # Prerequisite edges
    # -------------------------
    def create_prerequisite_edges(self, all_rows: list[dict]) -> None:
        count = 0

        for row in all_rows:
            target_code = row["Course Code"]
            prereq_codes = extract_course_codes(row["Prerequisites"])

            for prereq_code in prereq_codes:
                if prereq_code == target_code:
                    continue

                self.run("""
                    MATCH (target:Course {code:$target_code})
                    MATCH (source:Course {code:$prereq_code})
                    MERGE (target)-[:REQUIRES]->(source)
                """, target_code=target_code, prereq_code=prereq_code)
                count += 1

        print(f"[OK] Prerequisite edges created: {count}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not NEO4J_PASSWORD:
        raise ValueError("Missing NEO4J_PASSWORD in .env")

    print(f"[INFO] DATA_DIR: {DATA_DIR}")

    data = {}
    all_rows = []

    for key, file_name in FILES.items():
        rows = read_excel_safe(DATA_DIR / file_name)
        data[key] = rows
        all_rows.extend(rows)
        print(f"[INFO] Loaded {key}: {len(rows)}")

    importer = GraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        importer.clear_graph()
        importer.create_constraints()
        importer.create_base_structure()
        importer.create_cs_structure()
        importer.create_gened_structure()
        importer.create_minor_structure()

        importer.import_math(data["Mathematics"])
        importer.import_science_engineering(data["ScienceEngineering"])
        importer.import_electives(data["Elective"])
        importer.import_cs(data["ComputerScience"])
        importer.import_gened(data["GenEd"])

        for spec in ["ACS", "AI", "BDA", "CSys", "SE"]:
            importer.import_specialization(spec, data[spec])

        importer.import_minor(data["Minor"])
        importer.create_prerequisite_edges(all_rows)

        print("\n[DONE] Part 2 completed successfully")

    finally:
        importer.close()


if __name__ == "__main__":
    main()