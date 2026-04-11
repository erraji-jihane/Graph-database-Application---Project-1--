import os
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
DATA_DIR = Path(os.getenv("DATA_DIR", "../test_data")).resolve()

FILES = {
    "Mathematics": "math_test.xlsx",
    "ComputerScience": "ComputerScience_test.xlsx",
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
def norm(value):
    if value is None:
        return "NONE"
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "null", "none", "n/a"}:
        return "NONE"
    return value

def normalize_code(code):
    return norm(code).upper().replace(" ", "").replace("^", "")

def read_excel_safe(path: Path):
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
        rows.append({
            "Course Code": normalize_code(row["Course Code"]),
            "Course Title": norm(row["Course Title"]),
            "Course Credits": norm(row["Course Credits"]),
            "Prerequisites": norm(row["Prerequisites"]),
            "Category": norm(row["Category"]),
        })
    return rows

# ============================================================
# GRAPH IMPORTER
# ============================================================
class GraphImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run(self, query, **params):
        with self.driver.session() as session:
            session.run(query, **params)

    def clear_graph(self):
        self.run("MATCH (n) DETACH DELETE n")
        print("[OK] Graph cleared")

    def create_constraints(self):
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (n:Program) REQUIRE n.name IS UNIQUE
        """)
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (n:Category) REQUIRE n.name IS UNIQUE
        """)
        self.run("""
            CREATE CONSTRAINT IF NOT EXISTS FOR (n:Course) REQUIRE n.code IS UNIQUE
        """)
        print("[OK] Constraints created")

    def create_base_structure(self):
        self.run("MERGE (:Program {name:'BSCS'})")

        categories = [
            "GenEd",
            "ComputerScience",
            "Mathematics",
            "ScienceEngineering",
            "Electives",
            "Minor",
        ]

        for cat in categories:
            self.run("""
                MATCH (p:Program {name:'BSCS'})
                MERGE (c:Category {name:$cat})
                MERGE (p)-[:HAS_CATEGORY]->(c)
            """, cat=cat)

        print("[OK] Base structure created")

    def create_cs_structure(self):
        for group in ["Required", "Specializations", "ComputingElective"]:
            self.run("""
                MATCH (c:Category {name:'ComputerScience'})
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

    def create_gened_structure(self):
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
                MATCH (c:Category {name:'GenEd'})
                MERGE (g:GenEdGroup {name:$group})
                MERGE (c)-[:HAS_GROUP]->(g)
            """, group=group)

        print("[OK] GenEd structure created")

    def create_minor_structure(self):
        minors = [
            "Business Administration",
            "General Engineering",
            "Communication Studies",
        ]
        for minor in minors:
            self.run("""
                MATCH (c:Category {name:'Minor'})
                MERGE (m:MinorType {name:$minor})
                MERGE (c)-[:HAS_MINOR_TYPE]->(m)
            """, minor=minor)

        print("[OK] Minor structure created")

    def merge_course(self, row, source):
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

    def import_math(self, rows):
        for row in rows:
            self.merge_course(row, "math_test.xlsx")
            self.run("""
                MATCH (cat:Category {name:'Mathematics'})
                MATCH (c:Course {code:$code})
                MERGE (cat)-[:HAS_COURSE]->(c)
            """, code=row["Course Code"])
        print("[OK] Mathematics imported")

    def import_electives(self, rows):
        for row in rows:
            self.merge_course(row, "Elective_test.xlsx")
            self.run("""
                MATCH (cat:Category {name:'Electives'})
                MATCH (c:Course {code:$code})
                MERGE (cat)-[:HAS_COURSE]->(c)
            """, code=row["Course Code"])
        print("[OK] Electives imported")

    def import_cs(self, rows):
        for row in rows:
            self.merge_course(row, "ComputerScience_test.xlsx")

            cat = row["Category"].lower()
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

    def import_specialization(self, spec, rows):
        for row in rows:
            self.merge_course(row, f"{spec}_test.xlsx")

            subgroup = row["Category"].title()
            if subgroup not in {"Required", "Optional"}:
                subgroup = "Optional"

            self.run("""
                MATCH (sg:SpecGroup {name:$subgroup, specialization:$spec})
                MATCH (c:Course {code:$code})
                MERGE (sg)-[:HAS_COURSE]->(c)
            """, subgroup=subgroup, spec=spec, code=row["Course Code"])

        print(f"[OK] {spec} imported")

    def import_minor(self, rows):
        for row in rows:
            self.merge_course(row, "Minor_test.xlsx")
            self.run("""
                MATCH (m:MinorType {name:$minor})
                MATCH (c:Course {code:$code})
                MERGE (m)-[:HAS_COURSE]->(c)
            """, minor=row["Category"], code=row["Course Code"])

        print("[OK] Minor imported")

# ============================================================
# MAIN
# ============================================================
def main():
    data = {}
    for key, file in FILES.items():
        rows = read_excel_safe(DATA_DIR / file)
        data[key] = rows
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
        importer.import_electives(data["Elective"])
        importer.import_cs(data["ComputerScience"])

        for spec in ["ACS", "AI", "BDA", "CSys", "SE"]:
            importer.import_specialization(spec, data[spec])

        importer.import_minor(data["Minor"])

        print("\n[DONE] Part 2 completed successfully")

    finally:
        importer.close()


if __name__ == "__main__":
    main()