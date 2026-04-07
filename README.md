
# CSC3331 – Project 1: Graph Database (BSCS Flowchart)

CSC3331 project (Spring 2026) that builds a Neo4j graph database for the BSCS course‑sequence catalog using Python, OCR, and Excel files.

## 🎯 What this project does

- Uses **OCR** (Pyyesseract or similar) to extract course data from the **BSCS course‑sequence PDF** into Excel files.
- Generates 7 Excel files:
  - `Mathematics.xls`
  - `ComputerScience.xls`
  - `ScienceEngineering.xls`
  - `GenEd.xls`
  - `Elective.xls`
  - `AI.xls`
  - `Minor.xls`
- Each file has columns: `Course Code`, `Course Title`, `Course Credits`, `Prerequisites`, `Category` (use `"NONE"` if missing).

- A Python script reads these Excel files and builds a **Neo4j graph**:
  - Root node: `BSCS`.
  - Children: `GenEd`, `ComputerScience`, `Mathematics`, `ScienceEngineering`, `Electives`, `Minor`.
  - `ComputerScience` → `Required`, `Specializations`, `ComputingElective`.
  - `Specializations` → per‑specialization nodes (e.g., Software Engineering) with required/elective courses.
  - `GenEd` → blocks like `GenEd_SocialSciences` with their courses.
  - `Minor` → 3 minors (`Business Administration`, `General Engineering`, `Communication Studies`), each linked to 5 dummy courses `course1`–`course5`.

- Includes **Cypher queries** for:
  - Courses with “computer” in the title.
  - Prerequisites of a specialization course.
  - Art‑related courses and their prerequisites.
  - Enriching `Business Administration` with `required`/`optional` sub‑nodes and moving courses.
  - A sample course‑taking scenario when minoring in Business Administration.

## 📂 Folder structure
