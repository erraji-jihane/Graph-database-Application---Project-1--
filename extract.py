import os
import re
import json
from pathlib import Path

import fitz  # PyMuPDF
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "BSCSC_Course-Sequence_Catalog 2023-2025_Dec_2023.pdf"
OUTPUT_DIR = BASE_DIR / "output_xlsx"
MODEL_NAME = "gemini-2.5-flash"

EXPECTED_FILES = {
    "Mathematics": "Mathematics.xlsx",
    "ComputerScience": "ComputerScience.xlsx",
    "ScienceEngineering": "ScienceEngineering.xlsx",
    "GenEd": "GenEd.xlsx",
    "Elective": "Elective.xlsx",
    "Minor": "Minor.xlsx",
    "ACS": "ACS.xlsx",
    "AI": "AI.xlsx",
    "BDA": "BDA.xlsx",
    "CSys": "CSys.xlsx",
    "SE": "SE.xlsx",
}

REQUIRED_COLUMNS = [
    "Course Code",
    "Course Title",
    "Course Credits",
    "Prerequisites",
    "Category",
]

SPEC_BUCKETS = ["ACS", "AI", "BDA", "CSys", "SE"]

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CLEANUP RULES ONLY (post-extraction cleanup)
# ============================================================
# These are not used to create the dataset from scratch.
# They are only used to clean mismatched prerequisite values after extraction.
CLEAN_PREREQ_OVERRIDES = {
    # Mathematics
    "MTH1303": "NONE",
    "MTH1304": "NONE",
    "MTH2301": "MTH1303",
    "MTH2320": "MTH2301, MTH1304",
    "MTH3301": "MTH2320",

    # Science / Engineering
    "PHY1401": "NONE",
    "PHY1402": "PHY1401",
    "EGR2302": "PHY1401",

    # Main CS core
    "CSC1401": "NONE",
    "CSC2302": "CSC1401",
    "CSC2306": "CSC2302",
    "CSC2305": "CSC2302",
    "CSC3315": "CSC2306",
    "CSC3323": "CSC2306",
    "CSC3324": "CSC2306",
    "CSC3326": "CSC3323",
    "CSC3351": "CSC3315, CSC2305",
    "CSC3374": "CSC3351",
    "CSC3371": "CSC3323",

    # Page 2 specialization cleanup
    "CSC3309": "CSC2306, CSC3323, MTH3301",
    "CSC4308": "CSC3371",
    "CSC3328": "CSC3351",
    "CSC4399": "depends on the topic",

    "CSC3347": "CSC3308 or CSC3309",
    "CSC3310": "CSC3308 or CSC3309",
    "CSC3311": "CSC3308 or CSC3309",
    "CSC3348": "CSC3308 or CSC3309",

    "CSC3331": "CSC3326",
    "CSC4352": "CSC3331",
    "CSC3329": "CSC3371",
    "CSC3346": "CSC3326",
    "CSC3349": "CSC3326",
    "CSC4351": "MTH3301",

    "CSC3373": "CSC3351, CSC3371",
    "CSC3376": "CSC3351",

    "CSC4307": "CSC3326, CSC3351",
    "CSC4309": "CSC3374",
    "CSC3357": "CSC2306",
    "CSC3358": "CSC3326",
    "CSC3359": "CSC2306",
    "CSC4306": "CSC3324",
}


# ============================================================
# HELPERS
# ============================================================
def normalize_text(value):
    if value is None:
        return "NONE"
    value = str(value).strip()
    if not value or value.lower() in {"nan", "null", "none", "n/a"}:
        return "NONE"
    return value


def normalize_course_code(code):
    code = normalize_text(code).upper().replace(" ", "")
    code = code.replace("^", "")
    return code


def infer_credits_from_code(course_code, current_value="NONE"):
    """
    Credit = 2nd digit of the 4-digit numeric part.
    Examples:
      CSC1401 -> 4
      CSC2302 -> 3
      MTH1303 -> 3
    """
    current_value = normalize_text(current_value)

    # If already a clean numeric value, keep it
    if current_value.isdigit():
        return current_value

    code = normalize_course_code(course_code)
    match = re.search(r"[A-Z]{2,4}(\d{4})", code)
    if not match:
        return "NONE"

    digits = match.group(1)
    return digits[1]


def extract_json_from_text(text):
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Could not parse JSON from Gemini response.")


def extract_pdf_text(pdf_path):
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        pages.append(f"\n===== PAGE {i+1} =====\n{text}")
    doc.close()
    return "\n".join(pages)


def ensure_required_columns(records):
    cleaned = []
    for item in records:
        row = {}
        for col in REQUIRED_COLUMNS:
            row[col] = normalize_text(item.get(col, "NONE"))
        cleaned.append(row)
    return cleaned


def deduplicate_rows(records):
    seen = set()
    output = []
    for row in records:
        key = tuple(normalize_text(row.get(c, "NONE")) for c in REQUIRED_COLUMNS)
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def build_minor_rows():
    return [
        {
            "Course Code": f"course{i}",
            "Course Title": "EMPTY",
            "Course Credits": "3",
            "Prerequisites": "EMPTY",
            "Category": "Minor",
        }
        for i in range(1, 6)
    ]


# ============================================================
# PROMPT
# ============================================================
def build_prompt(raw_text):
    return f"""
You are extracting structured academic course data from a BSCSC flowchart PDF.

Return ONLY valid JSON.

Use EXACTLY these top-level keys:
{{
  "Mathematics": [],
  "ComputerScience": [],
  "ScienceEngineering": [],
  "GenEd": [],
  "Elective": [],
  "Minor": [],
  "ACS": [],
  "AI": [],
  "BDA": [],
  "CSys": [],
  "SE": []
}}

Each row must be an object with EXACTLY these keys:
{{
  "Course Code": "...",
  "Course Title": "...",
  "Course Credits": "...",
  "Prerequisites": "...",
  "Category": "..."
}}

Rules:
1. Use only information from the PDF text.
2. If information is missing, put "NONE".
3. Do NOT invent courses.
4. For grouped blocks with multiple courses, create one row per course.
5. Preserve prerequisite text as clearly as possible.

Category rules:
- Mathematics:
  only math courses from Area 1(a)
  Category = "Mathematics"

- ComputerScience:
  include ALL computer science courses appearing in the PDF,
  including:
    a) main page 1 CS core courses
    b) specialization courses from page 2
    c) computing elective placeholder
    d) specialization course placeholders from page 1
  Category must be ONLY one of:
    "Required", "Specialization", "Computing elective"

- ScienceEngineering:
  include page 1 science and engineering courses
  Category = "Science" or "Engineering"

- GenEd:
  include GenEd page 1 courses and listed options
  Category must be "GenEd" or "GenEd_BlockTitle"
  Examples:
    "GenEd_Arabic"
    "GenEd_Humanities"
    "GenEd_French"
    "GenEd_SocialSciences"
    "GenEd_HistoryOrPoliticalScience"
    "GenEd_ArtAppreciationCreation"
    "GenEd_CivicEngagement"

- Elective:
  include elective options from page 2 specialization sections
  Category = "elective"

- Minor:
  create exactly 5 rows:
    course1, course2, course3, course4, course5
  Course Title = EMPTY
  Prerequisites = EMPTY
  Course Credits = 3
  Category = Minor

- ACS / AI / BDA / CSys / SE:
  specialization files from page 2
  required courses => Category = "Required"
  elective choices => Category = "Optional"

Important:
- Include ALL "Choose 1" specialization options in the specialization files.
- Do not skip courses just because credits are missing.
- Return JSON only.

PDF text:
{raw_text}
""".strip()


# ============================================================
# GEMINI CALL
# ============================================================
def call_gemini(pdf_path):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in .env or environment.")

    client = genai.Client(api_key=api_key)
    raw_text = extract_pdf_text(pdf_path)
    prompt = build_prompt(raw_text)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    data = extract_json_from_text(response.text)
    return data


# ============================================================
# CLEANUP PIPELINE
# ============================================================
def merge_specializations_into_computerscience(final):
    """
    Make sure ComputerScience contains all CSC courses,
    including specialization CSC courses.
    """
    cs_rows = final.get("ComputerScience", [])
    existing_codes = {normalize_course_code(r.get("Course Code", "NONE")) for r in cs_rows}

    for spec_bucket in SPEC_BUCKETS:
        for row in final.get(spec_bucket, []):
            code = normalize_course_code(row.get("Course Code", "NONE"))
            if not code.startswith("CSC"):
                continue

            if code not in existing_codes:
                cs_rows.append({
                    "Course Code": code,
                    "Course Title": normalize_text(row.get("Course Title", "NONE")),
                    "Course Credits": infer_credits_from_code(code, row.get("Course Credits", "NONE")),
                    "Prerequisites": normalize_text(row.get("Prerequisites", "NONE")),
                    "Category": "Specialization",
                })
                existing_codes.add(code)

    final["ComputerScience"] = deduplicate_rows(ensure_required_columns(cs_rows))
    return final


def rebuild_elective_from_specializations(final):
    """
    Rebuild Elective.xlsx from specialization optional courses.
    """
    elective_rows = []
    seen = set()

    for spec_bucket in SPEC_BUCKETS:
        for row in final.get(spec_bucket, []):
            cat = normalize_text(row.get("Category", "NONE")).lower()
            code = normalize_course_code(row.get("Course Code", "NONE"))
            if cat not in {"optional", "elective"}:
                continue
            if code == "NONE":
                continue

            key = (code, normalize_text(row.get("Course Title", "NONE")))
            if key in seen:
                continue
            seen.add(key)

            elective_rows.append({
                "Course Code": code,
                "Course Title": normalize_text(row.get("Course Title", "NONE")),
                "Course Credits": infer_credits_from_code(code, row.get("Course Credits", "NONE")),
                "Prerequisites": normalize_text(row.get("Prerequisites", "NONE")),
                "Category": "elective",
            })

    final["Elective"] = deduplicate_rows(ensure_required_columns(elective_rows))
    return final


def apply_cleanup_rules(final):
    for bucket, rows in final.items():
        for row in rows:
            code = normalize_course_code(row.get("Course Code", "NONE"))
            row["Course Code"] = code

            # credits from course code
            row["Course Credits"] = infer_credits_from_code(code, row.get("Course Credits", "NONE"))

            # cleanup prerequisites only after extraction
            if code in CLEAN_PREREQ_OVERRIDES:
                row["Prerequisites"] = CLEAN_PREREQ_OVERRIDES[code]
            else:
                row["Prerequisites"] = normalize_text(row.get("Prerequisites", "NONE"))

            row["Course Title"] = normalize_text(row.get("Course Title", "NONE"))
            row["Category"] = normalize_text(row.get("Category", "NONE"))

    return final


def normalize_computerscience_categories(final):
    """
    Restrict ComputerScience categories to:
    Required / Specialization / Computing elective
    """
    for row in final.get("ComputerScience", []):
        code = normalize_course_code(row.get("Course Code", "NONE"))
        title = normalize_text(row.get("Course Title", "NONE"))
        cat = normalize_text(row.get("Category", "NONE")).lower()

        if title == "Computing Elective":
            row["Category"] = "Computing elective"
        elif title.startswith("Specialization Course"):
            row["Category"] = "Specialization"
        elif code.startswith("CSC") and cat in {"optional", "elective", "specialization"}:
            row["Category"] = "Specialization"
        elif code.startswith("CSC"):
            row["Category"] = "Required"

    return final


def postprocess_data(data):
    final = {}

    for bucket in EXPECTED_FILES.keys():
        records = data.get(bucket, [])
        if not isinstance(records, list):
            records = []
        records = ensure_required_columns(records)
        records = deduplicate_rows(records)
        final[bucket] = records

    # Minor placeholder rows
    if not final["Minor"]:
        final["Minor"] = build_minor_rows()

    # Keep page 1 placeholders if Gemini misses them
    existing_titles = {normalize_text(r.get("Course Title", "NONE")) for r in final["ComputerScience"]}
    placeholders = [
        {
            "Course Code": "NONE",
            "Course Title": "Computing Elective",
            "Course Credits": "3",
            "Prerequisites": "NONE",
            "Category": "Computing elective",
        },
        {
            "Course Code": "NONE",
            "Course Title": "Specialization Course 1",
            "Course Credits": "NONE",
            "Prerequisites": "NONE",
            "Category": "Specialization",
        },
        {
            "Course Code": "NONE",
            "Course Title": "Specialization Course 2",
            "Course Credits": "NONE",
            "Prerequisites": "NONE",
            "Category": "Specialization",
        },
        {
            "Course Code": "NONE",
            "Course Title": "Specialization Course 3",
            "Course Credits": "NONE",
            "Prerequisites": "NONE",
            "Category": "Specialization",
        },
    ]
    for row in placeholders:
        if row["Course Title"] not in existing_titles:
            final["ComputerScience"].append(row)

    # Merge specialization CSC courses into ComputerScience
    final = merge_specializations_into_computerscience(final)

    # Rebuild Elective from specialization optional courses
    final = rebuild_elective_from_specializations(final)

    # Apply post-extraction cleanup
    final = apply_cleanup_rules(final)

    # Normalize CS category values
    final = normalize_computerscience_categories(final)

    # Final dedupe
    for bucket in final:
        final[bucket] = deduplicate_rows(ensure_required_columns(final[bucket]))

    return final


# ============================================================
# EXCEL WRITER
# ============================================================
HDR_FILL = PatternFill("solid", fgColor="1F497D")
HDR_FONT = Font(bold=True, color="FFFFFF", size=11)
ALT_FILL = PatternFill("solid", fgColor="DCE6F1")
COL_WIDTHS = [16, 60, 15, 40, 24]


def write_excel(filename, rows, title="Sheet1"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = title[:31]

    ws.append(REQUIRED_COLUMNS)

    for col in range(1, 6):
        cell = ws.cell(1, col)
        cell.font = HDR_FONT
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 20

    for i, row in enumerate(rows, start=2):
        ws.append([row.get(col, "NONE") for col in REQUIRED_COLUMNS])

        if i % 2 == 0:
            for col in range(1, 6):
                ws.cell(i, col).fill = ALT_FILL

        for col in range(1, 6):
            ws.cell(i, col).alignment = Alignment(wrap_text=True, vertical="top")

    for idx, width in enumerate(COL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(idx)].width = width

    ws.freeze_panes = "A2"
    wb.save(str(OUTPUT_DIR / filename))


# ============================================================
# MAIN
# ============================================================
def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"Using PDF: {PDF_PATH}")
    print("Calling Gemini API...")

    raw_data = call_gemini(PDF_PATH)
    final = postprocess_data(raw_data)

    print("Saving XLSX files...")
    for bucket, filename in EXPECTED_FILES.items():
        rows = final.get(bucket, [])
        write_excel(filename, rows, title=bucket)
        print(f"Saved: {OUTPUT_DIR / filename} ({len(rows)} rows)")

    print("\nDone.")
    print(f"Files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()