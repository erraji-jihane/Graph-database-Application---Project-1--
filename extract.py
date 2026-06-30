import os
import re
import json
import base64
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import anthropic
from dotenv import load_dotenv

# these are the columns we expect in every course sheet
EXPECTED_COLUMNS = [
    "Course Code",
    "Course Title",
    "Course Credits",
    "Prerequisites",
    "Category",
]

# hardcoded some courses which pre-requisites are not detected 
HARDCODED_PREREQ_FIXES = {
    "CSC2305": "CSC2302, PHY1402",   
    "CSC3374": "CSC3351, CSC3326",  
    "EGR2302": "MTH1303",           
    "EGR4300": "FRN3210, ENG2303",   
    "EGR4402": "FRN3210, ENG2303",  
}

# output file names for each sheet
OUTPUT_FILES = {
    "math": "Mathematics.xlsx",
    "computer_science": "ComputerScience.xlsx",
    "science_engineering": "ScienceEngineering.xlsx",
    "gen_ed": "GenEd.xlsx",
    "elective": "Elective.xlsx",
    "ai": "AI.xlsx",
    "acs": "ACS.xlsx",
    "bda": "BDA.xlsx",
    "csys": "CSys.xlsx",
    "se": "SE.xlsx",
    "minor": "Minor.xlsx",
}


def pdf_to_base64(pdf_path: Path) -> str:
    # read the pdf file as bytes and encode it to base64 so claude can read it
    return base64.b64encode(pdf_path.read_bytes()).decode("utf-8")


def extract_json_from_text(text: str) -> Dict[str, Any]:
    # try to parse the raw text from claude as json
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # if claude wraps the json in ```json ... ``` then we want to remove that
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    # if the json is just inside a big text block, we can find { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    # if claude did not return valid json
    raise ValueError("Claude response did not contain valid JSON.")


def normalize_code(value: Any) -> str:     # uppercase and without spaces
    if value is None:
        return "NONE"
    s = str(value).strip()
    if not s:
        return "NONE"
    return s.replace(" ", "").upper()


def normalize_text(value: Any) -> str:
    # if it's missing values we use "NONE"
    if value is None:
        return "NONE"
    s = str(value).strip()
    return s if s else "NONE"


def normalize_prereq(value: Any) -> str:
    # Normalizing prerequisites into clear string (removing extra spaces) 
    if value is None:
        return "NONE"
    s = str(value).strip()
    if not s:
        return "NONE"
    s = re.sub(r"\s+", " ", s)   # compress multiple spaces into one
    return s


def normalize_credits(value: Any) -> int:
    # extracting course credit form course code 
    # if it's missing or no digits, we return 0
    if value is None:
        return 0
    s = str(value).strip()
    if not s:
        return 0
    match = re.search(r"\d+", s)
    return int(match.group()) if match else 0


def ensure_minor_rows() -> List[Dict[str, Any]]:
    # the minor sheet always has 5 rows, named COURSE1 to COURSE5
    rows = []
    for i in range(1, 6):
        rows.append(
            {
                "Course Code": f"COURSE{i}",
                "Course Title": "EMPTY",   # not filled in the pdf anyway
                "Course Credits": 3,        # all minors are 3 credits
                "Prerequisites": "EMPTY",
                "Category": "Minor",
            }
        )
    return rows


def normalize_rows(sheet_key: str, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    # take the raw course rows and clean them up
    cleaned: List[Dict[str, Any]] = []

    for row in rows:
        code = normalize_code(row.get("Course Code"))
        title = normalize_text(row.get("Course Title"))
        credits = normalize_credits(row.get("Course Credits"))
        prereq = normalize_prereq(row.get("Prerequisites"))
        category = normalize_text(row.get("Category"))

        # apply the hardcoded fixes for a few courses (the 5 ones that we hardcoded at the start since we found some problems with their extraction)
        if code in HARDCODED_PREREQ_FIXES:
            prereq = HARDCODED_PREREQ_FIXES[code]

        cleaned.append(
            {
                "Course Code": code,
                "Course Title": title,
                "Course Credits": credits,
                "Prerequisites": prereq,
                "Category": category,
            }
        )

    # if this is the minor sheet, we always use the 5 standard rows
    if sheet_key == "minor":
        cleaned = ensure_minor_rows()

    df = pd.DataFrame(cleaned)

    # if there are no rows we still want the expected columns
    if df.empty:
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    # make sure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = "NONE"

    # reorder columns so they are always the same
    df = df[EXPECTED_COLUMNS]

    # remove duplicate rows
    # computer_science should keep only unique (code, category) pairs
    # other sheets only need unique codes
    if sheet_key == "computer_science":
        df = df.drop_duplicates(subset=["Course Code", "Category"], keep="first")
    else:
        df = df.drop_duplicates(subset=["Course Code"], keep="first")

    df = df.reset_index(drop=True)
    return df


def propagate_specialization_rows(data: Dict[str, Any]) -> Dict[str, Any]:
    # making sure computer_science exists as a list
    if "computer_science" not in data or not isinstance(data["computer_science"], list):
        data["computer_science"] = []

  
    specialization_keys = ["ai", "acs", "bda", "csys", "se"]

    # keeping track of which (code, category) pairs already exist in computer_science
    existing_pairs = set()
    for row in data["computer_science"]:
        code = normalize_code(row.get("Course Code"))
        category = normalize_text(row.get("Category"))
        existing_pairs.add((code, category))

    # we also add the course in the specialization to the computer science excel file
    for spec_key in specialization_keys:
        rows = data.get(spec_key, [])
        if not isinstance(rows, list):
            continue

        for row in rows:
            code = normalize_code(row.get("Course Code"))
            if code == "NONE":
                continue

            pair = (code, "Specialization")
            if pair in existing_pairs:
                continue   # avoid adding it twice

            data["computer_science"].append(
                {
                    "Course Code": code,
                    "Course Title": normalize_text(row.get("Course Title")),
                    "Course Credits": normalize_credits(row.get("Course Credits")),
                    "Prerequisites": normalize_prereq(row.get("Prerequisites")),
                    "Category": "Specialization",
                }
            )
            existing_pairs.add(pair)

    return data


def move_egr_courses_to_science_engineering(data: Dict[str, Any]) -> Dict[str, Any]:
    # make sure computer_science exists as a list
    if "computer_science" not in data or not isinstance(data["computer_science"], list):
        data["computer_science"] = []

    if "science_engineering" not in data or not isinstance(data["science_engineering"], list):
        data["science_engineering"] = []

    # track which codes are already in science_engineering
    existing_se_codes = {
        normalize_code(row.get("Course Code"))
        for row in data["science_engineering"]
        if isinstance(row, dict)
    }

    remaining_cs = []

    # move any EGR‑prefixed course from cs to science_engineering
    for row in data["computer_science"]:
        code = normalize_code(row.get("Course Code"))

        if code.startswith("EGR"):
            moved_row = {
                "Course Code": code,
                "Course Title": normalize_text(row.get("Course Title")),
                "Course Credits": normalize_credits(row.get("Course Credits")),
                "Prerequisites": normalize_prereq(row.get("Prerequisites")),
                "Category": "Engineering",
            }

            if code not in existing_se_codes:
                data["science_engineering"].append(moved_row)
                existing_se_codes.add(code)
        else:
            remaining_cs.append(row)

    # after moving EGR courses, cs only has non‑EGR courses
    data["computer_science"] = remaining_cs
    return data


def build_prompt() -> str:
    # this is the system prompt that claude will use to extract data
    # it tells claude what json structure to return and how to format the data
    return """
You are extracting structured course data from ONE PDF:
the BSCSC course sequence PDF.


Your job is to return ONE valid JSON object only.
Do not add explanations.
Do not wrap the JSON in markdown.


Return this exact top-level structure:
{
  "math": [...],
  "computer_science": [...],
  "science_engineering": [...],
  "gen_ed": [...],
  "elective": [...],
  "ai": [...],
  "acs": [...],
  "bda": [...],
  "csys": [...],
  "se": [...],
  "minor": [...]
}


For every row in every array, use exactly these keys:
- "Course Code"
- "Course Title"
- "Course Credits"
- "Prerequisites"
- "Category"


Extraction rules:
1. Use only the PDF content.
2. If any value is missing, use "NONE".
3. For blocks containing several courses, create one row per course.
4. Course codes must not contain spaces. Example: "CSC 3309" becomes "CSC3309".
5. Course credits must be numeric.


6. Mathematics sheet:
- include math courses from the flowchart
- category must be exactly "Mathematics"


7. ScienceEngineering sheet:
- include science and engineering courses
- category must be exactly either "Science" or "Engineering"
- if a block lists multiple courses like BIO 1401, BIO 1402, CHE 1401, create one row per course


8. ComputerScience sheet:
- include all core/required computer science courses from page 1
- include specialization courses too if visible
- do NOT place EGR courses in computer_science
- category must be one of:
  "Required", "Specialization", "Computing elective"


9. Elective sheet:
- include specialization elective courses from page 2
- category must be exactly "elective"


10. Specialization sheets:
- create separate sheets for ai, acs, bda, csys, se
- categories must be exactly "Required" or "Optional"


11. GenEd sheet:
- include GenEd courses from page 1
- category should be either "GenEd" or a block category like:
  "GenEd_Arabic", "GenEd_French", "GenEd_Humanities",
  "GenEd_Art", "GenEd_HistoryPoliticalScience",
  "GenEd_SocialSciences", "GenEd_CivicEngagement"


12. Minor sheet:
- return exactly 5 rows only:
  COURSE1, COURSE2, COURSE3, COURSE4, COURSE5
- Course Title = EMPTY
- Prerequisites = EMPTY
- Course Credits = 3
- Category = Minor


13. Use visible prerequisite arrows and prerequisite text whenever available.
14. For specialization prerequisites, preserve expressions like:
- "CSC2306, CSC3323, MTH3301"
- "CSC3308 or CSC3309"
- "depends on the topic"


Return JSON only.
""".strip()


def call_claude(course_pdf: Path, model: str) -> Dict[str, Any]:
    # set up the anthropic client with the api key
    
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]) #from env

    # send the pdf and the prompt to claude (base64 used as claude can read it)
    # we use base64 because claude can read documents as base64
    response = client.messages.create(
        model=model,
        max_tokens=12000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_to_base64(course_pdf),
                        },
                    },
                    {
                        "type": "text",
                        "text": build_prompt(),
                    },
                ],
            }
        ],
    )

    # collect all the text parts from claude's response
    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)

    full_text = "\n".join(text_parts).strip()

    # parse the text into a json object
    # this is the main data we get from claude
    return extract_json_from_text(full_text)


def save_outputs(data: Dict[str, Any], out_dir: Path) -> None:
    # create the output directory if it doesn't exist (for teh excel files)
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # putting the specialization courses into the cs list
    data = propagate_specialization_rows(data)

    # moving EGR courses from cs to science_engineering
    data = move_egr_courses_to_science_engineering(data)

    # for each sheet key, clean the rows and save to an excel file
    for key, filename in OUTPUT_FILES.items():
        rows = data.get(key, [])
        df = normalize_rows(key, rows)
        output_path = out_dir / filename
        df.to_excel(output_path, index=False)
        print(f"Saved: {output_path}")


def main() -> None:
    # load the environment variables (api)
    load_dotenv()

    # this is the anthropic api key from the .env file
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is missing. Put it in your .env file.")

    # set up the directory where the script is running
    base_dir = Path(__file__).resolve().parent

    
    course_pdf = base_dir / "BSCSC_Course-Sequence_Catalog 2023-2025_Dec_2023.pdf"

    # output will go one level up in the "output" folder
    out_dir = (base_dir.parent / "output").resolve()

    # handling pdf not found error
    if not course_pdf.exists():
        raise FileNotFoundError(f"Missing course PDF: {course_pdf}")

    # choosing the claude model from env, default to sonnet‑4‑6
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()

    print(f"Using model: {model}")
    print("Sending PDF to Claude...")
    data = call_claude(course_pdf, model)
    print("Claude response parsed successfully.")

    # saving all the cleaned sheets to excel files
    save_outputs(data, out_dir)
    print(f"Done. Files are in: {out_dir}")


if __name__ == "__main__":
    main()