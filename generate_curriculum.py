import json
import openai
from pathlib import Path
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
import shutil
import re


# -----------------------------
# 1. Configuration
# -----------------------------
# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# File paths
curriculum_file = Path("curriculum.json")
output_file = Path("generated_curriculum.json")

# -----------------------------
# 2. Load subjects and courses
# -----------------------------
with curriculum_file.open("r", encoding="utf-8") as f:
    curriculum_data = json.load(f)

# -----------------------------
# 3. GPT Prompt Template
# -----------------------------
def build_lesson_prompt(
    subject_name,
    course_title,
    lesson_title,
    lesson_short,
    lesson_summary="Provide a 2-3 sentence summary",
    bite_count=7,
    quiz_count=5,
    levels=("ELI5", "ELI12", "ELI16", "ELI25"),
):
    # Build bites dynamically
    bites = []
    for i in range(1, bite_count + 1):
        explanations = ",\n".join(
            [f'                "{lvl}": "{{{lvl.lower()}_text_{i}}}"' for lvl in levels]
        )
        bite = f"""
            {{
              "bite_id": "{lesson_short}_b{i}",
              "explanations": {{
{explanations}
              }}
            }}"""
        bites.append(bite)
    bites_str = ",\n".join(bites)

    # Build quizzes dynamically per level
    quizzes = []
    for i in range(1, quiz_count + 1):
        level_questions = ",\n".join(
            [f'                "{lvl}": "{{quiz_question_{lvl.lower()}_{i}}}"' for lvl in levels]
        )
        level_options = ",\n".join(
            [f'                "{lvl}": ["{{option_1_{lvl.lower()}_{i}}}", "{{option_2_{lvl.lower()}_{i}}}", "{{option_3_{lvl.lower()}_{i}}}"]' for lvl in levels]
        )
        level_answers = ",\n".join(
            [f'                "{lvl}": {{correct_option_index_{lvl.lower()}_{i}}}' for lvl in levels]
        )
        quiz = f"""
            {{
              "quiz_id": "{lesson_short}_q{i}",
              "type": "multiple_choice",
              "questions": {{
{level_questions}
              }},
              "options": {{
{level_options}
              }},
              "correct_answers": {{
{level_answers}
              }}
            }}"""
        quizzes.append(quiz)
    quizzes_str = ",\n".join(quizzes)

    # Wrap-up explanations
    wrap_up = ",\n".join([f'            "{lvl}": "{{wrap_up_{lvl.lower()}}}"' for lvl in levels])

    # Full prompt
    prompt = f"""
You are an expert educator and course designer. Generate a micro-learning lesson in JSON format. 
You are right now working on the subject of "{subject_name}" for the course "{course_title}". 
The current lesson is titled "{lesson_title}". You only need to generate the content for this single lesson.

The lesson should include:
- lesson_id: "{lesson_short}"
- title: "{lesson_title}"
- summary: "{lesson_summary}"
- bites: {str(bite_count)} entries, each with {len(levels)} explanations ({', '.join(levels)})
- quizzes: {quiz_count} multiple-choice quizzes, each with {len(levels)} versions and correct answer per level
- wrap_up: summary text for each level

ELI5 = Explain Like I'm 5 (simple, basic)
ELI12 = Explain Like I'm 12 (more detail, examples)
ELI16 = Explain Like I'm 16 (even more detail, some complexity)
ELI25 = Explain Like I'm 25 (full detail, technical accuracy)

Respond ONLY with valid JSON matching this structure (no extra text or code blocks).

Lesson:
{{
  "lesson_id": "{lesson_short}",
  "title": "{lesson_title}",
  "summary_ELI5": "ELI5_summary",
  "summary_ELI12": "ELI12_summary",
  "summary_ELI16": "ELI16_summary",
  "summary_ELI25": "ELI25_summary",
  "bites": [
{bites_str}
  ],
  "quizzes": [
{quizzes_str}
  ],
  "wrap_up": {{
{wrap_up}
  }}
}}
"""
    return prompt


# -----------------------------
# 4. Generate lesson via GPT
# -----------------------------
def generate_lesson(subject_name, course_title, lesson_title, lesson_short, course_description, bite_count=7, quiz_count=5, levels=("ELI5", "ELI12", "ELI16", "ELI25")):
    prompt = build_lesson_prompt(
        subject_name=subject_name,
        course_title=course_title,
        lesson_title=lesson_title,
        lesson_short=lesson_short,
        bite_count=bite_count,
        quiz_count=quiz_count,
        levels=levels,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert educator."},
            {"role": "user", "content": prompt},
        ],
        # temperature=0.7,
        max_tokens=5000,
    )

    text = response.choices[0].message.content

    # Remove leading/trailing whitespace
    cleaned = text.strip()

    # Strip Markdown fences like ```json ... ``` or ``` ...
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Failed to parse JSON, returning raw text")
        return {"error": text}

# -----------------------------
# 5. Iterate over subjects and courses and save per subject
# -----------------------------
output_folder = Path("generated_curriculum")
output_folder.mkdir(exist_ok=True)

for subject in curriculum_data["subjects"]:
    subject_file = output_folder / f"{subject['short']}.json"
    if subject['short'] != 'hist':
        continue

    # Load existing subject file if it exists
    if subject_file.exists():
        with subject_file.open("r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_courses = {c["short"]: c for c in existing_data["subjects"][0]["courses"]}
    else:
        existing_data = {"subjects": [{"name": subject["name"], "short": subject["short"], "courses": []}]}
        existing_courses = {}

    for course in subject["courses"]:
        existing_course = existing_courses.get(course["short"], {})
        existing_lessons = {l["lesson_id"]: l for l in existing_course.get("lessons", [])}

        # Ensure course exists in existing_data
        course_in_file = next((c for c in existing_data["subjects"][0]["courses"] if c["short"] == course["short"]), None)
        if not course_in_file:
            course_in_file = course.copy()
            course_in_file["lessons"] = []
            existing_data["subjects"][0]["courses"].append(course_in_file)

        for lesson in course.get("lessons", []):
            lesson_short = f"{course['short']}_{course.get('lessons', []).index(lesson)+1}"

            # Skip already generated lessons
            if lesson_short in existing_lessons:
                print(f"Skipping existing lesson: {subject['name']} > {course['title']} > {lesson}")
                continue

            print(f"Generating lesson for {subject['name']} > {course['title']} > {lesson}")
            course_description = course.get("description", f"Learn the basics of {course['title']}.")

            lesson_content = generate_lesson(subject, course, lesson, lesson_short, course_description)

            # Only save valid JSON lessons
            if "error" not in lesson_content:
                if "error" not in lesson_content:
                    course_in_file["lessons"].append(lesson_content)

                    # Safe save: write to temp file first
                    temp_file = subject_file.with_suffix(".json.tmp")
                    try:
                        with temp_file.open("w", encoding="utf-8") as f:
                            json.dump(existing_data, f, indent=2, ensure_ascii=False)

                        # Replace original file
                        shutil.move(str(temp_file), str(subject_file))
                        print(f"Safely saved lesson {lesson_short} to {subject_file}")

                    except Exception as e:
                        print(f"Failed to save lesson {lesson_short}: {e}")
                        if temp_file.exists():
                            temp_file.unlink()  # remove incomplete temp file
            else:
                print(f"Skipped malformed lesson for {lesson}. Error: {lesson_content['error']}")

            # Rate limit to avoid API throttling
            time.sleep(1.5)



