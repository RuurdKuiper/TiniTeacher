import json
import base64
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Paths
curriculum_folder = Path("generated_curriculum")
image_folder = Path("images")
image_folder.mkdir(exist_ok=True)

def expand_icon_prompt(name: str, level: str) -> str:
    """
    Create a simplified, icon-style prompt for subject or course.
    Level is either 'subject' or 'course'.
    """
    return (
        f"A clean, modern flat-style icon representing the {level} '{name}'. "
        "The design should be minimal, symbolic, and consistent in visual style. "
        "Use smooth vector-like shapes, bright but limited colors, soft shadows, "
        "and avoid text, numbers, or equations. "
        "It should look like an educational app icon, not a detailed scene."
    )


def expand_prompt(explanation: str, subject: str, course: str, lesson: str) -> str:
    """
    Use GPT-4o to generate a detailed, visual prompt for image generation
    based on the educational explanation. The output always starts with the
    general template followed by the concept-specific description.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert educational illustrator prompt-writer. "
            "Your job is to take a short explanation of a concept and expand it "
            "into a rich, detailed prompt for an image generator. "
            "Always use this format:\n\n"
            "General Template:\n"
            "A modern, clean educational illustration in a consistent visual style. "
            "The image should use smooth shapes, soft shadows, and bright but limited colors. "
            "Show the concept visually without text or labels, relying only on simple forms, and, only if really necessary, arrows, "
            "and color coding. Avoid realism or childish style — keep it minimal, clear, and informative.\n"
            "Concept: <your detailed visual description of the concept here>\n\n"
            "Never include text, numbers, or equations inside the image."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            "This is an example of a successfully created image prompt using the input:\n\n"
            "\"Biology > Cell Biology > Cell Structure: Ribosomes create protein chains using instructions "
            "from the nucleus, essential for repairing and building cell parts.\"\n\n"
            "Output:\n"
            "General Template:\n"
            "A modern, clean educational illustration in a consistent visual style. "
            "The image should use smooth shapes, soft shadows, and bright but limited colors. "
            "Show the concept visually without text or labels, relying only on simple forms, and, only if really necessary, arrows, "
            "and color coding. Avoid realism or childish style — keep it minimal, clear, and informative.\n"
            "Concept: A simple cell cross-section with a blue nucleus containing DNA, an orange strand of "
            "mRNA leaving the nucleus, small dark ribosomes on the ER and in the cytoplasm, and each ribosome "
            "producing a short green bead-chain protein moving outward toward the cell membrane.\n\n"
            f"Now create a similar prompt for:\n"
            f"Subject: {subject} > {course} > {lesson}\n"
            f"Concept explanation: {explanation}"
        ),
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[system_msg, user_msg],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()



def generate_image(prompt: str, out_path: Path):
    """Generate and save an image from a text prompt."""
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="high"
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        print(f"✅ Saved image to {out_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to generate {out_path}: {e}")
        return False


def create_prompt(subject, course, lesson, bite):
    """Ask GPT-4o to expand the explanation into a strong image prompt."""
    explanation = bite["explanations"].get("ELI16") or bite["explanations"].get("ELI12")
    if not explanation:
        return None
    expanded = expand_prompt(explanation, subject["name"], course["title"], lesson["title"])
    print(f"🔍 Expanded prompt:\n{expanded}\n")
    return expanded


# Iterate over all subject JSONs
for subject_file in curriculum_folder.glob("*.json"):
    with subject_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    subject = data["subjects"][0]

    # --- Subject icon ---
    subject_out = image_folder / subject["short"] / f"{subject['short']}_icon.png"
    if not subject_out.exists():
        prompt = expand_icon_prompt(subject["name"], "subject")
        print(f"Generating subject icon for {subject['name']} with prompt:\n{prompt}\n")
        generate_image(prompt, subject_out)

    # # --- Course icons ---
    # for course in subject["courses"]:
    #     course_out = image_folder / subject["short"] / course["short"] / f"{course['short']}_icon.png"
    #     if not course_out.exists():
    #         prompt = expand_icon_prompt(course["title"], "course")
    #         print(f"Generating course icon for {course['title']} with prompt:\n{prompt}\n")
    #         generate_image(prompt, course_out)



# # Iterate over all subject JSONs
# generated = True
# for subject_file in curriculum_folder.glob("*.json"):
#     with subject_file.open("r", encoding="utf-8") as f:
#         data = json.load(f)

#     if subject_file.stem != "bio":
#         continue

#     subject = data["subjects"][0]
#     for course in subject["courses"]:
#         for lesson in course["lessons"]:
#             for bite in lesson["bites"]:
#                 bite_id = bite["bite_id"]
#                 out_path = image_folder / subject["short"] / course["short"] / f"{bite_id}.png"
#                 print(out_path)

#                 if out_path.exists():
#                     print(f"⏭️ Skipping existing image: {out_path}")
#                     continue
#                 print(bite)
#                 prompt = create_prompt(subject, course, lesson, bite)

#                 if not prompt:
#                     continue

#                 generated = generate_image(prompt, out_path)
#                 if not generated:
#                     break
#             if not generated:
#                 break
#         if not generated:
#             break
#     if not generated:
#         break
