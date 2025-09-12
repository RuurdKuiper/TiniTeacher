import json
import base64
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
from google import generativeai as genai
from PIL import Image
from io import BytesIO
from google.generativeai import types


# Initialize OpenAI client
load_dotenv()
google_api_key=os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Paths
curriculum_folder = Path("generated_curriculum")
image_folder = Path("images")
image_folder.mkdir(exist_ok=True)

def expand_icon_prompt(name: str, level: str, parent: str | None = None) -> str:
    """
    Create a simplified, icon-style prompt.
    level: 'subject', 'course', or 'lesson'
    parent: optional parent heading (subject for course, course for lesson)
    """
    base = (
        f"A clean, modern flat-style icon representing the {level} '{name}'. Please make the icon very specific to this {level}."
        "The design should be minimal, symbolic, and consistent in visual style. "
        "Use smooth vector-like shapes, bright but limited colors, soft shadows, "
        "and avoid text, numbers, or equations. "
        "It should look like an educational app icon, not a detailed scene."
    )
    
    if parent:
        base += f" This {level} is part of the '{parent}' { 'subject' if level=='course' else 'course' }."
    
    return base



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
            "into a clear prompt for an image generator. Describe very simple, informative images, "
            "with minimal elements, and a clean, modern style."
            "Describe a simple scene with only one or a few elements. "
            "Always use this format:\n\n"
            "General Template:\n"
            "A modern, clean educational illustration in a consistent visual style. "
            "The image should use smooth shapes, soft shadows, and bright but limited colors. "
            "Show the concept visually without text or labels, relying only on simple forms, "
            "and color coding. Avoid realism — keep it minimal, clear, and informative.\n"
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



def generate_image(prompt: str, out_path: Path, model_name: str):
    """Generate and save an image from a text prompt using the specified model."""
    try:
        # Check for a valid model name
        if model_name not in ["gpt-image-1", "gemini-2.5-flash-image-preview", "imagen-4.0-fast-generate-001"]:
            print(f"❌ Error: Unsupported model '{model_name}'.")
            return False

        # --- OpenAI Model ---
        if model_name == "gpt-image-1":
            openai_client = OpenAI()
            result = openai_client.images.generate(
                model=model_name,
                prompt=prompt,
                size="1024x1024",
                quality="medium"
            )
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            print(f"✅ Saved image to {out_path}")
            return True

        # --- Gemini Model ---
        elif model_name == "gemini-2.5-flash-image-preview" or model_name == "imagen-4.0-fast-generate-001":
            genai.configure() # Ensure the API key is configured
            
            # Use 'contents=[prompt]' as per the Gemini API
            response = genai.GenerativeModel(model_name=model_name).generate_content(
                contents=[prompt],
                stream=False
            )

            image_found = False
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None and part.inline_data.mime_type.startswith("image/"):
                    image = Image.open(BytesIO(part.inline_data.data))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(out_path)
                    image_found = True
                    break
            
            if image_found:
                print(f"✅ Saved image to {out_path}")
                return True
            else:
                print(f"❌ Failed to generate {out_path}: No image data found in Gemini response.")
                return False
                
        elif model_name == "imagen-4.0-fast-generate-001":
            genai.configure(api_key=google_api_key)
            response = client.models.generate_images(
                model='imagen-4.0-generate-001',
                prompt='Robot holding a red skateboard',
                config=types.GenerateImagesConfig(
                    number_of_images= 4,
                )
            )

    except Exception as e:
        print(f"❌ Failed to generate {out_path} with model '{model_name}': {e}")
        return False


def create_prompt(subject, course, lesson, bite):
    """Ask GPT-4o to expand the explanation into a strong image prompt."""
    explanation = bite["explanations"].get("ELI16") or bite["explanations"].get("ELI12")
    if not explanation:
        return None
    expanded = expand_prompt(explanation, subject["name"], course["title"], lesson["title"])
    print(f"🔍 Expanded prompt:\n{expanded}\n")
    return expanded

image_generator_model = "gpt-image-1"
# Iterate over all subject JSONs
for subject_file in curriculum_folder.glob("*.json"):
    with subject_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    subject = data["subjects"][0]

    # Subject icon
    subject_out = image_folder / subject["short"] / f"{subject['short']}_icon.png"
    subject_out.parent.mkdir(parents=True, exist_ok=True)
    if not subject_out.exists():
        prompt = expand_icon_prompt(subject["name"], "subject")
        print(prompt)
        generate_image(prompt, subject_out, image_generator_model)

    # Course icons
    for course in subject["courses"]:
        course_out = image_folder / subject["short"] / course["short"] / f"{course['short']}_icon.png"
        course_out.parent.mkdir(parents=True, exist_ok=True)
        if not course_out.exists():
            prompt = expand_icon_prompt(course["title"], "course", parent=subject["name"])
            print(prompt)
            generate_image(prompt, course_out, image_generator_model)

        # Lesson icons
        for lesson in course["lessons"]:
            lesson_out = image_folder / subject["short"] / course["short"] / f"{lesson['lesson_id']}_icon.png"
            lesson_out.parent.mkdir(parents=True, exist_ok=True)
            if not lesson_out.exists():
                prompt = expand_icon_prompt(lesson["title"], "lesson", parent=course["title"])
                print(prompt)
                generate_image(prompt, lesson_out, image_generator_model)







# Iterate over all subject JSONs
generated = True
for subject_file in curriculum_folder.glob("*.json"):
    with subject_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if subject_file.stem != "hist":
        continue

    subject = data["subjects"][0]
    for course in subject["courses"]:
        for lesson in course["lessons"]:
            for bite in lesson["bites"]:
                bite_id = bite["bite_id"]
                out_path = image_folder / subject["short"] / course["short"] / f"{bite_id}.png"
                print(out_path)

                if out_path.exists():
                    print(f"⏭️ Skipping existing image: {out_path}")
                    continue
                print(bite)
                prompt = create_prompt(subject, course, lesson, bite)

                if not prompt:
                    continue

                generated = generate_image(prompt, out_path, image_generator_model)
    #             if not generated:
    #                 break
    #         if not generated:
    #             break
    #     if not generated:
    #         break
    # if not generated:
    #     break
