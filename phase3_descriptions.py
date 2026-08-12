import os
import json
import time
from google import genai
from google.genai import errors as genai_errors

# ---- Setup Gemini client (reads API key from environment variable) ----
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set. Run: $env:GEMINI_API_KEY='your_key' in terminal first.")

client = genai.Client(api_key=api_key)

# ---- Load class names (same subset as before) ----
data_dir = "CUB_200_2011/images"
all_classes = sorted(os.listdir(data_dir))
subset_classes = all_classes[:30]
class_names = [c.split('.', 1)[1].replace('_', ' ') for c in subset_classes]

# ---- Load any existing progress so we don't redo classes we already have ----
if os.path.exists("class_descriptions.json"):
    with open("class_descriptions.json", "r") as f:
        class_descriptions = json.load(f)
else:
    class_descriptions = {}

# ---- Generate descriptions for one class, with retry on rate limit ----
def generate_descriptions(class_name, n=5, max_retries=5):
    prompt = f"""List {n} short, visually distinctive descriptions of a {class_name} bird.
Focus on plumage color, beak shape, size, and markings that would help distinguish it
from similar bird species. One sentence each. No numbering, no bullet points, no extra text,
just {n} plain sentences separated by newlines."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )
            text = response.text
            lines = [line.strip("-• ").strip() for line in text.split('\n') if line.strip()]
            return lines[:n]
        except genai_errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60  # free tier resets per minute, wait a full minute to be safe
                print(f"    Rate limited, waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries for {class_name}")

# ---- Generate for all classes, skipping ones already done ----
for i, name in enumerate(class_names):
    if name in class_descriptions and class_descriptions[name]:
        print(f"[{i+1}/{len(class_names)}] Skipping (already have): {name}")
        continue

    print(f"[{i+1}/{len(class_names)}] Generating descriptions for: {name}")
    try:
        descs = generate_descriptions(name)
        class_descriptions[name] = descs
        for d in descs:
            print(f"    - {d}")
    except Exception as e:
        print(f"    ERROR: {e}")
        class_descriptions[name] = []

    # Save progress after every class, so a crash doesn't lose work
    with open("class_descriptions.json", "w") as f:
        json.dump(class_descriptions, f, indent=2)

    # Small delay between requests to stay under 15/min limit
    time.sleep(4)

print(f"\nDone. Saved descriptions for {len(class_descriptions)} classes to class_descriptions.json")