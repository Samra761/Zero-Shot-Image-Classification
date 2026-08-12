import os
import torch
import open_clip
from PIL import Image

# ---- Load class subset (same as phase 1) ----
data_dir = "CUB_200_2011/images"
all_classes = sorted(os.listdir(data_dir))
subset_classes = all_classes[:30]
class_names = [c.split('.', 1)[1].replace('_', ' ') for c in subset_classes]

# ---- Load train/test split info ----
images_txt = "CUB_200_2011/images.txt"          # image_id -> relative path
split_txt = "CUB_200_2011/train_test_split.txt"  # image_id -> is_train (1) or test (0)

id_to_path = {}
with open(images_txt, 'r') as f:
    for line in f:
        img_id, path = line.strip().split(' ', 1)
        id_to_path[img_id] = path

id_to_split = {}
with open(split_txt, 'r') as f:
    for line in f:
        img_id, is_train = line.strip().split(' ')
        id_to_split[img_id] = int(is_train)

# ---- Build list of test images for our subset classes only, capped per class ----
MAX_PER_CLASS = 15
test_images = []  # list of (image_path, true_class_name)

for img_id, rel_path in id_to_path.items():
    if id_to_split[img_id] != 0:  # 0 = test image
        continue
    class_folder = rel_path.split('/')[0]  # e.g. "001.Black_footed_Albatross"
    if class_folder not in subset_classes:
        continue
    full_path = os.path.join(data_dir, rel_path)
    class_name = class_folder.split('.', 1)[1].replace('_', ' ')
    test_images.append((full_path, class_name))

# Cap per class
from collections import defaultdict
capped_test_images = []
counts = defaultdict(int)
for path, cname in test_images:
    if counts[cname] < MAX_PER_CLASS:
        capped_test_images.append((path, cname))
        counts[cname] += 1

print(f"Total test images selected: {len(capped_test_images)}")
print(f"Classes covered: {len(counts)}")

# ---- Load CLIP model ----
print("\nLoading CLIP model (this may take a minute)...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# ---- Build baseline text prompts ----
prompts = [f"a photo of a {name}, a type of bird" for name in class_names]
text_tokens = tokenizer(prompts)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ---- Classify each test image ----
def classify_image(img_path, text_features, class_names):
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_idx = similarity.argmax().item()
    return class_names[top_idx]

print("\nRunning baseline classification on test images...")
correct = 0
total = 0
results_log = []  # for later failure analysis

for i, (img_path, true_class) in enumerate(capped_test_images):
    pred_class = classify_image(img_path, text_features, class_names)
    is_correct = (pred_class == true_class)
    correct += int(is_correct)
    total += 1
    results_log.append({"image": img_path, "true": true_class, "pred": pred_class, "correct": is_correct})
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(capped_test_images)}...")

baseline_accuracy = correct / total
print(f"\n=== BASELINE RESULTS ===")
print(f"Correct: {correct}/{total}")
print(f"Baseline Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")

# ---- Save results for later comparison ----
import json
with open("baseline_results.json", "w") as f:
    json.dump({"accuracy": baseline_accuracy, "results": results_log}, f, indent=2)

print("\nSaved results to baseline_results.json")