import os
import json
import torch
import open_clip
from PIL import Image
from collections import defaultdict

# ---- Load class subset (same as before) ----
data_dir = "CUB_200_2011/images"
all_classes = sorted(os.listdir(data_dir))
subset_classes = all_classes[:30]
class_names = [c.split('.', 1)[1].replace('_', ' ') for c in subset_classes]

# ---- Load LLM-generated descriptions ----
with open("class_descriptions.json", "r") as f:
    class_descriptions = json.load(f)

# ---- Load train/test split info (same as Phase 2) ----
images_txt = "CUB_200_2011/images.txt"
split_txt = "CUB_200_2011/train_test_split.txt"

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

MAX_PER_CLASS = 15
test_images = []
for img_id, rel_path in id_to_path.items():
    if id_to_split[img_id] != 0:
        continue
    class_folder = rel_path.split('/')[0]
    if class_folder not in subset_classes:
        continue
    full_path = os.path.join(data_dir, rel_path)
    class_name = class_folder.split('.', 1)[1].replace('_', ' ')
    test_images.append((full_path, class_name))

counts = defaultdict(int)
capped_test_images = []
for path, cname in test_images:
    if counts[cname] < MAX_PER_CLASS:
        capped_test_images.append((path, cname))
        counts[cname] += 1

print(f"Total test images: {len(capped_test_images)}")

# ---- Load CLIP model (same as Phase 2) ----
print("\nLoading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# ---- Build enhanced text features by averaging description embeddings per class ----
print("Building enhanced text features from LLM descriptions...")

def get_enhanced_text_features(class_descriptions, class_names):
    all_features = []
    for name in class_names:
        descs = class_descriptions.get(name, [])
        if not descs:
            descs = [f"a photo of a {name}, a type of bird"]
        formatted_descs = [f"a photo of a {name}, which {d[0].lower()}{d[1:]}" for d in descs]
        tokens = tokenizer(formatted_descs)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
            avg_feat = feats.mean(dim=0)
            avg_feat /= avg_feat.norm()
        all_features.append(avg_feat)
    return torch.stack(all_features)

enhanced_text_features = get_enhanced_text_features(class_descriptions, class_names)

# ---- Classify function (same as Phase 2) ----
def classify_image(img_path, text_features, class_names):
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_idx = similarity.argmax().item()
    return class_names[top_idx]

# ---- Run classification with enhanced prompts ----
print("\nRunning enhanced classification on test images...")
correct = 0
total = 0
results_log = []

for i, (img_path, true_class) in enumerate(capped_test_images):
    pred_class = classify_image(img_path, enhanced_text_features, class_names)
    is_correct = (pred_class == true_class)
    correct += int(is_correct)
    total += 1
    results_log.append({"image": img_path, "true": true_class, "pred": pred_class, "correct": is_correct})
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(capped_test_images)}...")

enhanced_accuracy = correct / total
print(f"\n=== ENHANCED RESULTS ===")
print(f"Correct: {correct}/{total}")
print(f"Enhanced Accuracy: {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")

with open("enhanced_results.json", "w") as f:
    json.dump({"accuracy": enhanced_accuracy, "results": results_log}, f, indent=2)

print("\nSaved results to enhanced_results.json")

# ---- Quick comparison against baseline if available ----
if os.path.exists("baseline_results.json"):
    with open("baseline_results.json", "r") as f:
        baseline = json.load(f)
    baseline_acc = baseline["accuracy"]
    print(f"\n=== COMPARISON ===")
    print(f"Baseline (plain prompt) accuracy:   {baseline_acc*100:.2f}%")
    print(f"Enhanced (LLM description) accuracy: {enhanced_accuracy*100:.2f}%")
    diff = (enhanced_accuracy - baseline_acc) * 100
    sign = "+" if diff >= 0 else ""
    print(f"Difference: {sign}{diff:.2f} percentage points")