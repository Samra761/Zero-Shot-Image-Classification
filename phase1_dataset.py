import os

data_dir = "CUB_200_2011/images"
all_classes = sorted(os.listdir(data_dir))

# Pick a genuinely confusable subset - similar species that are visually close
# Using first 30 classes covers Albatross/Auklet/Blackbird/Bunting groups - good fine-grained mix
subset_classes = all_classes[:30]

print(f"Using {len(subset_classes)} classes for this project:")
for c in subset_classes:
    print(f"  {c}")

# Clean names for prompts later (e.g. "001.Black_footed_Albatross" -> "Black footed Albatross")
class_names = [c.split('.', 1)[1].replace('_', ' ') for c in subset_classes]
print("\nCleaned class names (for CLIP prompts):")
for name in class_names:
    print(f"  {name}")