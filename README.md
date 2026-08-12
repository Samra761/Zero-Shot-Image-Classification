# Zero-Shot Fine-Grained Bird Classification with LLM-Generated Semantic Descriptions

## Overview

This internship project investigates **Zero-Shot Image Classification (ZSIC)** for fine-grained visual recognition.

The main research question is:

> **Can an LLM automatically generate useful, class-discriminative semantic descriptions that improve zero-shot classification of visually similar image categories?**

The project uses a subset of the **CUB-200-2011 (Caltech-UCSD Birds)** dataset and a CLIP-based zero-shot classifier. Gemini-generated bird descriptions are used as additional semantic information and compared against a plain class-name baseline.

A key finding of the experiment is that **LLM-generated descriptions can either hurt or improve CLIP performance depending on how the descriptions are formatted before being encoded**.

---

## Research Motivation

Traditional zero-shot image classification relies on auxiliary semantic information such as:

- Manually defined attributes
- Word embeddings
- Class-level textual descriptions

For fine-grained categories, these representations may be too generic to distinguish visually similar classes.

This project investigates an LLM-assisted alternative:

1. Generate class-level visual descriptions automatically.
2. Use those descriptions as semantic information for the CLIP text encoder.
3. Compare the resulting classifier with a plain class-name baseline.
4. Analyze both improvements and failure cases.

This is particularly relevant for bird species, where multiple classes can share similar colours, plumage, body shapes, and other visual characteristics.

---

## Objectives

The project is designed around the following objectives:

1. Investigate zero-shot image classification using a vision-language model.
2. Reduce dependence on manually constructed semantic attributes.
3. Generate visual descriptions automatically using an LLM.
4. Evaluate whether generated descriptions improve fine-grained classification.
5. Compare different ways of integrating LLM-generated descriptions.
6. Analyze classes where semantic descriptions help or hurt performance.
7. Build a reusable pipeline for future zero-shot and fine-grained recognition experiments.

---

## Dataset

The current experiment uses a **30-class subset of CUB-200-2011**.

The dataset directory used by the project is:

```text
CUB_200_2011/
```

The experiment evaluates:

- **30 bird classes**
- **445 test images**
- Up to **15 test images per class**

The selected subset contains several visually similar groups, including cormorants and cowbirds, making it useful for studying fine-grained classification failures.

The current implementation does **not** evaluate the full 200-class CUB-200-2011 dataset.

---

## Method

### 1. Class Selection

The pipeline reads the class folders from:

```text
CUB_200_2011/images
```

and uses the first 30 classes after sorting the class-folder names.

The class names are extracted from the CUB folder names and used throughout the baseline, description-generation, and enhanced classification stages.

---

### 2. LLM Description Generation

For each class, the project generates **five short visual descriptions**.

The descriptions focus on characteristics such as:

- Plumage colour
- Plumage pattern
- Bill shape
- Body size
- Distinctive markings

The current description-generation implementation targets:

```text
gemini-3.5-flash-lite
```

The generated descriptions are saved in:

```text
class_descriptions.json
```

The description-generation script also supports:

- Resuming previously completed classes
- Saving progress after each class
- Waiting between requests to reduce rate-limit errors
- Retrying rate-limited requests

The completed experiment produced descriptions for all 30 classes. The generation log confirms:

```text
Done. Saved descriptions for 30 classes to class_descriptions.json
```

---

### 3. Baseline Zero-Shot Classification

The baseline classifier uses **CLIP ViT-B/32** without fine-tuning.

For each class, a simple class-name text prompt is encoded by CLIP. Test images are encoded using the CLIP image encoder, and predictions are made using image-text similarity.

The baseline achieved:

```text
259 / 445 correct
Accuracy: 58.20%
```

The results are stored in:

```text
baseline_results.json
```

---

### 4. LLM-Description Classification

The generated descriptions are encoded using CLIP and used to construct class-level semantic representations.

The experiment tested two important forms of LLM-generated descriptions.

#### Condition A — Raw LLM Descriptions

The generated descriptions were directly passed to the CLIP tokenizer.

This condition achieved:

```text
179 / 445 correct
Accuracy: 40.22%
Change from baseline: -17.98 percentage points
```

This demonstrates that simply adding more descriptive text does **not** necessarily improve zero-shot classification.

---

#### Condition B — Caption-Formatted LLM Descriptions

The same type of generated descriptions was reformatted into a caption-style representation before being encoded.

The formatting anchors the description around the bird class and makes the text more similar to a short image-caption prompt.

This condition achieved:

```text
276 / 445 correct
Accuracy: 62.02%
Change from baseline: +3.82 percentage points
```

This is the main positive result of the experiment.

---

## Results

| Condition | Correct | Accuracy | Change vs. Baseline |
|---|---:|---:|---:|
| Baseline — plain class-name prompt | 259/445 | **58.20%** | — |
| Raw LLM descriptions | 179/445 | **40.22%** | **−17.98 pp** |
| Caption-formatted LLM descriptions | 276/445 | **62.02%** | **+3.82 pp** |

### Main finding

The experiment shows that **the way LLM-generated semantic information is integrated into CLIP matters substantially**.

The raw-description condition reduced accuracy by 17.98 percentage points, while the caption-formatted condition improved accuracy by 3.82 percentage points over the baseline.

Therefore, the result is not simply:

> "LLM descriptions improve classification."

A more accurate conclusion is:

> **LLM-generated semantic descriptions can improve fine-grained zero-shot classification when they are integrated using an appropriate caption-style representation, but poorly formatted descriptions can substantially reduce performance.**

---

## Failure Analysis

### Cormorant Confusion

Cormorants are one of the clearest failure cases.

The generated descriptions for:

- Brandt Cormorant
- Red-faced Cormorant
- Pelagic Cormorant

contain several overlapping visual concepts, including dark plumage, iridescence, and breeding-season feather features.

For closely related species, these generic characteristics can cause the class representations to become too similar.

The descriptions can therefore be **factually reasonable while still being insufficiently discriminative**.

---

### Cowbird Confusion

A similar issue occurs among cowbird species.

For example, the generated descriptions for Bronzed Cowbird and Shiny Cowbird both emphasize dark, glossy, iridescent plumage and red eyes.

These shared characteristics can make the classes difficult for CLIP to separate.

This illustrates an important limitation of automatic semantic generation:

> **An accurate description is not necessarily a discriminative description.**

---

## Examples of Useful Descriptions

Some generated descriptions contain distinctive visual anchors that are more useful for fine-grained classification.

Examples from the generated descriptions include:

### Spotted Catbird

The descriptions repeatedly emphasize:

- Green plumage
- Strong white spotting
- Pale/ivory bill

These features provide more specific information than simply identifying the bird as a green bird.

### Groove-billed Ani

The descriptions emphasize the bird's distinctive bill structure, including vertical ridges.

### Brown Creeper

The generated descriptions identify several visually meaningful features, including:

- Bark-patterned brown upperparts
- Downcurved bill
- White underparts
- Stiff tail feathers used against tree trunks

These examples demonstrate why **specific visual anchors** can be more useful than generic colour or family-level descriptions.

---

## Why the Raw Descriptions Failed

The raw-description result of **40.22%** is an important ablation result.

The experiment suggests that longer or stylistically different language can move the CLIP text representation away from the representation expected for image-caption matching.

This means that adding semantic information alone is insufficient.

The experiment instead supports the following principle:

> **Semantic content and prompt format must both be considered when using LLM-generated descriptions with a vision-language model.**

---

## Implementation Pipeline

The repository is organized into five experimental phases.

### Phase 1 — Dataset Preparation

```text
phase1_dataset.py
```

Prepares and inspects the dataset/class subset used by the experiment.

### Phase 2 — Baseline

```text
phase2_baseline.py
```

Runs the plain class-name CLIP zero-shot classifier and saves:

```text
baseline_results.json
```

### Phase 3 — LLM Description Generation

```text
phase3_descriptions.py
```

Generates five descriptions per class and saves:

```text
class_descriptions.json
```

### Phase 4 — Enhanced Classification

```text
phase4_enhanced.py
```

Uses the generated descriptions with CLIP and saves:

```text
enhanced_results.json
```

### Phase 5 — Analysis

```text
phase5_analysis.py
```

Produces comparative analysis between baseline and enhanced predictions, including:

```text
per_class_comparison.json
confusion_matrices.png
```

---

## Repository Structure

```text
Zero-Shot-Image-Classification/
│
├── .gitignore
├── LICENSE
├── README.md
│
├── phase1_dataset.py
├── phase2_baseline.py
├── phase3_descriptions.py
├── phase4_enhanced.py
├── phase5_analysis.py
│
├── baseline_results.json
├── class_descriptions.json
├── enhanced_results.json
├── per_class_comparison.json
└── confusion_matrices.png
```

The large CUB dataset and local Python virtual environment are excluded from GitHub through `.gitignore`.

---

## Requirements

The project uses Python and the following main packages:

- Python 3.x
- PyTorch
- `open_clip`
- Pillow
- Google GenAI Python SDK (`google-genai`)
- NumPy
- Matplotlib

The Gemini API key is read from the environment variable:

```text
GEMINI_API_KEY
```

For PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key"
```

**Never commit an API key to GitHub.**

---

## Running the Project

Activate the project environment first:

```powershell
.\zsic_env\Scripts\Activate.ps1
```

Run the phases in order:

```powershell
python phase1_dataset.py
python phase2_baseline.py
python phase3_descriptions.py
python phase4_enhanced.py
python phase5_analysis.py
```

If `class_descriptions.json` already contains completed descriptions, Phase 3 resumes from the existing file rather than regenerating completed classes.

---

## Limitations

The current experiment has several limitations:

- Only 30 of the 200 CUB classes are evaluated.
- Only 445 test images are used.
- Only CLIP ViT-B/32 is evaluated.
- CLIP is used without fine-tuning.
- LLM descriptions are generated automatically and are not validated by an ornithology expert.
- Some descriptions contain generic features shared by closely related species.
- The current evaluation does not include a generalized zero-shot learning harmonic mean of seen and unseen accuracy.
- Domain-shift robustness has not yet been systematically evaluated.
- The caption-formatting strategy was selected empirically from the experiment rather than established as a general theoretical rule.

---

## Future Work

The next stages of the internship research can extend the current prototype in several directions.

### Larger datasets

Evaluate the method on:

- Full CUB-200-2011
- AWA2
- SUN

### Better description generation

Investigate:

- Multiple description candidates
- Description ranking
- Contrastive prompts
- Explicit comparison against visually similar classes
- Automatic filtering of generic descriptions
- Description consistency across multiple LLM generations

### Additional vision-language models

Compare CLIP ViT-B/32 with stronger or alternative vision-language models.

### Robustness and domain shift

Test whether LLM-generated descriptions remain useful when the visual test distribution differs from the conditions represented by the generated semantic information.

### Expert validation

Compare LLM-generated descriptions with expert-created attributes to determine whether generated descriptions are both accurate and discriminative.

---

## Reproducibility Note

The numerical results in this README correspond to the experiments recorded in the project files:

- Baseline: **58.20%**
- Raw LLM-description condition: **40.22%**
- Caption-formatted LLM-description condition: **62.02%**

The **40.22% raw-description result is directly recorded in the Phase 4 execution output**:

```text
Correct: 179/445
Enhanced Accuracy: 0.4022 (40.22%)
Baseline (plain prompt) accuracy: 58.20%
Difference: -17.98 percentage points
```

The final **62.02% caption-formatted result should be treated as a separate experimental condition**, not as the raw-description result.

The description-generation script currently targets:

```text
gemini-3.5-flash-lite
```

When reproducing the experiment, record the exact Gemini model used for description generation because changing the generation model can change the generated descriptions and therefore the downstream results.

---

## Reference Paper

The internship project is motivated by the zero-shot image classification literature, including:

> Xu, S., Wang, Y., Zhu, X., et al. "A Survey of Zero-Shot Image Classification: Concepts, Developments, and Challenges." *Tsinghua Science and Technology*, 2026, 31(6), 2792–2821.

DOI:

https://doi.org/10.26599/TST.2025.9010131

---

## Gemini Documentation

For current Gemini API model information and lifecycle information:

- Gemini API models: https://ai.google.dev/gemini-api/docs/models
- Gemini API deprecations: https://ai.google.dev/gemini-api/docs/deprecations

---

## Conclusion

This project provides a proof-of-concept investigation of **LLM-assisted zero-shot fine-grained image classification**.

The experiments demonstrate three important outcomes:

1. A plain class-name CLIP baseline achieved **58.20%** accuracy.
2. Directly using raw LLM descriptions reduced accuracy to **40.22%**.
3. Caption-formatted LLM descriptions increased accuracy to **62.02%**, a **3.82 percentage-point improvement** over the baseline.

The results show that LLM-generated semantic information has potential as an alternative source of auxiliary knowledge for zero-shot classification, but its usefulness depends strongly on whether the generated information is **specific enough to distinguish classes and compatible with the vision-language model's text representation**.

The project therefore contributes not only a working LLM-assisted ZSIC pipeline, but also an experimentally supported guideline:

> **LLM-generated descriptions are most promising when they contain discriminative visual features and are integrated into a CLIP-compatible caption format; generic or poorly formatted descriptions can substantially hurt performance.**
