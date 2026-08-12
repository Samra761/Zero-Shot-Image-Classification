# Zero-Shot Fine-Grained Bird Classification with LLM-Generated Semantic Descriptions

## 1. Introduction

Zero-shot image classification (ZSIC) allows a model to recognize object categories it has never seen labeled examples of, by transferring knowledge from auxiliary semantic information associated with both seen and unseen classes. Existing ZSIC methods rely heavily on manually curated or embedding-based auxiliary information, which is expensive to construct and often insufficient to distinguish visually similar fine-grained categories.

This project tests whether an LLM can automatically generate class-discriminative semantic descriptions to replace manual attribute annotation, and measures when this helps versus hurts fine-grained zero-shot classification accuracy.

## 2. Method

The pipeline has three stages:

1. **Description generation** — for each class name, Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) is prompted to produce a short, visually grounded description focused on discriminative features (plumage colour and pattern, bill shape, size, and markings).

   Prompt template:
   ```
   "Describe the visual appearance of a [CLASS NAME] bird in one or two
   sentences. Focus only on distinguishing visual features: colour,
   pattern, bill shape, and size. Do not use flowery or subjective
   language. Do not mention the bird's name in the description."
   ```

2. **Prompt formatting** — raw LLM output could not be fed directly into CLIP's text encoder without a drop in accuracy. Descriptions are reformatted into a fixed template before encoding:
   ```
   "a photo of a [CLASS NAME], [DESCRIPTION]"
   ```

3. **Classification** — each formatted description is encoded with the CLIP (ViT-B/32) text encoder to produce a class embedding. Each test image is encoded with the CLIP image encoder and predicted as the class whose text embedding has the highest cosine similarity to the image embedding.

## 3. Experimental Setup

- **Dataset**: 30-class fine-grained subset of CUB-200-2011 (Caltech-UCSD Birds), chosen to include visually similar species (three cormorants, five blackbird/cowbird species, two albatrosses). 445 test images.
- **Models**: Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) for semantic-description generation and CLIP ViT-B/32 for zero-shot classification. CLIP is used zero-shot with no fine-tuning.
- **Conditions**: (1) baseline plain class-name prompt, (2) enhanced formatted LLM description, (3) ablation with raw unformatted LLM description.

## 4. Results

| Configuration | Accuracy | Change vs baseline |
|---|---|---|
| Baseline (plain class-name prompt) | 58.20% (259/445) | — |
| Enhanced (formatted LLM description) | 62.02% (276/445) | +3.82 pp |
| Ablation: raw flowery LLM prompt | 40.22% | -17.98 pp |

See `figures/confusion_matrices.png` for the baseline vs. enhanced confusion matrices across all 30 classes.

**Cormorants are the clearest failure case.** Red-faced Cormorant dropped from 26.7% to 0%, and Brandt Cormorant dropped from 40% to 13.3%, both collapsing into Pelagic Cormorant predictions. All three cormorant descriptions independently emphasise "iridescent," "glossy," and "dark plumage," plus breeding-season white plumes — generically accurate for the genus but not discriminative within it. Bronzed Cowbird shows the same pattern, dropping from 13.3% to 0% and collapsing into Shiny Cowbird, with both cowbird descriptions emphasising "iridescent," "glossy blue-black/bronze," and "red eye" without a strong distinguishing anchor.

**Improved classes anchor on rare, specific features instead.** Spotted Catbird's description repeatedly anchors on "white spots" and "ivory bill" (+33.3 pp). Groove-billed Ani's description emphasises "vertical ridges on the bill" and a "prehistoric" shaggy throat (+26.7 pp). Chuck-will's-widow (+26.7 pp), Shiny Cowbird (+20.0 pp), and Laysan Albatross (+20.0 pp) follow the same pattern of a distinctive trait not implied by the class name.

The raw-prompt ablation failed more broadly: unformatted LLM output tends toward longer, subjective phrasing that shifts the text embedding away from the short, object-focused caption style CLIP was trained on. This affected nearly all classes, not just visually similar ones.

## 6. Guidelines: When LLM Descriptions Help vs. Hurt

**Help when:**
- The class has a genuinely distinctive visual feature not implied by its name.
- The description is formatted to preserve the class name and CLIP's expected caption style.
- The description stays short and feature-focused.

**Hurt when:**
- Two or more classes are near-duplicates and their LLM descriptions converge on the same generic, family-level features (for example "iridescent," "glossy," "dark plumage" across all cormorant species).
- The description is used in raw, unformatted form.
- The LLM has limited visual knowledge of the specific class and produces an accurate but non-discriminative description.

## 7. Limitations

- Evaluated on a 30-class subset of CUB-200, not the full 200-class benchmark.
- Only one CLIP backbone (ViT-B/32) tested.
- LLM descriptions were generated once per class and were not validated against an ornithologist. For reproducible future runs, use the pinned Gemini model ID `gemini-3.5-flash-lite`.
- The formatting fix was found empirically, not derived from a general principle.
- No harmonic-mean seen/unseen evaluation, since this setup is fully zero-shot with no seen-class training split.

## 8. Conclusion

Formatted LLM-generated descriptions improved zero-shot classification accuracy on a fine-grained 30-class bird subset by 3.82 percentage points over a plain class-name baseline, but the benefit was not uniform: classes with distinctive traits improved substantially, while near-duplicate species pairs worsened. Prompt structure proved at least as important as the presence of descriptive content. LLM-generated auxiliary information is a viable, low-cost replacement for manual attribute annotation in zero-shot classification, provided it is formatted correctly and applied selectively. The description-generation implementation should use the current stable Gemini 3.5 Flash-Lite model (`gemini-3.5-flash-lite`). The accuracy figures above are retained from the supplied experiment and should not be interpreted as Gemini 3.5 Flash-Lite results unless the experiment is rerun with that model.

## 9. Gemini Model/API Update

The description-generation stage is now targeted at **Gemini 3.5 Flash-Lite** using the model ID:

```text
gemini-3.5-flash-lite
```

This replaces the earlier `gemini-2.5-flash` target for new runs. For high-volume generation, the implementation should also respect the provider's current rate limits, retry transient `429`/resource-exhaustion responses, save progress after each completed class, and resume from an existing descriptions JSON file rather than regenerating completed classes.

**Reproducibility note:** the numerical results in Section 4 come from the supplied experiment and are preserved unchanged. They should only be attributed specifically to Gemini 3.5 Flash-Lite after rerunning the description-generation and evaluation pipeline with `gemini-3.5-flash-lite`.

Official references:
- Google Gemini model documentation: https://ai.google.dev/gemini-api/docs/models
- Google Gemini deprecations: https://ai.google.dev/gemini-api/docs/deprecations

## Repository Structure

```
zsic-llm-descriptions/
├── README.md
├── data/
│   └── descriptions.json      # LLM-generated descriptions per class (add your file)
├── scripts/                   # or notebook(s) (add your file)
└── figures/
    └── confusion_matrices.png
```

## Requirements

- Python 3.x
- `google-genai` (Google GenAI Python SDK)
- `clip` (OpenAI CLIP) or `open_clip`
- `torch`
