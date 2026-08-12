import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# ---- Load results from both phases ----
with open("baseline_results.json", "r") as f:
    baseline = json.load(f)

with open("enhanced_results.json", "r") as f:
    enhanced = json.load(f)

# ---- Get sorted class list (consistent ordering for both matrices) ----
all_true_labels = sorted(set(r["true"] for r in baseline["results"]))

# ---- Build confusion matrices ----
def build_confusion_matrix(results, labels):
    y_true = [r["true"] for r in results]
    y_pred = [r["pred"] for r in results]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm

cm_baseline = build_confusion_matrix(baseline["results"], all_true_labels)
cm_enhanced = build_confusion_matrix(enhanced["results"], all_true_labels)

# ---- Plot both confusion matrices side by side ----
fig, axes = plt.subplots(1, 2, figsize=(24, 11))

for ax, cm, title in [(axes[0], cm_baseline, "Baseline (plain prompt)"),
                        (axes[1], cm_enhanced, "Enhanced (LLM description, formatted)")]:
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(len(all_true_labels)))
    ax.set_yticks(range(len(all_true_labels)))
    ax.set_xticklabels(all_true_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(all_true_labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
print("Saved confusion_matrices.png")

# ---- Identify most-confused class pairs for each method ----
def top_confusions(cm, labels, n=10):
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i][j] > 0:
                confusions.append((labels[i], labels[j], cm[i][j]))
    confusions.sort(key=lambda x: -x[2])
    return confusions[:n]

print("\n=== TOP CONFUSIONS: BASELINE ===")
for true_c, pred_c, count in top_confusions(cm_baseline, all_true_labels):
    print(f"  True: {true_c:30s} -> Predicted: {pred_c:30s} ({count} times)")

print("\n=== TOP CONFUSIONS: ENHANCED ===")
for true_c, pred_c, count in top_confusions(cm_enhanced, all_true_labels):
    print(f"  True: {true_c:30s} -> Predicted: {pred_c:30s} ({count} times)")

# ---- Per-class accuracy comparison (which classes improved/worsened most) ----
def per_class_accuracy(results, labels):
    class_correct = {c: 0 for c in labels}
    class_total = {c: 0 for c in labels}
    for r in results:
        class_total[r["true"]] += 1
        if r["correct"]:
            class_correct[r["true"]] += 1
    return {c: (class_correct[c] / class_total[c] if class_total[c] > 0 else 0) for c in labels}

baseline_per_class = per_class_accuracy(baseline["results"], all_true_labels)
enhanced_per_class = per_class_accuracy(enhanced["results"], all_true_labels)

print("\n=== PER-CLASS ACCURACY CHANGE (Enhanced - Baseline) ===")
changes = []
for c in all_true_labels:
    diff = enhanced_per_class[c] - baseline_per_class[c]
    changes.append((c, baseline_per_class[c], enhanced_per_class[c], diff))

changes.sort(key=lambda x: x[3])  # sort by diff, worst first

print("\nMost WORSENED classes:")
for c, b, e, d in changes[:5]:
    print(f"  {c:30s} baseline={b*100:.1f}%  enhanced={e*100:.1f}%  change={d*100:+.1f}pp")

print("\nMost IMPROVED classes:")
for c, b, e, d in changes[-5:][::-1]:
    print(f"  {c:30s} baseline={b*100:.1f}%  enhanced={e*100:.1f}%  change={d*100:+.1f}pp")

# ---- Save full per-class comparison to a file for the report ----
with open("per_class_comparison.json", "w") as f:
    json.dump({c: {"baseline": baseline_per_class[c], "enhanced": enhanced_per_class[c]}
               for c in all_true_labels}, f, indent=2)

print("\nSaved per_class_comparison.json")
print("\nDone. Review confusion_matrices.png and the printed output above for your report's failure analysis section.")