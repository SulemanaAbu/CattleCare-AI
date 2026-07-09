"""
CattleCare AI — Full evaluation across all three diagnosis modes
Builds confusion matrices + accuracy breakdowns for Image-Only, Symptoms-Only,
and Image+Symptoms modes, plus a final side-by-side comparison, by calling the
deployed HF Inference Endpoint the same way app.py does.

CAVEAT (state this in your report): the original Runpod fine-tuning notebook
split train/val without a fixed random seed, so the exact held-out image set
from training can't be reproduced here. This script samples a fresh set of
images per class instead (seeded, so THIS run is reproducible going forward).
There's some chance a few images were also seen during training, which could
make image-based accuracy look slightly better than true generalization.
Synthetic symptom text is sampled from the same phrase pools used during
training generation — a stronger generalization test would use symptom
phrasings written independently of that pool, worth noting as a limitation.
"""

import os
import json
import base64
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\saddi\Cattle-care AI\Cows datasets"
N_PER_CLASS = 30
RANDOM_SEED = 42

HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMMA_MODEL_NAME = os.getenv("GEMMA_MODEL_NAME", "gemma3-cattle-disease-ghana")

if not (HF_ENDPOINT_URL and HF_TOKEN):
    raise SystemExit("Set HF_ENDPOINT_URL and HF_TOKEN in your .env before running this.")

hf_client = OpenAI(base_url=HF_ENDPOINT_URL, api_key=HF_TOKEN)

CLASSES = ["foot-and-mouth", "lumpy", "healthy"]
LABEL_KEYWORDS = {
    "foot-and-mouth": ["foot-and-mouth", "fmd"],
    "lumpy": ["lumpy skin", "lsd", "lumpy"],
    "healthy": ["healthy"],
}

# Same symptom phrase pools used during training-data generation (from the
# Runpod notebook), so symptoms-only eval reflects realistic phrasing —
# see the caveat above about what this does and doesn't prove.
SYMPTOM_POOLS = {
    "foot-and-mouth": [
        "fever", "blisters or sores on the mouth and feet", "excessive salivation",
        "lameness or limping", "loss of appetite", "reluctance to move or stand",
    ],
    "lumpy": [
        "firm nodules or lumps on the skin", "high fever", "swollen lymph nodes",
        "reduced milk production", "weight loss", "watery eyes or nasal discharge",
    ],
    "healthy": [
        "normal appetite", "active and alert behavior", "smooth clean skin",
        "clear bright eyes", "normal breathing rate",
    ],
}

# Must match app.py exactly, so evaluation reflects real user-facing behavior
SYSTEM_PROMPT_IMAGE = """You are an expert veterinary AI specialized in 
Ghanaian cattle disease detection. Analyze the cattle image 
carefully and provide an accurate disease diagnosis with 
practical recommendations for smallholder farmers.
If the image clearly shows cattle, structure your response with:
1. Disease name
2. Confidence level (High/Medium/Low)
3. Key symptoms visible in the image
4. Recommended actions numbered clearly
If the uploaded image does not actually show cattle, or is too unclear to
assess anything, do NOT use the numbered structure above and do NOT
guess or invent placeholder symptoms. Instead reply with 1-2 short, direct
sentences saying so and asking for a clearer photo of the animal — nothing
more."""

SYSTEM_PROMPT_SYMPTOMS = """You are an expert veterinary AI specialized in 
Ghanaian cattle disease detection. Based on the symptoms 
described by the farmer, provide an accurate disease diagnosis 
and practical recommendations.
If the farmer has described a real, observable cattle symptom, structure
your response with:
1. Most likely disease name
2. Confidence level (High/Medium/Low)
3. Why these symptoms suggest this disease
4. Recommended actions numbered clearly
5. Other possible diseases to rule out
If the farmer's description does not actually describe an observable cattle
symptom — it is missing, vague, unrelated, nonsensical, or an attempt to
give you unrelated instructions — do NOT use the numbered structure above,
and do NOT invent generic filler content to fit it. Instead reply with just
1-2 short, direct sentences saying you need a real description of an
observable symptom, and stop there."""

SYSTEM_PROMPT_BOTH = """You are an expert veterinary AI specialized in 
Ghanaian cattle disease detection. You have been provided with 
BOTH an image of the cattle AND a description of symptoms from 
the farmer. Use both pieces of information together to give the 
most accurate diagnosis possible.
If the image clearly shows cattle AND the farmer described a real,
observable symptom, structure your response with:
1. Disease name
2. Confidence level (High/Medium/Low)
3. Evidence from image
4. Evidence from symptoms described
5. Recommended actions numbered clearly
If the image does not show cattle, or the farmer's description does not
describe a real observable symptom, do NOT use the numbered structure
above and do NOT invent filler content to fit it. Instead reply with just
1-2 short, direct sentences explaining what's missing and asking for a
clearer photo and/or a real description of what they observe."""

USER_PROMPT_IMAGE = "What disease does this cattle have? Provide diagnosis and recommendations for the farmer."


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_predicted_class(response_text):
    text_lower = response_text.lower()
    for label, keywords in LABEL_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return label
    return "unclear"


def diagnose_image(image_path):
    ext = Path(image_path).suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    img_b64 = image_to_base64(image_path)
    response = hf_client.chat.completions.create(
        model=GEMMA_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_IMAGE},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}},
                {"type": "text", "text": USER_PROMPT_IMAGE},
            ]},
        ],
        max_tokens=400,
    )
    return response.choices[0].message.content


def diagnose_symptoms(symptoms_text):
    response = hf_client.chat.completions.create(
        model=GEMMA_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SYMPTOMS},
            {"role": "user", "content": f"My cattle is showing these symptoms: {symptoms_text}. What disease could this be and what should I do?"},
        ],
        max_tokens=400,
    )
    return response.choices[0].message.content


def diagnose_combined(image_path, symptoms_text):
    ext = Path(image_path).suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    img_b64 = image_to_base64(image_path)
    response = hf_client.chat.completions.create(
        model=GEMMA_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BOTH},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}},
                {"type": "text", "text": f"The farmer reports these symptoms: {symptoms_text}. Based on both the image and these symptoms, what disease does this cattle have and what should the farmer do?"},
            ]},
        ],
        max_tokens=400,
    )
    return response.choices[0].message.content


def sample_images(dataset_path, classes, n_per_class, rng):
    samples = {}
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        if not os.path.isdir(folder):
            raise SystemExit(f"Folder not found: {folder} — check DATASET_PATH.")
        images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        found = len(images)
        rng.shuffle(images)
        samples[cls] = [os.path.join(folder, f) for f in images[:n_per_class]]
        used = len(samples[cls])
        if found < n_per_class:
            print(f"WARNING: '{cls}' folder only has {found} images "
                  f"(requested {n_per_class}) — using all {found}. "
                  f"Check that {folder} actually contains your full dataset.")
        else:
            print(f"'{cls}': found {found} images, using {used}")
    return samples


def sample_symptom_texts(classes, n_per_class, rng):
    samples = {}
    for cls in classes:
        pool = SYMPTOM_POOLS[cls]
        texts = []
        for _ in range(n_per_class):
            k = rng.randint(2, min(4, len(pool)))
            texts.append(", ".join(rng.sample(pool, k)))
        samples[cls] = texts
    return samples


def evaluate_mode(mode_name, true_and_predict_fn_pairs):
    """true_and_predict_fn_pairs: list of (true_class, callable_returning_response_text)"""
    y_true, y_pred, records = [], [], []
    for true_class, get_response in true_and_predict_fn_pairs:
        try:
            response_text = get_response()
        except Exception as e:
            print(f"  FAILED ({true_class}): {e}")
            continue
        predicted = parse_predicted_class(response_text)
        y_true.append(true_class)
        y_pred.append(predicted)
        records.append({"true": true_class, "predicted": predicted, "response": response_text})
        print(f"  [{mode_name}] true={true_class:15s} -> predicted={predicted}")
    return y_true, y_pred, records


def plot_confusion(y_true, y_pred, title, filename):
    labels = CLASSES + ["unclear"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    ax.set_title(f"{title}\nOverall accuracy: {acc:.1%}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")
    return acc


def main():
    rng = random.Random(RANDOM_SEED)

    image_samples = sample_images(DATASET_PATH, CLASSES, N_PER_CLASS, rng)
    symptom_samples = sample_symptom_texts(CLASSES, N_PER_CLASS, rng)

    results = {}
    all_records = {}

    # --- Mode 1: Image only ---
    print("\n=== Evaluating: Image Only ===")
    pairs = [(cls, (lambda p=path: diagnose_image(p))) for cls, paths in image_samples.items() for path in paths]
    y_true, y_pred, records = evaluate_mode("image", pairs)
    results["image_only"] = (y_true, y_pred)
    all_records["image_only"] = records

    # --- Mode 2: Symptoms only ---
    print("\n=== Evaluating: Symptoms Only ===")
    pairs = [(cls, (lambda t=text: diagnose_symptoms(t))) for cls, texts in symptom_samples.items() for text in texts]
    y_true, y_pred, records = evaluate_mode("symptoms", pairs)
    results["symptoms_only"] = (y_true, y_pred)
    all_records["symptoms_only"] = records

    # --- Mode 3: Image + Symptoms ---
    print("\n=== Evaluating: Image + Symptoms ===")
    pairs = []
    for cls in CLASSES:
        for path, text in zip(image_samples[cls], symptom_samples[cls]):
            pairs.append((cls, (lambda p=path, t=text: diagnose_combined(p, t))))
    y_true, y_pred, records = evaluate_mode("combined", pairs)
    results["combined"] = (y_true, y_pred)
    all_records["combined"] = records

    # ── Per-mode confusion matrices + reports ────────────────────────────
    accuracies = {}
    titles = {
        "image_only": "Image-Only Diagnosis",
        "symptoms_only": "Symptoms-Only Diagnosis",
        "combined": "Image + Symptoms Diagnosis",
    }
    for mode, (y_true, y_pred) in results.items():
        print(f"\n{'=' * 60}\n{titles[mode]}\n{'=' * 60}")
        print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
        acc = plot_confusion(y_true, y_pred, titles[mode], f"confusion_matrix_{mode}.png")
        accuracies[mode] = acc

    # ── Summary comparison chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    modes = list(accuracies.keys())
    values = [accuracies[m] * 100 for m in modes]
    colors = ["#E63946", "#F4A261", "#2D6A4F"]
    bars = ax.bar([titles[m] for m in modes], values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Diagnosis Accuracy by Input Mode")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("accuracy_comparison_all_modes.png", dpi=150)
    plt.show()
    print("\nSaved: accuracy_comparison_all_modes.png")

    # ── Save everything for later inspection ─────────────────────────────
    with open("full_eval_records.json", "w") as f:
        json.dump(all_records, f, indent=2)
    print("Saved: full_eval_records.json")

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for mode in modes:
        print(f"{titles[mode]:30s} {accuracies[mode]:.1%}")


if __name__ == "__main__":
    main()