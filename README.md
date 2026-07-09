# 🐄 CattleCare AI

**AI-Based Cattle Disease Detection System for Ghanaian Farmers and Traders**

Final Year Project — University of Ghana, Department of Computer Science (2025/2026)

CattleCare AI helps smallholder farmers and cattle traders get an early, AI-assisted read on three common conditions — **Foot-and-Mouth Disease (FMD)**, **Lumpy Skin Disease (LSD)**, and **Healthy** — from a photo, a description of observed symptoms, or both. It's built around a Gemma 3 vision-language model fine-tuned specifically on Ghanaian cattle disease presentations, served through a Streamlit interface.

> ⚠️ **Disclaimer:** This is a decision-support tool, not a replacement for veterinary diagnosis. Always consult a qualified veterinarian for confirmation and treatment.

---

## Features

- **Three ways to get a diagnosis:**
  - 📷 Upload a photo of the animal
  - 📝 Describe symptoms (checkboxes + free text)
  - 🔗 Both together, for the most accurate result
- **Input validation before diagnosis** — a lightweight classification layer checks that an uploaded photo actually shows cattle and that a symptom description is real and observable, rather than letting the model guess at chitchat, junk input, or irrelevant images.
- **Transparent source attribution** — the UI shows exactly which inputs (image / symptom description) were actually used for a given diagnosis.
- **Symmetric fallback logic** — a bad photo won't derail a real symptom description, and vice versa; the app automatically falls back to whichever input is actually usable.
- **Model comparison mode** — base model vs. fine-tuned model, side by side, on the same input.
- **Disease reference guide** — a built-in tab covering symptoms, spread, and recommended actions for each condition.
- **Emergency contacts** — direct numbers for the Ghana Veterinary Service and Animal Research Institute surfaced alongside every diagnosis.

## How It Works

```
                     ┌─────────────────────┐
   Photo + Symptoms  │   Input Validation   │   gpt-4o-mini classifiers check:
   ─────────────────▶│  (chitchat / image   │   "is this a real symptom?"
                      │   relevance checks)  │   "does this image show cattle?"
                      └──────────┬───────────┘
                                 │ valid input(s)
                                 ▼
                      ┌─────────────────────┐
                      │  Fine-tuned Gemma 3  │   Hosted on a Hugging Face
                      │   (via HF Inference  │   Inference Endpoint,
                      │      Endpoint)       │   OpenAI-compatible API
                      └──────────┬───────────┘
                                 ▼
                      Structured diagnosis: disease name, confidence,
                      supporting evidence, recommended actions
```

## Model

| | |
|---|---|
| **Base model** | `google/gemma-3-4b-it` (multimodal — text + vision) |
| **Fine-tuning method** | QLoRA (4-bit), r=16, α=32, all attention + MLP modules |
| **Training data** | 900 images (300 per class: FMD, LSD, healthy) + synthetic symptom-text and combined examples |
| **Epochs** | 3 |
| **Training hardware** | Rented GPU (Runpod) |
| **Deployment** | Merged adapter pushed to [Hugging Face Hub](https://huggingface.co), served via a Hugging Face Inference Endpoint |

### Evaluation

Accuracy on a held-out sample (30 examples per class per mode), evaluated against the deployed endpoint:

| Mode | Accuracy |
|---|---|
| Image only | 95.6% |
| Symptoms only | 94.4% |
| Image + Symptoms | **100%** |

Full per-class precision/recall and confusion matrices are in [`evaluate_all_modes.py`](./evaluate_all_modes.py) and its generated outputs.

**Known limitations** (see below) — worth reading before citing these numbers as generalization performance.

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **Model serving:** Hugging Face Inference Endpoints (vLLM, OpenAI-compatible API)
- **Fine-tuning:** 🤗 `transformers`, `peft`, `trl`, `bitsandbytes`
- **Validation/comparison model:** OpenAI `gpt-4o-mini`
- **Evaluation:** `scikit-learn`, `matplotlib`

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key (used for input validation and base-model comparison)
- A Hugging Face account with a deployed Inference Endpoint running the fine-tuned model (or use your own — see [Training](#model) above)

### Installation

```bash
git clone https://github.com/<your-username>/cattle-care-ai.git
cd cattle-care-ai
uv sync   # or: pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
HF_ENDPOINT_URL=https://your-endpoint.endpoints.huggingface.cloud/v1
HF_TOKEN=your-huggingface-token
GEMMA_MODEL_NAME=your-username/gemma3-cattle-disease-ghana
```

### Run

```bash
streamlit run app.py
```

## Project Structure

```
cattle-care-ai/
├── app.py                      # Streamlit application
├── evaluate_all_modes.py       # Evaluation script (confusion matrices, accuracy)
├── requirements.txt
├── .env                        # Not committed — see Configuration above
└── Cows datasets/               # Not committed — see Dataset below
    ├── foot-and-mouth/
    ├── lumpy/
    └── healthy/
```

## Dataset

Images sourced and merged from two public Kaggle datasets covering cattle disease imagery (foot-and-mouth, lumpy skin disease) and healthy cattle. Not included in this repository due to size — see the original sources for access.

## Limitations

Documented honestly here rather than left implicit, since they matter for interpreting the evaluation results above:

- The original train/validation split (during fine-tuning) was not seeded, so it can't be exactly reproduced. Evaluation images used post-training may partially overlap with training data, which could inflate the reported image-based accuracy relative to true generalization.
- Symptom-text evaluation draws from the same phrase pools used to generate training data. It reflects strong performance on trained phrasing patterns more than open-ended generalization to arbitrary farmer wording.
- The `healthy` class shows the weakest recall in single-input modes, meaning some genuinely healthy animals are misclassified — this is the most valuable direction for future data collection.
- Vision-based diagnosis from a single photo is inherently harder than text-pattern matching; performance is expected to vary more with photo quality, lighting, and angle than symptom-only mode does with phrasing.

## Acknowledgments

- Cattle disease image datasets: Kaggle contributors
- Base model: Google DeepMind's Gemma 3
- Ghana Veterinary Service and Animal Research Institute contact information used for in-app emergency guidance

## License

*"All rights reserved" - this is a submitted coursework not intended for reuse.)*