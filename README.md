# 🐄 CattleCare AI

**AI-Based Cattle Disease Detection System for Ghanaian Farmers and Traders**

Final Year Project — University of Ghana, Department of Computer Science (2025/2026)

CattleCare AI helps smallholder farmers and cattle traders get an early, AI-assisted read on three common conditions — **Foot-and-Mouth Disease (FMD)**, **Lumpy Skin Disease (LSD)**, and **Healthy** — from a photo, a description of observed symptoms, or both. It is built around a Gemma 3 vision-language model fine-tuned specifically on Ghanaian cattle disease presentations, served through a Streamlit interface.

> ⚠️ **Disclaimer:** This is a decision-support tool, not a replacement for veterinary diagnosis. Always consult a qualified veterinarian for confirmation and treatment.

---

## Features

- **Three ways to get a diagnosis:**
  - 📷 Upload a photo of the animal
  - 📝 Describe symptoms (checkboxes + free text)
  - 🔗 Both together, for the most accurate result
- **Input validation before diagnosis** — a lightweight classification layer checks that an uploaded photo actually shows cattle and that a symptom description is real and observable, rather than letting the model guess at chitchat, junk input, or irrelevant images.
- **Transparent source attribution** — the UI shows exactly which inputs (image / symptom description) were actually used for a given diagnosis.
- **Symmetric fallback logic** — a bad photo will not derail a real symptom description, and vice versa; the app automatically falls back to whichever input is actually usable.
- **Model comparison mode** — base model vs. fine-tuned model, side by side, on the same input.
- **Disease reference guide** — a built-in tab covering symptoms, spread, and recommended actions for each condition.
- **Emergency contacts** — direct numbers for the Ghana Veterinary Service and Animal Research Institute surfaced alongside every diagnosis.

---

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

---

## Model

| | |
|---|---|
| **Base model** | `google/gemma-3-4b-it` (multimodal — text + vision) |
| **Fine-tuning method** | QLoRA (4-bit), r=16, α=32, all attention + MLP modules |
| **Training data** | 900 images (300 per class: FMD, LSD, Healthy) + Ghana Veterinary Service symptom-text examples and combined multimodal examples |
| **Epochs** | 3 |
| **Training hardware** | Rented GPU (Runpod RTX 4090, 24GB VRAM) |
| **Deployment** | Merged adapter pushed to [Hugging Face Hub](https://huggingface.co), served via a Hugging Face Inference Endpoint |

### Evaluation

Accuracy on a held-out sample (30 examples per class per mode), evaluated against the deployed endpoint:

| Mode | Accuracy |
|---|---|
| Image only | 95.6% |
| Symptoms only | 94.4% |
| Image + Symptoms | **100%** |

Full per-class precision/recall and confusion matrices are in [`evaluate_all_modes.py`](./evaluate_all_modes.py) and its generated outputs.

**Known limitations** (see below) — worth reading before citing these numbers as generalisation performance.

---

## Dataset

### Image Data

Cattle disease images were sourced and merged from the following credible repositories:

- **Ghana Veterinary Service (GVS)** — disease presentation documentation and field case imagery for Foot-and-Mouth Disease and Lumpy Skin Disease specific to Ghanaian cattle breeds and environmental conditions.
- **CSIR Animal Research Institute (CSIR-ARI), Achimota, Ghana** — archival disease imagery and clinical records contributed through veterinary research collaboration.
- **International Livestock Research Institute (ILRI)** — peer-reviewed livestock disease image data covering West African cattle disease presentations.
- **FAO EMPRES-i Disease Database** — documented outbreak imagery and case reports for FMD and LSD in West Africa.
- **Supplementary public repositories** — additional images sourced from two Kaggle datasets to bolster class volume where institutional data was limited. These images were screened for visual quality and clinical relevance before inclusion.

The combined image dataset comprises **4,268 images** across three classes:

| Class | Images |
|---|---|
| Foot-and-Mouth Disease (FMD) | 746 |
| Lumpy Skin Disease (LSD) | 1,531 |
| Healthy Cattle | 1,991 |
| **Total** | **4,268** |

Images are not included in this repository due to size. Contact the project author for access details.

### Symptom Text Data

The symptom-to-disease training examples were constructed from veterinary literature and institutional guidance sourced from:

- **Ghana Veterinary Service (GVS)** — official disease management protocols, symptom checklists, and farmer advisory materials for FMD and LSD in Ghana.
- **CSIR Animal Research Institute (CSIR-ARI), Achimota** — clinical symptom profiles for cattle diseases prevalent in Ghana, including seasonal outbreak patterns and breed-specific presentations.
- **Food and Agriculture Organization of the United Nations (FAO)** — FAO Animal Health Manuals and EMPRES-i disease factsheets for Foot-and-Mouth Disease and Lumpy Skin Disease.
- **International Livestock Research Institute (ILRI)** — peer-reviewed research on livestock disease symptom patterns in sub-Saharan Africa.

Each training example pairs a farmer-style symptom description (written to reflect the natural language patterns of Ghanaian farmers) with a structured veterinary diagnostic response including disease identification, confidence level, clinical reasoning, and numbered recommendations aligned with Ghana Veterinary Service protocols. A total of **15 curated symptom examples** spanning multiple clinical presentations and severity levels were used in fine-tuning, covering FMD (5 variations), LSD (5 variations), and Healthy cattle (3 descriptions including one borderline monitoring case).

---

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **Model serving:** Hugging Face Inference Endpoints (vLLM, OpenAI-compatible API)
- **Fine-tuning:** 🤗 `transformers`, `peft`, `trl`, `bitsandbytes`
- **Validation / comparison model:** OpenAI `gpt-4o-mini`
- **Evaluation:** `scikit-learn`, `matplotlib`

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key (used for input validation and base-model comparison)
- A Hugging Face account with a deployed Inference Endpoint running the fine-tuned model

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

---

## Project Structure

```
cattle-care-ai/
├── app.py                      # Streamlit application
├── evaluate_all_modes.py       # Evaluation script (confusion matrices, accuracy)
├── requirements.txt
├── .env                        # Not committed — see Configuration above
└── Cows datasets/              # Not committed — see Dataset section above
    ├── foot-and-mouth/
    ├── lumpy/
    └── healthy/
```

---

## Limitations

Documented honestly rather than left implicit, since they matter for interpreting the evaluation results above:

- The original train/validation split during fine-tuning was not seeded, so it cannot be exactly reproduced. Evaluation images used post-training may partially overlap with training data, which could inflate the reported image-based accuracy relative to true generalisation performance.
- Symptom-text evaluation draws from the same phrase pools and clinical patterns used to generate training examples. Performance reflects strong learning on trained phrasing more than open-ended generalisation to arbitrary farmer wording.
- The `healthy` class shows the weakest recall in single-input modes, meaning some genuinely healthy animals are misclassified as diseased. This is the most valuable direction for future data collection.
- Vision-based diagnosis from a single photo is inherently harder than text-pattern matching. Performance is expected to vary with photo quality, lighting, distance, and camera angle more than symptom-only mode varies with phrasing.
- The current system supports English-language input only. Support for Ghanaian local languages (Twi, Dagbani, Hausa) is identified as a priority for future work.

---

## Acknowledgments

- **Ghana Veterinary Service (GVS)** — disease protocols, symptom data, and emergency contact information integrated into the system.
- **CSIR Animal Research Institute, Achimota** — clinical disease data and veterinary expertise informing the training dataset.
- **International Livestock Research Institute (ILRI)** and **FAO** — livestock disease research and documentation.
- **Google DeepMind** — Gemma 3 base model.
- **Kaggle contributors** — supplementary cattle disease image data.

---

## Citation

If you use CattleCare AI or its dataset methodology in your research, please cite:

```
Sulemana, A. (2026). CattleCare AI: AI-Based Cattle Disease Detection System
for Ghanaian Smallholder Farmers Using Fine-tuned Multimodal Large Language Models.
BSc Thesis, Department of Computer Science, University of Ghana.
```

---

## License

All rights reserved. This project is submitted as coursework for the University of Ghana Department of Computer Science and is not licensed for reuse without written permission from the author.
