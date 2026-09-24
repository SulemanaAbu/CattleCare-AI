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
- **Model toggle**: switch between the fine-tuned model and the original Gemma 3 to compare their answers on the same input.
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
| **Base model** | `google/gemma-3-4b-it` (multimodal: text + vision) |
| **Fine-tuning method** | LoRA with 4-bit quantisation (QLoRA-style), r=16, α=32, attention + MLP modules |
| **Training data** | 900 images (300 per class: FMD, LSD, Healthy) + veterinary-grounded symptom-text and combined multimodal examples |
| **Epochs** | 3 |
| **Training hardware** | Rented GPU (Runpod) |
| **Deployment** | Merged model on a Hugging Face Inference Endpoint (vLLM, Nvidia L4); the original `google/gemma-3-4b-it` runs on a second, identical endpoint as a baseline |

### Evaluation: fine-tuned vs original Gemma 3

Both models answered the same 270 test cases (90 per mode) with identical prompts, temperature 0, on identical hardware.

| Mode | Original Gemma 3 | Fine-tuned | McNemar p |
|---|---|---|---|
| Image only | 32.2% | 95.6% | 1.4 × 10⁻¹⁷ |
| Symptoms only | 43.3% | 100% | 8.9 × 10⁻¹⁶ |
| Image + Symptoms | 57.8% | 100% | 7.3 × 10⁻¹² |
| **All modes** | **44.4%** | **98.5%** | **2.2 × 10⁻⁴⁴** |

There was no case in which the original model was correct and the fine-tuned model was wrong. Evaluation script: [`evaluate_all_models.py`](./evaluate_all_models.py).

**Read the limitations below before citing these numbers as real-world performance.**

---

## Dataset

### Images

The images come from two public Kaggle collections, merged into one folder per class:

- [Cattle Diseases Datasets](https://www.kaggle.com/datasets/devang03mgr/cattle-diseases-datasets) (devang03mgr): foot-and-mouth, lumpy, healthy
- [Lumpy Skin Images Dataset](https://www.kaggle.com/datasets/warcoder/lumpy-skin-images-dataset) (warcoder): lumpy skin, normal skin

| Class | Images available | Used for fine-tuning |
|---|---|---|
| Foot-and-Mouth Disease (FMD) | 746 | 300 |
| Lumpy Skin Disease (LSD) | 1,531 | 300 |
| Healthy | 1,991 | 300 |

The images were **not** collected in Ghana. Images are not included in this repository; download them from the links above.

### Symptom text

Symptom-to-diagnosis training examples were written from published veterinary descriptions of FMD and LSD (for example the MSD Veterinary Manual and peer-reviewed studies of FMD in Ghana), with recommendations pointing farmers to the Ghana Veterinary Service. They are synthetic, template-based examples, not records from real cases.

---

## Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **Model serving:** Hugging Face Inference Endpoints (vLLM, OpenAI-compatible API)
- **Fine-tuning:** 🤗 `transformers`, `peft`, `trl`, `bitsandbytes`
- **Input-validation model:** OpenAI `gpt-4o-mini` (validation only; never diagnoses)
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
HF_BASE_ENDPOINT_URL=https://your-base-endpoint.endpoints.huggingface.cloud
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
├── evaluate_all_models.py      # Base vs fine-tuned evaluation (metrics, McNemar, charts)
├── requirements.txt
├── .env                        # Not committed — see Configuration above
└── Cows datasets/              # Not committed — see Dataset section above
    ├── foot-and-mouth/
    ├── lumpy/
    └── healthy/
```

---

## Limitations

- **Possible train/test image overlap.** The original train/validation split was not seeded, and test images were sampled from the same folders, so an estimated ~20 of the 90 image test cases may have been seen in training. This can inflate the fine-tuned model's image accuracy by a few points; it cannot explain the gap to the original model. A test on new, independently sourced images is future work.
- **Symptom tests reused training phrasing.** The 100% symptom accuracy reflects familiar wording. Testing on farmers' own wording is future work.
- **Template-like explanations.** The "key symptoms visible" text repeats a fixed description per class, and the stated confidence is not calibrated.
- **Not validated in Ghana.** The images are from public collections, not Ghanaian farms.
- **Input validation not measured.** The validation checks were demonstrated on example inputs but not evaluated on a labelled test set.
- **English only** for now.

---

## Acknowledgments

- Supervisor: Professor Isaac Wiafe, Department of Computer Science, University of Ghana.
- Kaggle contributors devang03mgr and warcoder for the public cattle disease image collections.
- Google DeepMind for Gemma 3; Hugging Face, vLLM and Streamlit for the open-source tooling.
- Emergency contacts shown in the app: Ghana Veterinary Service and CSIR Animal Research Institute. These organisations did not provide data for this project.

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