import streamlit as st
import base64
import os
import re
import html
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# ── OpenAI client (used as the base-model fallback option) ───────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Hugging Face Inference Endpoint client (your fine-tuned Gemma model) ─────
# The OpenAI-compatible route means we can reuse the same OpenAI SDK,
# just pointed at a different base_url + token.
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")  # e.g. https://xxxx.endpoints.huggingface.cloud/v1
HF_TOKEN = os.getenv("HF_TOKEN")
GEMMA_MODEL_NAME = os.getenv("GEMMA_MODEL_NAME", "gemma3-cattle-disease-ghana")

hf_client = None
if HF_ENDPOINT_URL and HF_TOKEN:
    hf_client = OpenAI(base_url=HF_ENDPOINT_URL, api_key=HF_TOKEN)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CattleCare AI",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B4332, #2D6A4F);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: bold; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }
    .result-card {
        background: #F0FFF4;
        border-left: 5px solid #2D6A4F;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #1B4332;
    }
    .disease-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    .badge-fmd { background: #FFE0E0; color: #E63946; }
    .badge-lsd { background: #FFF3CD; color: #856404; }
    .badge-healthy { background: #D8F3DC; color: #1B4332; }
    .info-box {
        background: #EEF2FF;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        color: #1E2A5A;
    }
    .info-box h4 { color: #1E2A5A; margin-top: 0; }
    .symptom-box {
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        color: #212529;
    }
    .stButton > button {
        background-color: #2D6A4F;
        color: white;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton > button:hover { background-color: #1B4332; }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .source-used {
        background: #D8F3DC;
        color: #1B4332;
        border: 1px solid #95D5B2;
    }
    .source-not-used {
        background: #F1F3F5;
        color: #868E96;
        border: 1px solid #DEE2E6;
        text-decoration: line-through;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────────
def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

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


def _get_client_and_model(use_finetuned):
    """Returns (client, model_name) — fine-tuned requests go to the HF
    Inference Endpoint running your Gemma model; base requests go to OpenAI."""
    if use_finetuned:
        if hf_client is None:
            raise RuntimeError(
                "Fine-tuned model selected but HF_ENDPOINT_URL / HF_TOKEN are not "
                "set in your .env file. Add them or switch off 'Use Fine-tuned Model'."
            )
        return hf_client, GEMMA_MODEL_NAME
    return client, "gpt-4o-mini"


def analyze_image_only(image_bytes, image_type="jpeg", use_finetuned=True):
    """Analyze cattle using image only"""
    img_base64 = image_to_base64(image_bytes)
    active_client, model = _get_client_and_model(use_finetuned)

    response = active_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_IMAGE},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_type};base64,{img_base64}"}
                    },
                    {
                        "type": "text",
                        "text": "What disease does this cattle have? Provide diagnosis and recommendations for the farmer."
                    }
                ]
            }
        ],
        max_tokens=600
    )
    return response.choices[0].message.content


def analyze_symptoms_only(symptoms_text, use_finetuned=True):
    """Analyze cattle using symptom description only"""
    active_client, model = _get_client_and_model(use_finetuned)

    response = active_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SYMPTOMS},
            {
                "role": "user",
                "content": f"My cattle is showing these symptoms: {symptoms_text}. What disease could this be and what should I do?"
            }
        ],
        max_tokens=600
    )
    return response.choices[0].message.content


def analyze_image_and_symptoms(image_bytes, symptoms_text, image_type="jpeg", use_finetuned=True):
    """Analyze cattle using both image and symptom description — most accurate"""
    img_base64 = image_to_base64(image_bytes)
    active_client, model = _get_client_and_model(use_finetuned)

    response = active_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BOTH},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_type};base64,{img_base64}"}
                    },
                    {
                        "type": "text",
                        "text": f"The farmer reports these symptoms: {symptoms_text}. Based on both the image and these symptoms, what disease does this cattle have and what should the farmer do?"
                    }
                ]
            }
        ],
        max_tokens=700
    )
    return response.choices[0].message.content


def bold_inline(text):
    """Escape HTML-sensitive characters, then convert markdown **bold** to <b>."""
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text


# The exact field labels used across the three system prompts above. Matching
# on this text (rather than tracking numbers) is robust to the model being
# inconsistent about numbering — it sometimes omits numbers on top-level
# fields, drops numbers partway through a list, or restarts numbering on the
# last field. Sorted longest-first so "Disease name" can't shadow-match
# inside "Most likely disease name".
_KNOWN_LABELS = sorted([
    "Most likely disease name",
    "Disease name",
    "Confidence level",
    "Key symptoms visible in the image",
    "Why these symptoms suggest this disease",
    "Evidence from image",
    "Evidence from symptoms described",
    "Recommended actions",
    "Other possible diseases to rule out",
], key=len, reverse=True)

_LABEL_PATTERN = re.compile(
    r"(?:\d+\.\s*)?(" + "|".join(re.escape(l) for l in _KNOWN_LABELS) + r")\s*:\s*",
    re.IGNORECASE,
)


def _split_action_items(content):
    """Split a 'Recommended actions' block into individual items, whether
    the model separated them with real line breaks, inline numbering, or
    (partway through) no numbering at all."""
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if len(lines) > 1:
        return [re.sub(r"^\d+\.\s*", "", ln) for ln in lines]

    parts = [p.strip() for p in re.split(r"\d+\.\s+", content) if p.strip()]
    if len(parts) > 1:
        return parts

    return [content] if content else []


def render_diagnosis(text):
    """Parse the model's structured diagnosis into clean, well-spaced HTML,
    matching on known field labels rather than the model's own (unreliable)
    numbering, and rendering 'Recommended actions' as a real ordered list."""
    text = text.strip()
    matches = list(_LABEL_PATTERN.finditer(text))

    if not matches:
        return f"<p>{bold_inline(text)}</p>"

    html_parts = []

    preamble = text[:matches[0].start()].strip()
    if preamble:
        html_parts.append(f'<p style="margin:0 0 10px 0;">{bold_inline(preamble)}</p>')

    for i, m in enumerate(matches):
        label = m.group(1)
        content_start = m.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if "action" in label.lower():
            html_parts.append(f'<p style="margin:0 0 6px 0;"><b>{bold_inline(label)}:</b></p>')
            items = _split_action_items(content)
            if items:
                items_html = "".join(f"<li style='margin-bottom:5px;'>{bold_inline(it)}</li>" for it in items)
                html_parts.append(f'<ol style="margin:0 0 14px 22px; padding:0;">{items_html}</ol>')
            else:
                html_parts.append(f'<p style="margin:0 0 14px 0;">{bold_inline(content)}</p>')
        else:
            html_parts.append(f'<p style="margin:0 0 14px 0;"><b>{bold_inline(label)}:</b> {bold_inline(content)}</p>')

    return "".join(html_parts)


def render_source_badges(image_used, symptoms_used):
    """Deterministic (non-LLM) indicator of which inputs actually fed into
    this diagnosis — based on which analysis function ran, not on the model
    self-reporting, since the fine-tuned model has proven unreliable at
    accurately describing its own reasoning."""
    def badge(label, used):
        cls = "source-used" if used else "source-not-used"
        icon = "✓" if used else "✕"
        return f'<span class="source-badge {cls}">{icon} {html.escape(label)}</span>'

    return (
        '<div style="margin-bottom:0.5rem;">'
        + badge("Image", image_used)
        + badge("Symptom description", symptoms_used)
        + "</div>"
    )


def get_disease_badge(response_text):
    response_lower = response_text.lower()
    if "foot-and-mouth" in response_lower or "fmd" in response_lower:
        return "badge-fmd", "🔴 Foot-and-Mouth Disease"
    elif "lumpy skin" in response_lower or "lsd" in response_lower:
        return "badge-lsd", "🟡 Lumpy Skin Disease"
    elif "healthy" in response_lower:
        return "badge-healthy", "🟢 Healthy"
    else:
        return "badge-fmd", "⚠️ Disease Detected — Consult Vet"


_CHITCHAT_PHRASES = {
    "hi", "hello", "hello there", "hey", "hey there", "yo", "sup", "what's up",
    "whats up", "test", "testing", "ok", "okay", "thanks", "thank you",
    "good morning", "good afternoon", "good evening", "how are you", "hiya",
    "greetings", "yes", "no", "please", "help",
}


def _looks_like_chitchat(symptoms_text):
    """Fast, free, zero-API-call check for the most obvious junk input."""
    cleaned = symptoms_text.strip().lower().rstrip("!.?")
    if len(cleaned) < 8:
        return True
    if cleaned in _CHITCHAT_PHRASES:
        return True
    return False


def _classify_is_symptom(symptoms_text):
    """Second-layer check using a cheap, general-purpose model call. The
    fine-tuned Gemma model has proven unreliable at judging this itself —
    it has invented full diagnoses from a single word like 'hi', and
    separately echoed its own system instructions back verbatim instead of
    following them. A small, fast classification call is far more reliable
    than trying to keyword-match arbitrary natural language."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply with exactly one word: YES if the user's message "
                        "plausibly describes an observable health symptom in a "
                        "farm animal (e.g. fever, blisters, lameness, swelling, "
                        "reduced appetite, nasal discharge), even if brief or "
                        "informally worded. Reply NO if it is small talk, a "
                        "greeting, unrelated chatter, or doesn't describe any "
                        "symptom at all."
                    ),
                },
                {"role": "user", "content": symptoms_text},
            ],
            max_tokens=3,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return not answer.startswith("Y")  # True == needs more info
    except Exception:
        # If the classifier call itself fails (e.g. no OpenAI key set),
        # don't block the user — fall through and let the main model try.
        return False


def needs_more_info(symptoms_text):
    """Two-layer check: an instant free heuristic for obvious junk, then a
    real classification call for anything less obvious that slips past it."""
    if _looks_like_chitchat(symptoms_text):
        return True
    return _classify_is_symptom(symptoms_text)


def _classify_is_cattle_image(image_bytes, image_type="jpeg"):
    """Vision classification check mirroring _classify_is_symptom above.
    There's no keyword heuristic possible for an image, so every upload
    gets one quick, cheap gpt-4o-mini vision call asking a plain yes/no
    question, rather than trusting the fine-tuned model to notice on its
    own that a photo isn't cattle at all (it hasn't reliably done so)."""
    try:
        img_base64 = image_to_base64(image_bytes)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Reply with exactly one word: YES if the image clearly "
                        "shows one or more cattle (cows/bulls), even partially, "
                        "even if blurry or at a distance. Reply NO if it shows "
                        "no cattle at all — a different animal, a person, an "
                        "object, a landscape, text, or anything unrelated."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_type};base64,{img_base64}"}
                        },
                        {"type": "text", "text": "Does this image show cattle?"},
                    ],
                },
            ],
            max_tokens=3,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("Y")  # True == valid cattle image
    except Exception:
        # If the classifier call itself fails, don't block the user —
        # fall through and let the main model try with the raw image.
        return True

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐄 CattleCare AI")
    st.markdown("**AI-Based Cattle Disease Detection**")
    st.markdown("---")

    st.markdown("### About")
    st.markdown("""
    This system uses a fine-tuned Gemma 3 vision model 
    trained on farmer-observable symptoms in Ghanaian cattle disease data to help 
    farmers detect diseases early and also help traders make informed decisions.
    """)

    if hf_client is None:
        st.warning("⚠️ Fine-tuned Gemma endpoint not configured — set `HF_ENDPOINT_URL` and `HF_TOKEN` in your .env file.")

    st.markdown("### Diseases Detected")
    st.markdown("""
    - 🔴 Foot-and-Mouth Disease
    - 🟡 Lumpy Skin Disease
    - 🟢 Healthy (No disease)
    """)

    st.markdown("### How to Use")
    st.markdown("""
    1. Upload a photo of your cattle
    2. Optionally describe symptoms you see
    3. Click **Analyze**
    4. Follow the recommendations
    """)

    st.markdown("---")
    st.markdown("""
    ⚠️ **Disclaimer:** This system is a decision-support 
    tool. Always consult a qualified veterinarian for 
    professional diagnosis and treatment.
    """)
    st.markdown("---")
    st.markdown("**University of Ghana | Computer Science**")
    st.markdown("*Final Year Project 2025/2026*")

# ── Main Page ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🐄 CattleCare AI</h1>
    <p>AI-Based Cattle Disease Detection System for Ghanaian Farmers and Traders</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🔍 Disease Detection",
    "ℹ️ Disease Information"
])

# ── Tab 1: Disease Detection ──────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Step 1 — Upload Cattle Image")
        uploaded_file = st.file_uploader(
            "Take or upload a clear photo of your cattle",
            type=["jpg", "jpeg", "png"],
            help="For best results ensure the affected area is clearly visible"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")

        st.markdown("### Step 2 — Describe Symptoms (Optional)")
        st.markdown("*Describing symptoms alongside the image gives a more accurate diagnosis*")
        st.markdown("**Select symptoms you observe:**")
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            fever = st.checkbox("Fever / High temperature")
            blisters = st.checkbox("Blisters on mouth or feet")
            lumps = st.checkbox("Lumps or nodules on skin")
            lameness = st.checkbox("Limping / Lameness")
            loss_appetite = st.checkbox("Loss of appetite")
            weight_loss = st.checkbox("Weight loss")

        with col_s2:
            salivation = st.checkbox("Excessive salivation")
            reduced_milk = st.checkbox("Reduced milk production")
            swollen_lymph = st.checkbox("Swollen lymph nodes")
            nasal_discharge = st.checkbox("Nasal discharge")
            lethargy = st.checkbox("Lethargy / Weakness")
            skin_lesions = st.checkbox("Skin lesions or wounds")

        # Additional free text
        additional_symptoms = st.text_area(
            "Any other symptoms you notice?",
            placeholder="e.g. the cow has been making unusual sounds, not drinking water...",
            height=80,
            max_chars=300,
            help="Keep this to observable symptoms — max 300 characters"
        )

        # Build symptom string
        selected_symptoms = []
        if fever: selected_symptoms.append("fever")
        if blisters: selected_symptoms.append("blisters on mouth or feet")
        if lumps: selected_symptoms.append("lumps or nodules on skin")
        if lameness: selected_symptoms.append("limping or lameness")
        if loss_appetite: selected_symptoms.append("loss of appetite")
        if weight_loss: selected_symptoms.append("weight loss")
        if salivation: selected_symptoms.append("excessive salivation")
        if reduced_milk: selected_symptoms.append("reduced milk production")
        if swollen_lymph: selected_symptoms.append("swollen lymph nodes")
        if nasal_discharge: selected_symptoms.append("nasal discharge")
        if lethargy: selected_symptoms.append("lethargy or weakness")
        if skin_lesions: selected_symptoms.append("skin lesions or wounds")
        if additional_symptoms:
            selected_symptoms.append(additional_symptoms)

        symptoms_text = ", ".join(selected_symptoms) if selected_symptoms else ""

        st.markdown("### Step 3 — Analysis Settings")
        use_finetuned = st.toggle(
            "Use Fine-tuned Model",
            value=True,
            help="Fine-tuned Gemma model (hosted on your HF Inference Endpoint), specialized for Ghanaian cattle diseases"
        )
        model_label = "Fine-tuned Gemma 3 (HF Endpoint)" if use_finetuned else "Base GPT-4o-mini"
        st.info(f"Using: **{model_label}**")

        analyze_btn = st.button("🔍 Analyze Cattle Health", type="primary")

    with col2:
        st.markdown("### Analysis Results")

        if analyze_btn:
            if not uploaded_file and not symptoms_text:
                st.warning("Please upload an image or describe symptoms to continue.")
            else:
                with st.spinner("Analyzing your cattle... Please wait"):
                    try:
                        insufficient_info = False
                        image_used = False
                        symptoms_used = False

                        image_bytes, ext = None, None
                        if uploaded_file:
                            image_bytes = uploaded_file.getvalue()
                            ext = uploaded_file.name.split(".")[-1].lower()
                            if ext == "jpg":
                                ext = "jpeg"

                        # Validate each input independently before deciding
                        # which analysis path to take. Neither check is
                        # trusted to the fine-tuned model itself — both go
                        # through a small, cheap gpt-4o-mini classification
                        # call, since the fine-tuned model has proven
                        # unreliable at noticing bad input on its own.
                        image_valid = bool(uploaded_file) and _classify_is_cattle_image(image_bytes, ext)
                        symptoms_valid = bool(symptoms_text) and not needs_more_info(symptoms_text)

                        if image_valid and symptoms_valid:
                            response = analyze_image_and_symptoms(
                                image_bytes, symptoms_text, ext, use_finetuned
                            )
                            mode = "Image + Symptoms"
                            image_used = True
                            symptoms_used = True

                        elif image_valid:
                            # Covers: valid image + no symptoms, or valid
                            # image + symptoms that were junk/chitchat —
                            # fall back to the image alone rather than
                            # letting bad text derail a good photo.
                            response = analyze_image_only(
                                image_bytes, ext, use_finetuned
                            )
                            mode = "Image Only"
                            image_used = True

                        elif symptoms_valid:
                            # Covers: valid symptoms + no image, or valid
                            # symptoms + an image that wasn't actually
                            # cattle — fall back to the text alone rather
                            # than letting a bad photo derail real symptoms.
                            response = analyze_symptoms_only(
                                symptoms_text, use_finetuned
                            )
                            mode = "Symptoms Only"
                            symptoms_used = True

                        else:
                            # Neither input was usable — explain specifically
                            # what's wrong with each rather than a generic message.
                            response = None
                            mode = "Insufficient Input"
                            insufficient_info = True
                            problems = []
                            if uploaded_file and not image_valid:
                                problems.append("the uploaded photo doesn't appear to show cattle")
                            if symptoms_text and not symptoms_valid:
                                problems.append("the symptom description doesn't describe an observable symptom")
                            if not uploaded_file and not symptoms_text:
                                problems.append("no image or symptom description was provided")
                            insufficient_reason = "; ".join(problems) if problems else None

                        # Show analysis mode
                        st.success(f"Analysis complete — **{mode}** mode")

                        if insufficient_info:
                            detail = f" ({insufficient_reason})" if insufficient_reason else ""
                            st.info(
                                "🐄 I need a bit more to work with" + detail + " — please "
                                "upload a clear photo that actually shows the animal, and/or "
                                "describe an observable symptom (e.g. fever, blisters, "
                                "lameness, reduced appetite)."
                            )
                        else:
                            # Disease badge
                            badge_class, disease_label = get_disease_badge(response)

                            # Display result — source badges show exactly which
                            # inputs were actually sent to the model for this
                            # specific diagnosis, computed deterministically
                            # from the mode above rather than asked of the model.
                            st.markdown(f"""
                            <div class='result-card'>
                                <span class='disease-badge {badge_class}'>
                                    {disease_label}
                                </span>
                                {render_source_badges(image_used, symptoms_used)}
                                <div style="margin-top:0.75rem; line-height:1.5;">
                                    {render_diagnosis(response)}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Emergency contacts
                            st.markdown("---")
                            st.markdown("### 📞 Emergency Contacts")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.error(
                                    "**Ghana Veterinary Service**\n\n📞 0302-666-626"
                                )
                            with col_b:
                                st.warning(
                                    "**Animal Research Institute**\n\n📞 0302-412-186"
                                )

                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.info("Check your API keys/endpoint URL in the .env file. If using the fine-tuned model, the HF endpoint may be scaled to zero — the first request after idle time can take 1-2 minutes to cold-start.")

        else:
            st.markdown("""
            <div class='info-box'>
                <h4>👈 Get started</h4>
                <p>Upload a photo of your cattle, describe the symptoms 
                you observe, then click <b>Analyze</b> to get an AI-powered 
                disease diagnosis and recommendations.</p>
                <br>
                <p><b>Tip:</b> Using both image and symptoms together gives 
                the most accurate result.</p>
            </div>
            """, unsafe_allow_html=True)

# ── Tab 2: Disease Information ────────────────────────────────────────────────
with tab2:
    st.markdown("### Cattle Disease Reference Guide")

    diseases = [
        {
            "name": "🔴 Foot-and-Mouth Disease (FMD)",
            "symptoms": [
                "Blisters on mouth, feet and teats",
                "Excessive salivation",
                "Lameness or limping",
                "Loss of appetite",
                "High fever"
            ],
            "spread": "Highly contagious — spreads through direct contact and air",
            "action": "Isolate immediately, report to Ghana Veterinary Service, do not move animal off farm",
            "color": "#FFE0E0",
            "border": "#E63946"
        },
        {
            "name": "🟡 Lumpy Skin Disease (LSD)",
            "symptoms": [
                "Nodules and lumps on skin",
                "High fever",
                "Reduced milk production",
                "Swollen lymph nodes",
                "Weight loss"
            ],
            "spread": "Spread by insects — mosquitoes, flies and ticks",
            "action": "Isolate animal, vaccinate healthy herd, spray insecticide, contact vet",
            "color": "#FFF3CD",
            "border": "#E9C46A"
        },
        {
            "name": "🟢 Healthy Cattle",
            "symptoms": [
                "Normal appetite and eating well",
                "Active and alert movement",
                "Smooth and clean skin",
                "Clear and bright eyes",
                "Normal breathing rate"
            ],
            "spread": "N/A",
            "action": "Continue regular monitoring, maintain hygiene, keep vaccinations up to date",
            "color": "#D8F3DC",
            "border": "#2D6A4F"
        }
    ]

    for disease in diseases:
        with st.expander(disease["name"], expanded=False):
            st.markdown(f"""
            <div style='background:{disease["color"]}; padding:1.2rem;
            border-radius:8px; border-left:4px solid {disease["border"]}; color:#212529;'>
            <b>Visible Symptoms:</b><br>
            {"<br>".join(f"• {s}" for s in disease["symptoms"])}
            <br><br>
            <b>How it spreads:</b> {disease["spread"]}
            <br><br>
            <b>Recommended Action:</b> {disease["action"]}
            </div>
            """, unsafe_allow_html=True)