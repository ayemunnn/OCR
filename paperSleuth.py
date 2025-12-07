from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image
import pytesseract
from huggingface_hub import InferenceClient
from fpdf import FPDF
import json
import os
# -----------------------------
# 1. Auth & model config
# -----------------------------
API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise ValueError("Hugging Face API key not found. Set the 'HF_API_KEY' environment variable.")

MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

# Initialize Hugging Face client (no OpenAI here)
client = InferenceClient(
    model=MODEL_NAME,
    token=API_KEY,        # <- use token instead of api_key
    provider="nebius",    # using HF router provider
)

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.set_page_config(page_title="📄 Mistral OCR 2503", layout="centered")
st.title("📄 Mistral OCR 2503 - Document Parser")

uploaded_file = st.file_uploader("Upload a scanned document (image)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Step 1: OCR
    ocr_text = pytesseract.image_to_string(img)
    st.subheader("📝 OCR Extracted Text")
    ocr_text = st.text_area("OCR Output", ocr_text, height=200)

    if st.button("🧠 Extract Structured Data with Mistral OCR 2503"):
        with st.spinner("Contacting Mistral OCR 2503..."):

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Mistral OCR 2503, an AI assistant that extracts structured fields "
                        "from scanned document text. Return ONLY clean, valid JSON with keys like "
                        "'document_type', 'name', 'date', 'total_amount', 'address'. "
                        "Do not include any explanation, markdown, or backticks."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Extract structured data from this document:\n\n{ocr_text}",
                },
            ]

            try:
                # HF-style chat completion (NOT OpenAI client)
                completion = client.chat_completion(
                    model=MODEL_NAME,      # explicit; OK even though set in client
                    messages=messages,
                    temperature=0.15,
                    max_tokens=512,
                )

                # Handle both dict-style and object-style message
                choice = completion.choices[0].message
                if isinstance(choice, dict):
                    output_json_str = (choice.get("content") or "").strip()
                else:
                    output_json_str = getattr(choice, "content", "") or ""
                    output_json_str = output_json_str.strip()

                # Clean ```json fences if the model adds them
                if output_json_str.startswith("```"):
                    output_json_str = output_json_str.strip("`")
                    if output_json_str.lower().startswith("json"):
                        output_json_str = output_json_str[4:].strip()

                # Try to isolate JSON between first { and last }
                start = output_json_str.find("{")
                end = output_json_str.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_candidate = output_json_str[start : end + 1]
                else:
                    json_candidate = output_json_str

                try:
                    extracted_data = json.loads(json_candidate)

                    st.subheader("📦 Structured Output")
                    st.json(extracted_data)

                    # Step 4: Generate PDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)

                    for key, value in extracted_data.items():
                        # stringify nested objects
                        if not isinstance(value, (str, int, float, type(None))):
                            value = json.dumps(value, ensure_ascii=False)
                        pdf.multi_cell(0, 10, txt=f"{key}: {value}")

                    pdf_file = "extracted_data.pdf"
                    pdf.output(pdf_file)

                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Download PDF", f, file_name="extracted_output.pdf")

                except json.JSONDecodeError:
                    st.error("Mistral did not return valid JSON. Raw output:")
                    st.code(output_json_str)

            except Exception as e:
                st.error(f"API call failed: {str(e)}")
