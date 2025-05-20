from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from huggingface_hub import InferenceClient
from fpdf import FPDF
import json
import os

API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise ValueError("Hugging Face API key not found. Set the 'HF_API_KEY' environment variable.")

MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

# Initialize Hugging Face client
client = InferenceClient(
    provider="nebius",
    api_key=API_KEY,
)

st.set_page_config(page_title="📄 Mistral OCR 2503", layout="centered")
st.title("📄 Mistral OCR 2503 - Document Parser")

uploaded_file = st.file_uploader("Upload a scanned document (image)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Step 1: OCR
    ocr_text = pytesseract.image_to_string(img)
    st.subheader("📝 OCR Extracted Text")
    st.text_area("OCR Output", ocr_text, height=200)

    if st.button("🧠 Extract Structured Data with Mistral OCR 2503"):
        with st.spinner("Contacting Mistral OCR 2503..."):

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Mistral OCR 2503, an AI assistant that extracts structured fields "
                        "from scanned document text. Return only clean, valid JSON with keys like "
                        "'document_type', 'name', 'date', 'total_amount', 'address'."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract structured data from this document:\n\n{ocr_text}"
                }
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.15
                )

                output_json_str = completion.choices[0].message["content"]

                try:
                    extracted_data = json.loads(output_json_str)
                    st.subheader("📦 Structured Output")
                    st.json(extracted_data)

                    # Step 4: Generate PDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for key, value in extracted_data.items():
                        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
                    pdf_file = "extracted_data.pdf"
                    pdf.output(pdf_file)

                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Download PDF", f, file_name="extracted_output.pdf")

                except json.JSONDecodeError:
                    st.error("Mistral did not return valid JSON. Output:")
                    st.code(output_json_str)

            except Exception as e:
                st.error(f"API call failed: {str(e)}")
