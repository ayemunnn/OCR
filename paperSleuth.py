import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from huggingface_hub import InferenceClient
from fpdf import FPDF
import json
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()
API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise ValueError("Missing HF_API_KEY. Please set it in .env or as an environment variable.")

MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

client = InferenceClient(api_key=API_KEY)

st.set_page_config(page_title="PaperSleuth", layout="centered")
st.title("PaperSleuth")

uploaded_file = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg"])

ocr_text = ""

def run_ocr_on_image(image):
    return pytesseract.image_to_string(image)

if uploaded_file:
    file_type = uploaded_file.type
    st.subheader("Uploaded Document")

    if file_type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
        st.write(f"PDF has {len(images)} pages. Processing...")
        pages = []
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}")
            pages.append(run_ocr_on_image(img))
        ocr_text = "\n\n".join(pages)

    elif "image" in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        ocr_text = run_ocr_on_image(image)

    st.subheader("OCR Text")
    st.text_area("Extracted Text", ocr_text, height=200)

    if st.button("Extract Structured Data"):
        with st.spinner("Calling Mistral..."):

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Mistral OCR 2503, an AI assistant that extracts structured fields "
                        "from scanned document text. Return valid JSON with keys like "
                        "'document_type', 'name', 'date', 'total_amount', 'address'."
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract structured data from:\n\n{ocr_text}"
                }
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.15
                )
                output_json = completion.choices[0].message["content"]

                try:
                    result = json.loads(output_json)
                    st.subheader("Structured Output")
                    st.json(result)

                    # Create downloadable PDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for k, v in result.items():
                        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
                    pdf.output("output.pdf")

                    with open("output.pdf", "rb") as f:
                        st.download_button("Download Output PDF", f, file_name="output.pdf")

                except json.JSONDecodeError:
                    st.error("Mistral did not return valid JSON.")
                    st.code(output_json)

            except Exception as e:
                st.error(f"Error calling Mistral: {str(e)}")
