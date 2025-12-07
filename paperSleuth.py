from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from huggingface_hub import InferenceClient
from fpdf import FPDF
import json
import os

# -------------------------------------------------------------------
API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise ValueError("Hugging Face API key not found. Set 'HF_API_KEY' in your environment.")

MODEL_NAME = "google/gemma-3-27b-it"

client = InferenceClient(
    model=MODEL_NAME,
    token=API_KEY,
)

# -------------------------------------------------------------------
# Streamlit page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Document Skimmer · OCR + LLM",
    layout="centered"
)

st.title("Document Skimmer")
st.caption(
    "Upload any PDF or image. The application will run OCR and have a large language model skim the text, "
    "returning a summary, key points, and extracted entities in JSON."
)

uploaded_file = st.file_uploader(
    "Upload a document (PDF or image)",
    type=["pdf", "png", "jpg", "jpeg"]
)

ocr_text = ""

# -------------------------------------------------------------------
# File handling & OCR
# -------------------------------------------------------------------
if uploaded_file:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        st.subheader("Preview")
        pdf_bytes = uploaded_file.read()
        pages = convert_from_bytes(pdf_bytes)

        all_page_text = []
        for i, page_img in enumerate(pages):
            st.image(page_img, caption=f"Page {i + 1}", use_column_width=True)
            page_text = pytesseract.image_to_string(page_img)
            all_page_text.append(page_text)

        ocr_text = "\n\n".join(all_page_text)

    else:
        # Image
        st.subheader("Preview")
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        ocr_text = pytesseract.image_to_string(img)

    st.subheader("OCR Output")
    ocr_text = st.text_area(
        "Review or edit extracted text before analysis:",
        ocr_text,
        height=260
    )

    # -------------------------------------------------------------------
    # LLM: Generic “skim & extract info” for any document
    # -------------------------------------------------------------------
    if st.button("Analyze document"):
        if not ocr_text.strip():
            st.error("No OCR text available. Please upload a document first.")
        else:
            with st.spinner("Analyzing document with the language model..."):
                system_prompt = (
                    "You read OCR text from arbitrary documents: invoices, receipts, reports, "
                    "emails, forms, academic papers, legal contracts, handwritten notes, etc.\n\n"
                    "Your job is to quickly skim the text and extract the most important information "
                    "in a generic structure that works for any document type.\n\n"
                    "Return ONLY valid JSON (no markdown, no backticks, no extra commentary).\n"
                    "Use the following schema:\n\n"
                    "{\n"
                    '  \"summary\": \"Short 3–6 sentence overview of the document.\",\n'
                    '  \"key_points\": [\"Bullet point 1\", \"Bullet point 2\", ...],\n'
                    "  \"entities\": {\n"
                    '    \"people\": [\"names...\"],\n'
                    '    \"organizations\": [\"names...\"],\n'
                    '    \"locations\": [\"locations...\"],\n'
                    '    \"dates\": [\"dates mentioned...\"],\n'
                    '    \"amounts\": [\"monetary or numeric amounts with context if possible\"]\n'
                    "  },\n"
                    "  \"metadata\": {\n"
                    '    \"document_type\": \"Your best guess (e.g., invoice, email, report, letter, form, contract, unknown)\",\n'
                    '    \"language\": \"Language of the document if you can infer it (e.g., en, fr, es)\",\n'
                    '    \"confidence_notes\": \"Optional short note about how confident you are and any limitations.\"\n'
                    "  }\n"
                    "}\n\n"
                    "If you cannot fill a field, use null or an empty list/empty string as appropriate."
                )

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Here is the OCR text from the document:\n\n{ocr_text}",
                    },
                ]

                try:
                    completion = client.chat_completion(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.15,
                        max_tokens=1024,
                    )

                    choice = completion.choices[0].message
                    if isinstance(choice, dict):
                        output_json_str = (choice.get("content") or "").strip()
                    else:
                        output_json_str = getattr(choice, "content", "") or ""
                        output_json_str = output_json_str.strip()

                    # Clean potential ```json fences
                    if output_json_str.startswith("```"):
                        output_json_str = output_json_str.strip("`")
                        if output_json_str.lower().startswith("json"):
                            output_json_str = output_json_str[4:].strip()

                    # Try to isolate JSON between { and }
                    start = output_json_str.find("{")
                    end = output_json_str.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        json_candidate = output_json_str[start: end + 1]
                    else:
                        json_candidate = output_json_str

                    try:
                        extracted_data = json.loads(json_candidate)

                        st.subheader("Extracted Information (JSON)")
                        st.json(extracted_data)

                        # ---------------------------------------------------
                        # Export as PDF summary (optional)
                        # ---------------------------------------------------
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.set_font("Arial", size=12)

                        pdf.cell(0, 10, txt="Document Skimmer Summary", ln=True)
                        pdf.ln(5)

                        # Summary
                        summary = extracted_data.get("summary", "")
                        pdf.set_font("Arial", style="B", size=12)
                        pdf.cell(0, 8, txt="Summary:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 8, txt=str(summary))
                        pdf.ln(3)

                        # Key points
                        key_points = extracted_data.get("key_points", [])
                        pdf.set_font("Arial", style="B", size=12)
                        pdf.cell(0, 8, txt="Key Points:", ln=True)
                        pdf.set_font("Arial", size=12)
                        if isinstance(key_points, list):
                            for kp in key_points:
                                pdf.multi_cell(0, 8, txt=f"- {kp}")
                        else:
                            pdf.multi_cell(0, 8, txt=str(key_points))
                        pdf.ln(3)

                        # Entities
                        entities = extracted_data.get("entities", {})
                        pdf.set_font("Arial", style="B", size=12)
                        pdf.cell(0, 8, txt="Entities:", ln=True)
                        pdf.set_font("Arial", size=12)
                        if isinstance(entities, dict):
                            for ent_type, values in entities.items():
                                pdf.multi_cell(0, 8, txt=f"{ent_type.capitalize()}:")
                                if isinstance(values, list):
                                    for v in values:
                                        pdf.multi_cell(0, 8, txt=f"  - {v}")
                                else:
                                    pdf.multi_cell(0, 8, txt=f"  {values}")
                                pdf.ln(1)
                        else:
                            pdf.multi_cell(0, 8, txt=str(entities))
                        pdf.ln(3)

                        # Metadata
                        metadata = extracted_data.get("metadata", {})
                        pdf.set_font("Arial", style="B", size=12)
                        pdf.cell(0, 8, txt="Metadata:", ln=True)
                        pdf.set_font("Arial", size=12)
                        if isinstance(metadata, dict):
                            for k, v in metadata.items():
                                pdf.multi_cell(0, 8, txt=f"{k}: {v}")
                        else:
                            pdf.multi_cell(0, 8, txt=str(metadata))

                        pdf_path = "document_skimmer_summary.pdf"
                        pdf.output(pdf_path)

                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                label="Download Summary as PDF",
                                data=f,
                                file_name="document_skimmer_summary.pdf",
                                mime="application/pdf",
                            )

                    except json.JSONDecodeError:
                        st.error("The model did not return valid JSON. Raw output:")
                        st.code(output_json_str)

                except Exception as e:
                    st.error(f"Model API call failed: {str(e)}")
else:
    st.info("Upload a PDF or image file to begin.")
