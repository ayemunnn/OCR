import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from fpdf import FPDF
import json
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# -----------------------------
# 1. Load env variables
# -----------------------------
load_dotenv()

# Prefer Streamlit Secrets in the cloud, fall back to env locally
API_KEY = None
try:
    API_KEY = st.secrets["HF_API_KEY"]
except Exception:
    API_KEY = os.getenv("HF_API_KEY")

if not API_KEY:
    raise ValueError("Missing HF_API_KEY. Please set it in Streamlit Secrets or as an environment variable.")

# -----------------------------
# 2. Model configuration
# -----------------------------
# IMPORTANT: replace this with YOUR Mistral model (non-chat is fine)
# Example: "mistralai/Mistral-7B-Instruct-v0.3" or your own OCR/LLM model repo
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Hugging Face Inference client (no OpenAI)
client = InferenceClient(
    model=MODEL_NAME,
    token=API_KEY,
)

# -----------------------------
# 3. Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="PaperSleuth", layout="centered")
st.title("📄 PaperSleuth")
st.write("Upload a scanned PDF or image, extract text with OCR, and convert it into structured data.")

uploaded_file = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg"])

ocr_text = ""


def run_ocr_on_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image and return the extracted text."""
    return pytesseract.image_to_string(image)


# -----------------------------
# 4. Handle upload + OCR
# -----------------------------
if uploaded_file:
    file_type = uploaded_file.type
    st.subheader("Uploaded Document")

    if file_type == "application/pdf":
        # Read bytes once for pdf
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes)
        st.write(f"PDF has {len(images)} pages. Processing...")
        pages = []
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i + 1}", use_column_width=True)
            pages.append(run_ocr_on_image(img))
        ocr_text = "\n\n".join(pages)

    elif "image" in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        ocr_text = run_ocr_on_image(image)

    # Allow user to edit the OCR text before sending to the model
    st.subheader("OCR Text")
    ocr_text = st.text_area("Extracted Text (you can edit this):", value=ocr_text, height=250)

    # -----------------------------
    # 5. Call HF Mistral model (text_generation) for JSON extraction
    # -----------------------------
    if st.button("Extract Structured Data"):
        if not ocr_text or not ocr_text.strip():
            st.error("No OCR text found. Please upload a document first.")
        else:
            with st.spinner("Calling Mistral model via Hugging Face Inference API..."):
                # System-style instructions + OCR text in a single prompt
                instructions = (
                    "You are an AI assistant that extracts structured fields "
                    "from noisy OCR text of documents such as invoices, receipts, "
                    "letters, or forms.\n\n"
                    "Return ONLY valid JSON (no markdown, no backticks, no explanations).\n"
                    "If a field is missing, set its value to null or an empty string.\n"
                    "Use keys like: 'document_type', 'name', 'date', 'invoice_number', "
                    "'total_amount', 'address', 'items', or anything else appropriate.\n\n"
                    "OCR_TEXT:\n"
                )

                prompt = instructions + ocr_text

                try:
                    # For non-chat models: plain text generation
                    raw_output = client.text_generation(
                        prompt=prompt,
                        max_new_tokens=512,
                        temperature=0.15,
                        repetition_penalty=1.1,
                    )

                    # -----------------------------
                    # 6. Clean model output and extract JSON
                    # -----------------------------
                    output = raw_output.strip()

                    # Remove ```json or ``` fences if present
                    if "```" in output:
                        # Keep content between the first and last ```
                        parts = output.split("```")
                        if len(parts) >= 3:
                            output = parts[1]  # often the middle part
                        output = output.strip()
                        # Remove leading 'json' tag
                        if output.lower().startswith("json"):
                            output = output[4:].strip()

                    # Try to locate JSON object between first '{' and last '}'
                    start = output.find("{")
                    end = output.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        json_candidate = output[start : end + 1]
                    else:
                        json_candidate = output

                    # -----------------------------
                    # 7. Parse JSON and show result
                    # -----------------------------
                    try:
                        result = json.loads(json_candidate)

                        st.subheader("Structured Output")
                        st.json(result)

                        # -----------------------------
                        # 8. Create downloadable PDF from structured data
                        # -----------------------------
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.set_font("Arial", size=12)

                        pdf.cell(0, 10, txt="PaperSleuth - Extracted Data", ln=True)
                        pdf.ln(5)

                        if isinstance(result, dict):
                            for k, v in result.items():
                                # Convert complex objects to string
                                if not isinstance(v, (str, int, float, type(None))):
                                    v = json.dumps(v, ensure_ascii=False)
                                pdf.multi_cell(0, 8, txt=f"{k}: {v}")
                                pdf.ln(1)
                        else:
                            # Fallback if result is not a dict
                            pdf.multi_cell(0, 8, txt=json.dumps(result, ensure_ascii=False, indent=2))

                        output_pdf_path = "output.pdf"
                        pdf.output(output_pdf_path)

                        with open(output_pdf_path, "rb") as f:
                            st.download_button(
                                "Download Output as PDF",
                                f,
                                file_name="paperSleuth_output.pdf",
                                mime="application/pdf",
                            )

                    except json.JSONDecodeError:
                        st.error("Model did not return valid JSON. Here is the raw output I got:")
                        st.code(output)

                except Exception as e:
                    st.error(f"Error calling Hugging Face Mistral model: {str(e)}")
