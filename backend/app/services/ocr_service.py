from PIL import Image
import pytesseract


def extract_text_from_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)


def extract_text_from_images(images: list[Image.Image]) -> str:
    page_text = [extract_text_from_image(image) for image in images]
    return normalize_text("\n\n".join(page_text))


def normalize_text(text: str) -> str:
    return text.strip()
