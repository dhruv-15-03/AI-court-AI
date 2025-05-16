import PyPDF2
import requests

HUGGINGFACE_API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}


def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"


def summarize_text(text, max_length=512):
    try:
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": 300,
                "do_sample": False
            }
        }
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        summary = response.json()[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Error summarizing text: {e}"


def summarize_pdf(file_path):
    print(f"\nReading file: {file_path}")
    text = extract_text_from_pdf(file_path)

    if text.startswith("Error"):
        print(text)
        return text

    print("Extracting summary (this may take a few seconds)...")
    summary = summarize_text(text[:3000])

    print("\n📝 Summary:\n")
    print(summary)
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python summarize_pdf.py <path_to_pdf>")
    else:
        summarize_pdf(sys.argv[1])
