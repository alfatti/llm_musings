import pdfplumber
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI  # adjust if using a different wrapper

# --- Setup LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-pro")  # or however you set it up

def process_page(page_text, page_number):
    # You can customize the prompt here
    prompt = f"Extract structured rent roll data from page {page_number}:\n\n{page_text}"
    return {
        "page": page_number,
        "response": llm.invoke(prompt)
    }

# --- Load PDF ---
pdf_path = "your_rentroll.pdf"
pages = []
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            pages.append((text, i))

# --- Parallel Processing ---
results = []
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_page, text, i) for text, i in pages]
    for future in as_completed(futures):
        results.append(future.result())

# Sort by page number
results.sort(key=lambda x: x["page"])

# --- Combine Results ---
structured_outputs = [res["response"] for res in results]
