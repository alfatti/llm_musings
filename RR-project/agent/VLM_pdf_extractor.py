#!pip install protobuf==4.25.3 google-generativeai==0.5.2 langchain-google-genai==1.0.1 langchain==0.1.19 pandas markdown beautifulsoup4 pdf2image matplotlib


from pdf2image import convert_from_path
from pathlib import Path
import os

# Define the input PDF path and output directory
pdf_path = "/mnt/data/rent_roll_sample_final.pdf"
output_dir = "/mnt/data/pdf_images"
os.makedirs(output_dir, exist_ok=True)

# Convert PDF to images (one image per page)
images = convert_from_path(pdf_path, dpi=300)

# Save each image to the output directory
image_paths = []
for i, img in enumerate(images):
    image_path = Path(output_dir) / f"page_{i+1}.png"
    img.save(image_path, "PNG")
    image_paths.append(str(image_path))



#=========================================================
import base64
import requests
import pandas as pd
from markdownify import markdownify as md
from io import StringIO
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
GEMINI_API_KEY = "YOUR_API_KEY"  # Replace with your actual Gemini API Key
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

# --- FUNCTION TO LOAD IMAGE AS BASE64 ---
def load_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded

# --- GEMINI VISION PROMPT ---
VISION_PROMPT = [
    {
        "text": (
            "This is a rent roll page. Please extract all tabular information and return it as a clean Markdown table. "
            "Include all units, line items, and monetary amounts. Do not summarize. "
            "Make sure each table has column headers and no rows are skipped. "
            "Use one Markdown table per group if needed."
        )
    }
]

# --- FUNCTION TO CALL GEMINI VISION ---
def extract_table_markdown_from_image(image_path):
    image_base64 = load_image_base64(image_path)
    request_body = {
        "contents": [
            {
                "parts": VISION_PROMPT + [
                    {"inlineData": {"mimeType": "image/png", "data": image_base64}}
                ]
            }
        ]
    }
    response = requests.post(
        GEMINI_ENDPOINT,
        headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
        json=request_body,
    )
    if response.status_code != 200:
        raise Exception(f"Gemini Vision error: {response.text}")
    
    reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return reply

# --- FUNCTION TO PARSE MARKDOWN TO DATAFRAME ---
def markdown_to_dataframes(markdown_text):
    import re
    import markdown
    from bs4 import BeautifulSoup

    # Convert Markdown to HTML and parse tables
    html = markdown.markdown(markdown_text, extensions=["markdown.extensions.tables"])
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")

    dfs = []
    for table in tables:
        df = pd.read_html(str(table))[0]
        dfs.append(df)
    return dfs

# --- MAIN PROCESSING FUNCTION ---
def process_rent_roll_pages(image_paths):
    all_dataframes = []
    for img_path in image_paths:
        print(f"Processing: {img_path}")
        markdown = extract_table_markdown_from_image(img_path)
        dfs = markdown_to_dataframes(markdown)
        all_dataframes.extend(dfs)
    return all_dataframes


#===================================================
#langchain
#===================================================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
import pandas as pd
import markdown
from bs4 import BeautifulSoup

# LangChain LLM Client
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", temperature=0)

# Helper: Convert PNG image to inline base64 content
def encode_image_base64(image_path: str):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# LangChain-compatible function
def extract_markdown_tables_with_langchain(image_path: str) -> list[pd.DataFrame]:
    prompt = (
        "This is a rent roll page. Extract all tabular data as Markdown tables. "
        "Include headers, monetary amounts, and every row, even if some cells are blank. "
        "Do not summarize or skip details."
    )

    # Encode the image for Gemini Vision
    image_base64 = encode_image_base64(image_path)
    image_data = {
        "mime_type": "image/png",
        "data": image_base64
    }

    # Send the image + prompt to Gemini Vision via LangChain
    response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt},
                                                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}])])

    markdown_text = response.content

    # Convert markdown to DataFrames
    html = markdown.markdown(markdown_text, extensions=["markdown.extensions.tables"])
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    dfs = [pd.read_html(str(t))[0] for t in tables]
    return dfs

