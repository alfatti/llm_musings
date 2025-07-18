import gradio as gr
import pandas as pd
from io import BytesIO
import tempfile
import os
import platform
import subprocess

# Assume these are defined already
# def extract_rent_roll(df: pd.DataFrame) -> pd.DataFrame: ...
# def join_concessions(extracted_df: pd.DataFrame, concessions_df: pd.DataFrame) -> pd.DataFrame: ...

# Helper to open file locally
def open_file(path):
    if platform.system() == "Darwin":  # macOS
        subprocess.call(["open", path])
    elif platform.system() == "Windows":
        os.startfile(path)
    else:  # Linux
        subprocess.call(["xdg-open", path])

# Process files and output Excel
def process_files_with_preview(rent_roll_file, concession_file):
    if rent_roll_file is None and concession_file is not None:
        return "❌ Error: Rent Roll is required.", None, None, None, None

    if rent_roll_file is None:
        return "⚠️ Please upload at least the Rent Roll file.", None, None, None, None

    rent_df = pd.read_excel(rent_roll_file)

    if concession_file is None:
        extracted_df = extract_rent_roll(rent_df)
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
        extracted_df.to_excel(tmp_path, index=False)
        open_file(tmp_path)
        return (
            "✅ Only Rent Roll uploaded. Extraction complete!",
            rent_df.head(),
            None,
            extracted_df.head(),
            tmp_path
        )
    else:
        cons_df = pd.read_excel(concession_file)
        extracted_df = extract_rent_roll(rent_df)
        final_df = join_concessions(extracted_df, cons_df)
        tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
        final_df.to_excel(tmp_path, index=False)
        open_file(tmp_path)
        return (
            "✅ Extraction and Concession Join complete!",
            rent_df.head(),
            cons_df.head(),
            final_df.head(),
            tmp_path
        )

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🏢 RR AI Extractor")
    gr.Markdown("AI-powered Rent Roll Data Extraction Tool")

    gr.Markdown("### ℹ️ Instructions")
    gr.Markdown("""
    **Before uploading your sheets, please ensure:**
    - The file is in plain **tabular format**  
    - Irregular headers (like titles, unit type rows) are removed  
    - Rows with summary statistics (e.g., totals) are removed  
    """)

    with gr.Row():
        rent_roll_input = gr.File(label="📄 Upload Rent Roll Excel", file_types=[".xlsx", ".xls"])
        concession_input = gr.File(label="📄 Upload Concession Sheet (Optional)", file_types=[".xlsx", ".xls"])

    submit_btn = gr.Button("🚀 Run Extraction")
    message_output = gr.Textbox(label="🗨️ Status Message")

    with gr.Accordion("📊 Uploaded Sheet Previews", open=False):
        rent_preview = gr.Dataframe(label="Rent Roll Preview", interactive=False)
        concession_preview = gr.Dataframe(label="Concession Preview", interactive=False)

    with gr.Accordion("📈 Final Extract Preview", open=False):
        extract_preview = gr.Dataframe(label="Extracted Output Preview", interactive=False)

    file_output = gr.File(label="📥 Download Result", visible=False)

    submit_btn.click(
        fn=process_files_with_preview,
        inputs=[rent_roll_input, concession_input],
        outputs=[
            message_output,
            rent_preview,
            concession_preview,
            extract_preview,
            file_output
        ]
    )

demo.launch(share=False)
