from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
from io import BytesIO

from rent_roll_extractor import extract_rent_roll
from concession_joiner import join_concessions

app = FastAPI(title="Rent Roll Extractor API")

# Allow local Streamlit frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:8501"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Endpoint: Extract Only ===
@app.post("/extract")
async def extract_rent_roll_only(rent_roll: UploadFile = File(...)):
    df = pd.read_excel(rent_roll.file)
    extracted_df = extract_rent_roll(df)

    buffer = BytesIO()
    extracted_df.to_excel(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={
        "Content-Disposition": "attachment; filename=rent_roll_extract.xlsx"
    })

# === Endpoint: Extract + Join with Concessions ===
@app.post("/extract_and_join")
async def extract_and_join(
    rent_roll: UploadFile = File(...),
    concession: UploadFile = File(...)
):
    rent_df = pd.read_excel(rent_roll.file)
    cons_df = pd.read_excel(concession.file)

    extracted_df = extract_rent_roll(rent_df)
    final_df = join_concessions(extracted_df, cons_df)

    buffer = BytesIO()
    final_df.to_excel(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={
        "Content-Disposition": "attachment; filename=rent_roll_final.xlsx"
    })
