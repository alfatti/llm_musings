                          ┌───────────────────────────────┐
                          │     User Uploads Rent Roll     │
                          │            (PDF)              │
                          └──────────────┬────────────────┘
                                         ▼
                          ┌───────────────────────────────┐
                          │  Shred PDF → Images per Page  │
                          │   (e.g., pdf2image or poppler)│
                          └──────────────┬────────────────┘
                                         ▼
                             For each page image:
                                         ▼
                        ┌────────────────────────────────┐
                        │  Send to Gemini 2.5 Pro VLM     │
                        │  (zero-shot, image prompt only) │
                        │  ↓                              │
                        │  Return page-level JSON extract │
                        └────────────┬───────────────────┘
                                     ▼
                  ┌─────────────────────────────────────────┐
                  │  Validate JSON per page (schema check)  │
                  └────────────────┬────────────────────────┘
                                   ▼
                     ┌────────────────────────────┐
                     │ Pages with valid output     │
                     └────────────┬───────────────┘
                                   │
                     ┌────────────▼────────────┐
                     │ Pages with invalid JSON │◄────────────────────┐
                     └────────────┬────────────┘                     │
                                  │                                  │
                   Repeat below until valid or MAX_ITER             │
                                  ▼                                  │
               ┌────────────────────────────────────────────┐        │
               │   Rerun Invalid Pages through VLM again     │───────┘
               └────────────────────────────────────────────┘

After all pages processed:
                     ▼
       ┌────────────────────────────────────────────┐
       │ Merge All Valid JSON Pages into One Table  │
       └────────────────┬───────────────────────────┘
                        ▼
       ┌────────────────────────────────────────────┐
       │ Postprocess Table (type casting, cleanups) │
       └────────────────┬───────────────────────────┘
                        ▼
       ┌────────────────────────────────────────────┐
       │ Export Final Extract (Excel / CSV / JSON)  │
       └────────────────────────────────────────────┘
