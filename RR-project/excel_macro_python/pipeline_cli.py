# pipeline_cli.py
import argparse, sys, pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rent-roll", required=True)
    p.add_argument("--concessions")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # 1) Load inputs
    rr_df = pd.read_csv(args.rent_roll)
    conc_df = pd.read_csv(args.concessions) if args.concessions else None

    # 2) >>> YOUR GEMINI EXTRACTION HERE <<<
    #    - Normalize column names via LLM
    #    - Extract required fields to a stable schema: e.g. ['Unit','Tenant','Sqft', ...]
    #    - Validate none/NaNs as needed
    extracted = run_gemini_extract(rr_df)  # <-- your function

    # 3) Optional join on Unit
    if conc_df is not None:
        out_df = join_on_unit(extracted, conc_df)  # <-- your function; add is_concession, concession_amount
    else:
        out_df = extracted

    # 4) Save
    out_df.to_csv(args.out, index=False)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Print to stderr so VBA logs get it
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
