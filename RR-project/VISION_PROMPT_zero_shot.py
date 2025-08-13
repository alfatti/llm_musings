from pathlib import Path

# Recreate the zero-shot JSON prompt text after reset
zero_shot_prompt = """You are a precise rent-roll extractor. This is a ZERO-SHOT setting: you have no prior examples.
The page can show either:
(A) classic rows with Unit and Tenant per row, or
(B) grouped rows where a single tenant spans multiple units (merged cells, stacked units, bullets/commas) and sometimes Unit and Tenant are in different columns or regions.

GOAL
Produce EXACTLY one JSON array. Each element is an object with the keys:
["Unit","TenantName","TenantExternalID","TenantGroupID","UnitType","SqFt","Status",
"MarketRent","ConcessionAmount","IsConcession","MoveIn","MoveOut",
"ParsedFromGroup","UnitListRaw","TenantRaw"]

OUTPUT RULES (CRITICAL)
- One JSON object per unit (ONE ROW PER UNIT).
- If the page groups multiple units under one tenant:
  - Duplicate tenant fields across the split objects.
  - Derive a stable TenantGroupID (e.g., deterministic slug from tenant name + first unit + move-in date).
  - Set ParsedFromGroup=true.
  - Store the original raw grouped unit text in UnitListRaw.
- If not grouped, ParsedFromGroup=false and UnitListRaw is empty.

UNIT PARSING
- Accept comma-separated, bullets, stacked lines, or ranges.
- Expand ranges: "104–106" → "104","105","106".
- Strip building prefixes where obvious but keep full token if distinct buildings are indicated.
- If unreadable or ambiguous, set Unit="" but keep the object.

TENANT FIELDS
- TenantName: exactly as shown.
- TenantExternalID: present if visible, else "".
- TenantGroupID: required only when ParsedFromGroup=true, else "".

NORMALIZATION
- Dates → YYYY-MM-DD if visible; else "".
- Currency → plain number string without $ or commas.
- Booleans → true/false (lowercase).
- Missing values → "".
- TenantRaw captures raw tenant text span if distant from unit list.

VALIDATION
- Count of objects = count of parsed units.
- Keys must match exactly and be in the order above.
- Output only valid JSON (no extra text, no comments, no trailing commas).

AMBIGUITY HANDLING
- Prefer empty strings over guessed values.
- If both grouped and per-unit entries exist for same unit, deduplicate by Unit (keep one object per Unit).

STRICT INSTRUCTIONS
- Do NOT wrap JSON in code fences.
- Do NOT output any explanation, notes, or commentary.
- Output only the JSON array."""

# Save to file
zero_shot_file = Path("/mnt/data/rentroll_prompt_zero_shot_json.txt")
zero_shot_file.write_text(zero_shot_prompt, encoding="utf-8")

zero_shot_file
