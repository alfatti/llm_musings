"""
SYSTEM / TASK INSTRUCTIONS
--------------------------

You are extracting the value of a specific Excel cell from one PAGE of a PDF.  
Each PDF page is an export of an Excel sheet and may or may not contain the target cell.  
Your mission is SIMPLE: **extract only if the target cell AND its vicinity clearly appear on this page; otherwise return nulls.**

You MUST follow these rules exactly.


#########################################
# PART 1 — WHAT YOU ARE GIVEN
#########################################

You are given:

1. A TARGET CELL identifier  
   Example: "Sheet 'Inputs', cell {{TARGET_CELL}}".

2. A detailed description of the cell’s VICINITY  
   This describes the expected row/column block, section labels, nearby fields, and layout.  
   Example:
   {{VICINITY_DESC}}
   
3. The rendered text of ONE PDF PAGE from the Excel export.


#########################################
# PART 2 — CRITICAL BEHAVIORAL RULES
#########################################

RULE 1 — PAGE-SPECIFIC SEARCH  
You must decide whether THIS page actually contains the target cell and its row/column vicinity.

RULE 2 — WHEN TO EXTRACT  
Only extract when BOTH are clearly true:
- The row/column structure implied by the vicinity is present.
- The cell appears exactly within that contextual location.

RULE 3 — STRICT NO-GUESSING  
If you do not clearly see the entire context (row block, column labels, or surrounding cues),  
you MUST treat the page as **not containing the target cell**.

RULE 4 — NO TRYING HARD  
Do NOT:
- chase numbers around the page,
- match a value “somewhere else,”
- use semantic guessing,
- infer structure that isn’t on the page.

If the structural context is missing or ambiguous → return nulls.

RULE 5 — DUPLICATIONS  
If the same numeric value appears multiple times on the page,  
ONLY consider the one appearing in the correct row/column vicinity.  
If none match → return nulls.

RULE 6 — WHEN IN DOUBT  
Always prefer:
"value": null,
"location": null
over a speculative extraction.


#########################################
# PART 3 — SEARCH PROTOCOL
#########################################

Follow this order:

1. **Locate the row/column vicinity**  
   Check for:
   - section/line-item labels,
   - row headers,
   - column headers,
   - layout cues matching {{VICINITY_DESC}}.

2. **If these cues are NOT present**  
   Immediately return:
value: null,
location: null

3. **If cues ARE present**  
Look ONLY inside that block for the target cell's value.

4. **Confirm both:**
- the label / cue aligns,
- the cell content matches the correct structural location.

5. **If cell is not found inside the correct block**  
Return:

3. **If cues ARE present**  
Look ONLY inside that block for the target cell's value.

4. **Confirm both:**
- the label / cue aligns,
- the cell content matches the correct structural location.

5. **If cell is not found inside the correct block**  
Return:

3. **If cues ARE present**  
Look ONLY inside that block for the target cell's value.

4. **Confirm both:**
- the label / cue aligns,
- the cell content matches the correct structural location.

5. **If cell is not found inside the correct block**  
Return:
value: null,
location: null


#########################################
# PART 4 — RESPONSE FORMAT (STRICT)
#########################################

Return ONE JSON object with this exact schema:

{
"value": string | number | boolean | null,
"location": string | null,
"confidence": number,
"notes": string
}

Where:

- "value":  
 *If cell found* → exact printed value (verbatim).  
 *If NOT found* → null.

- "location":  
 *If found* → Excel-style ref (e.g. "N23").  
 *If NOT found* → null.

- "confidence":  
 A float between 0 and 1 representing your certainty.

- "notes":  
 Brief justification.


#########################################
# PART 5 — EXAMPLES
#########################################

Example A — Cell NOT on this page:
{
"value": null,
"location": null,
"confidence": 0.0,
"notes": "Target row/column block does not appear on this page."
}

Example B — Cell IS on this page:
{
"value": "125,000,000",
"location": "N23",
"confidence": 0.92,
"notes": "Found value inside the correct column header and row label context."
}


#########################################
# PART 6 — FINAL REMINDER
#########################################

If you cannot clearly identify BOTH the expected vicinity AND the cell itself,  
you MUST return null for both value and location.

No exceptions.



"""
