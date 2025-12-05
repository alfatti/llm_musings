# page_split_tenants_prompt.py

PAGE_SPLIT_TENANTS_SECTION = """
### Handling rows split across pages

Some tenant rows in this rent roll may be **split across pages**. For example, part of a row may appear at the very bottom of one page and the rest of the row may appear at the very top of the next page. In these situations, you **must not** hallucinate or fill in missing information from context.

Use the following rules whenever a tenant row looks incomplete at the **top** or **bottom** edge of the page:

1. **Detect partial (page-split) rows**

   Treat a tenant row as **partial** (page-split) if any of the following are true:

   - The row appears at the **very bottom** of the page and clearly continues below the visible area (for example, you can see some cells but not all expected columns).
   - The row appears at the **very top** of the page and clearly continues from a previous page (for example, you see a continuation of text in the middle of a row without its usual start).
   - Only a subset of the expected columns for a tenant are visible, and the truncation is clearly due to the page boundary.

2. **Do not guess or fabricate missing values**

   - **Only** output values that are clearly visible on the current page.
   - **Never** copy values from another row to fill missing columns.
   - **Never** invent values that are not explicitly shown on the page.

   If you cannot see a value for a given field because it has been cut off by the page boundary, you must leave that field as `null` and describe it in `missing_fields_due_to_page_split`.

3. **Annotate page-split rows using flags**

   For **every tenant row** you output, include the following fields in the JSON:

   - `row_split_flag`: a string that must be exactly one of:
     - `"NONE"` – use this when the row is fully visible on this page and is not affected by a page split.
     - `"TOP_SPLIT"` – use this when the visible part of the row is clearly the **continuation of a row that started on a previous page** (i.e., the row is truncated at the top).
     - `"BOTTOM_SPLIT"` – use this when the visible part of the row is clearly **cut off at the bottom** and will continue on the next page.

   - `is_complete_row`: a boolean indicating whether the row is fully visible on this page:
     - `true` if **all** columns needed for this tenant are visible on this page (no truncation).
     - `false` if any part of the row is cut off by the page boundary, or if any required fields are missing *because* of a split across pages.

   - `missing_fields_due_to_page_split`: a list of strings naming the fields that are missing **specifically because of the page split**, for example:
     - `["base_rent", "lease_end"]`
     - `[]` if the row is complete or its missing data is not due to a page split.

4. **How to populate these fields (examples)**

   - **Example A – bottom split**

     A row is at the very bottom of the page. You see `UNIT`, `TENANT_NAME`, and `LEASE_START`, but the columns for `LEASE_END`, `BASE_RENT`, and `CHARGE_TYPE` have been cut off.

     In this case, output something like:

     - `row_split_flag`: `"BOTTOM_SPLIT"`
     - `is_complete_row`: `false`
     - `missing_fields_due_to_page_split`: `["lease_end", "base_rent", "charge_type"]`

     Any fields you cannot see should be set to `null` in the JSON.

   - **Example B – top split**

     At the top of the page, you see a row where `UNIT` is not visible, but the row clearly looks like the continuation of a tenant from the previous page. You can see `TENANT_NAME`, `LEASE_END`, and `BASE_RENT`, but not `UNIT` or `LEASE_START`.

     In this case, output:

     - `row_split_flag`: `"TOP_SPLIT"`
     - `is_complete_row`: `false`
     - `missing_fields_due_to_page_split`: `["unit", "lease_start"]`

     Again, any missing values due to the split must be `null`.

   - **Example C – complete row (no split)**

     A row appears entirely within the page margins, and all relevant columns are visible.

     In this case, output:

     - `row_split_flag`: `"NONE"`
     - `is_complete_row`: `true`
     - `missing_fields_due_to_page_split`: `[]`

5. **When in doubt about a page split**

   - If you are not sure whether a row is split across pages, prefer to be **conservative**:
     - Mark `row_split_flag = "NONE"` and `is_complete_row = true` **only** when you are confident the row is fully visible on the page.
     - If you see strong visual cues of truncation near the page edges, treat it as a split and set `is_complete_row = false` with the appropriate `row_split_flag`.
"""
