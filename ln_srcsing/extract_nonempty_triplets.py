def extract_non_empty_triplets(results):
    """
    results: list of dicts returned per page.
    returns: list of (value, location, confidence) for all pages with non-null values.
    """

    triplets = []
    for item in results:
        val = item.get("value")

        # treat None, empty string, empty list, empty numeric as empty
        if val is None or val == "" or (isinstance(val, (list, dict)) and not val):
            continue

        triplets.append((
            val,
            item.get("location"),
            item.get("confidence")
        ))

    return triplets
