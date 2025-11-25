from sklearn.metrics import classification_report

def classification_report_on_df(df, true_col="true_category", pred_col="predicted_category"):
    """
    Compute classification report and force inclusion of any label
    predicted by the LLM even if it's not in the ground-truth labels.
    """

    y_true = df[true_col].astype(str).tolist()
    y_pred = df[pred_col].astype(str).tolist()

    # all classes seen anywhere (true OR predicted)
    all_labels = sorted(set(y_true) | set(y_pred))

    report_text = classification_report(
        y_true, 
        y_pred, 
        labels=all_labels,
        zero_division=0
    )

    report_dict = classification_report(
        y_true, 
        y_pred, 
        labels=all_labels, 
        zero_division=0, 
        output_dict=True
    )

    return report_text, report_dict
