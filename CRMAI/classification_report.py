from sklearn.metrics import classification_report

def classification_report_on_df(df, true_col="true_category", pred_col="predicted_category"):
    """
    Compute sklearn classification report on an entire test set stored in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain true labels and predicted labels.
    true_col : str
        Column name for the ground-truth labels.
    pred_col : str
        Column name for the predicted labels.

    Returns
    -------
    report_text : str
        Pretty text classification report.
    report_dict : dict
        Dictionary version of the classification report.
    """

    y_true = df[true_col].astype(str).tolist()
    y_pred = df[pred_col].astype(str).tolist()

    report_text = classification_report(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    return report_text, report_dict
