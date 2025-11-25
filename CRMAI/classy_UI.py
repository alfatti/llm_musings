%%writefile ai_email_app.py
import streamlit as st
from datetime import datetime
import json
import os

# Dummy implementation — replace with your real LLM function
def classy_email(email_text: str) -> dict:
    return {
        "category": "Address Change",
        "confidence": 0.82,
        "task_notes": "Send address update form and notify middle office."
    }



FEEDBACK_DIR = "feedback_records_csv"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

def save_feedback(email_text, prediction, feedback):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S-%f")
    filename = f"feedback_{ts}.csv"
    path = os.path.join(FEEDBACK_DIR, filename)

    df = pd.DataFrame([{
        "timestamp": ts,
        "email_text": email_text,
        **prediction,
        **feedback
    }])

    df.to_csv(path, index=False)
    print(f"Saved → {path}")




def main():
    st.set_page_config(page_title="AI Email Categorization Tool", layout="wide")
    st.title("AI Email Categorization Tool")

    st.subheader("Client Email")
    email_text = st.text_area("Paste the email text:", height=250)

    if st.button("Classify Email"):
        result = classy_email(email_text)

        st.success("Classification Complete")
        st.write("**Predicted Category:**", result["category"])
        st.write("**Confidence:**", f"{result['confidence']*100:.1f}%")
        st.write("**Task Notes:**", result["task_notes"])

        st.markdown("---")
        st.subheader("Feedback")

        category_is_correct = st.radio(
            "Is the predicted category correct?",
            ["Yes", "No", "Not sure"]
        )
        category_suggested = ""
        if category_is_correct == "No":
            category_suggested = st.text_input("Correct category:")

        notes_are_helpful = st.radio(
            "Are the task notes helpful?",
            ["Yes", "No", "Partially"]
        )
        notes_feedback = st.text_area("Comments on task notes:")

        if st.button("Submit Feedback"):
            feedback = {
                "category_is_correct": category_is_correct,
                "category_suggested": category_suggested,
                "notes_are_helpful": notes_are_helpful,
                "notes_feedback": notes_feedback,
            }
            save_feedback(email_text, result, feedback)
            st.success("Feedback submitted.")

if __name__ == "__main__":
    main()
