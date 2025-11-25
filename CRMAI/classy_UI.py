%%writefile ai_email_app.py
import streamlit as st
from datetime import datetime

# Dummy implementation — replace with your real LLM function
def classy_email(email_text: str) -> dict:
    return {
        "category": "Address Change",
        "confidence": 0.82,
        "task_notes": "Send address update form and notify middle office."
    }


FEEDBACK_FILE = "email_feedback.csv"

def save_feedback(email_text, prediction, feedback):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "email_text": email_text,
        "predicted_category": prediction.get("category"),
        "predicted_confidence": prediction.get("confidence"),
        "predicted_task_notes": prediction.get("task_notes"),
        "category_is_correct": feedback.get("category_is_correct"),
        "category_suggested": feedback.get("category_suggested"),
        "notes_are_helpful": feedback.get("notes_are_helpful"),
        "notes_feedback": feedback.get("notes_feedback"),
    }

    # Append or create CSV
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(FEEDBACK_FILE, index=False)
    else:
        pd.DataFrame([row]).to_csv(FEEDBACK_FILE, index=False)


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
