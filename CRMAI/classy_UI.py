import streamlit as st
from datetime import datetime

# -------------------------------------------------------------------
# Assume you already have this implemented somewhere:
# from your_module import classy_email
#
# It should have a signature like:
# def classy_email(email_text: str) -> dict:
#     return {
#         "category": "Address Change",
#         "confidence": 0.87,
#         "task_notes": "Send DDA address change form to client..."
#     }
# -------------------------------------------------------------------

def classy_email(email_text: str) -> dict:
    """
    DUMMY implementation for local testing.
    Replace this with your real classy_email import.
    """
    return {
        "category": "DUMMY_CATEGORY",
        "confidence": 0.75,
        "task_notes": "DUMMY task notes: this is where RA instructions would go."
    }


def save_feedback(email_text, prediction, feedback):
    """
    Placeholder feedback handler.
    Right now it just prints to the terminal.
    You can replace this with a DB write, API call, etc.
    """
    record = {
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
    # TODO: replace with actual persistence (DB, S3, etc.)
    print("FEEDBACK_RECORD:", record)


def main():
    st.set_page_config(
        page_title="AI Email Categorization Tool",
        layout="wide",
    )

    st.title("AI Email Categorization Tool")
    st.markdown(
        """
        Paste an email from a client and let the model:
        1. **Predict a category**,  
        2. **Estimate its confidence**, and  
        3. **Propose task notes** for the Relationship Associate.  

        Then, **provide feedback** so the model can be improved.
        """
    )

    # --- Input area ---
    st.subheader("Client Email")
    email_text = st.text_area(
        "Paste the full email text here (including any relevant context):",
        height=250,
        key="email_input",
    )

    classify_btn = st.button("Classify Email")

    if classify_btn:
        if not email_text.strip():
            st.warning("Please paste an email before running the classification.")
            return

        with st.spinner("Running AI classification..."):
            try:
                result = classy_email(email_text)
            except Exception as e:
                st.error(f"Error while calling classy_email: {e}")
                return

        # Normalize result in case it’s returned as tuple, etc.
        if isinstance(result, dict):
            category = result.get("category")
            confidence = result.get("confidence")
            task_notes = result.get("task_notes")
        elif isinstance(result, (list, tuple)) and len(result) >= 3:
            category, confidence, task_notes = result[:3]
            result = {
                "category": category,
                "confidence": confidence,
                "task_notes": task_notes,
            }
        else:
            st.error("Unexpected output format from classy_email.")
            return

        st.success("Classification complete.")

        # --- Show model outputs ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted Category")
            st.info(category if category is not None else "N/A")

        with col2:
            st.subheader("Model Confidence")
            try:
                if confidence is not None:
                    st.metric(
                        "Confidence",
                        f"{float(confidence) * 100:.1f} %",
                    )
                else:
                    st.write("Confidence: N/A")
            except Exception:
                st.write(f"Confidence: {confidence}")

        st.subheader("Proposed Task Notes")
        st.write(task_notes if task_notes is not None else "N/A")

        st.markdown("---")

        # --- Feedback section ---
        st.subheader("Feedback (for RA / user)")

        st.markdown("### Category Feedback")
        category_is_correct = st.radio(
            "Is the predicted category correct?",
            options=["Yes", "No", "Not sure"],
            index=0,
            key="category_is_correct",
        )

        category_suggested = ""
        if category_is_correct == "No":
            category_suggested = st.text_input(
                "What should the correct category be?",
                placeholder="Type the correct category here...",
                key="category_suggested",
            )

        st.markdown("### Task Notes Feedback")
        notes_are_helpful = st.radio(
            "Are the proposed task notes helpful and complete?",
            options=["Yes", "No", "Partially"],
            index=0,
            key="notes_are_helpful",
        )

        notes_feedback = st.text_area(
            "Any comments / corrections for the task notes?",
            placeholder=(
                "e.g., 'Missing step: notify client of turnaround time', "
                "'Wrong middle office team', etc."
            ),
            key="notes_feedback",
        )

        feedback_btn = st.button("Submit Feedback")

        if feedback_btn:
            feedback_data = {
                "category_is_correct": category_is_correct,
                "category_suggested": category_suggested.strip()
                if category_is_correct == "No"
                else "",
                "notes_are_helpful": notes_are_helpful,
                "notes_feedback": notes_feedback.strip(),
            }

            save_feedback(email_text=email_text, prediction=result, feedback=feedback_data)
            st.success("Thank you! Your feedback has been recorded.")


if __name__ == "__main__":
    main()
