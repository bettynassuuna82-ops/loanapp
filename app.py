import os
import joblib
import pandas as pd
import streamlit as st

MODEL_FILE = "loan_model.pkl"
APPLICATION_FILE = "loan_applications.csv"

st.set_page_config(page_title="SACCO Loan Approval", page_icon="💰")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def predict_new_loan(pipeline, member, employment, income,
                     loan_amount, loan_term, credit_history):
    application = pd.DataFrame([{
        "member": member,
        "employment": employment,
        "income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_history": credit_history,
    }])

    ml_prediction = pipeline.predict(application)[0]

    classes = list(pipeline.named_steps["model"].classes_)
    probability = pipeline.predict_proba(application)[0][classes.index("Approved")]

    limit = 7_000_000 if member.strip().lower() == "yes" else 4_000_000
    reasons = []
    if loan_amount > limit:
        reasons.append(f"Loan amount exceeds the limit of UGX {limit:,.0f}.")
    if credit_history.strip().lower() != "good":
        reasons.append("Credit history is not Good.")

    final = "Approved" if ml_prediction == "Approved" and not reasons else "Rejected"
    return {
        "ml_prediction": ml_prediction,
        "probability": probability,
        "limit": limit,
        "policy_reasons": reasons,
        "final_decision": final,
    }


def save_application(record):
    row = pd.DataFrame([record])
    if os.path.exists(APPLICATION_FILE):
        row.to_csv(APPLICATION_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(APPLICATION_FILE, index=False)


st.title("💰 Loan Approval Prediction")
st.caption("Machine learning prediction + SACCO policy checks")

if not os.path.exists(MODEL_FILE):
    st.error(f"'{MODEL_FILE}' not found. Put it in the same folder as app.py.")
    st.stop()

artifact = load_model()
pipeline = artifact["pipeline"]

with st.form("loan_form"):
    national_id = st.text_input("Applicant National ID")
    member = st.selectbox("Member of the SACCO?", ["Yes", "No"])
    tel_no = st.text_input("Telephone number")
    employment = st.selectbox("Employment status",
                              ["Employed", "Self-employed", "Unemployed"])
    income = st.number_input("Monthly income (UGX)", min_value=0.0,
                             step=50_000.0, format="%.0f")
    loan_amount = st.number_input("Loan amount (UGX)", min_value=0.0,
                                  step=100_000.0, format="%.0f")
    loan_term = st.number_input("Loan term (months)", min_value=1, value=12, step=1)
    credit_history = st.selectbox("Credit history", ["Good", "Poor"])

    st.markdown("**Co-applicant**")
    co_national_id = st.text_input("Co-applicant National ID")
    co_tel_no = st.text_input("Co-applicant telephone number")

    terms = st.checkbox("I accept the Terms & Conditions")
    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    if not national_id.strip():
        st.warning("Please enter the applicant National ID.")
    elif not tel_no.strip():
        st.warning("Please enter the applicant telephone number.")
    elif income <= 0:
        st.warning("Income must be greater than zero.")
    elif loan_amount <= 0:
        st.warning("Loan amount must be greater than zero.")
    elif not terms:
        st.warning("Please accept the Terms & Conditions.")
    else:
        result = predict_new_loan(pipeline, member, employment, income,
                                  loan_amount, int(loan_term), credit_history)

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("ML prediction", result["ml_prediction"])
        col2.metric("Approval probability", f"{result['probability']:.1%}")
        st.write(f"Applicable loan limit: **UGX {result['limit']:,.0f}**")

        if result["policy_reasons"]:
            st.write("**Policy checks failed:**")
            for reason in result["policy_reasons"]:
                st.write(f"- {reason}")
        else:
            st.write("Policy checks: **passed**")

        if result["final_decision"] == "Approved":
            st.success("✅ LOAN APPROVED")
        else:
            st.error("❌ LOAN REJECTED")

        save_application({
            "national_id": national_id, "member": member, "tel_no": tel_no,
            "employment": employment, "income": income,
            "loan_amount": loan_amount, "loan_term": int(loan_term),
            "credit_history": credit_history,
            "co_national_id": co_national_id, "co_tel_no": co_tel_no,
            "terms_accepted": terms,
            "ml_prediction": result["ml_prediction"],
            "ml_approved_probability": result["probability"],
            "final_decision": result["final_decision"],
        })
