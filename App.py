import streamlit as st
import pandas as pd
import pickle as pk

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="SBI Loan Predictor", page_icon="🏦", layout="centered")

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #F4F6F7;
}

/* Header */
h1, h2, h3 {
    color: #1C4E80;
    text-align: center;
    font-weight: bold;
}

/* Buttons */
.stButton>button {
    background-color: #1C4E80;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #2E86C1;
}

/* Input fields */
.stTextInput>div>div>input {
    border: 1px solid #1C4E80;
    border-radius: 6px;
}

/* Select box */
.stSelectbox>div>div {
    border: 1px solid #1C4E80;
    border-radius: 6px;
}

/* Success / Error */
.stSuccess {
    color: green;
}
.stError {
    color: red;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MAIN APP ---------------- #

model = pk.load(open("model.pkl", "rb"))
scaler = pk.load(open("scaler.pkl", "rb"))

st.title("SBI Loan Approval System")

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Applicant Details")

no_of_dep = st.slider("No of Dependents", 0, 5)
grad = st.selectbox("Education", ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
Annual_Income = st.slider('Annual Income', 0, 10000000)
Loan_Amount = st.slider('Loan Amount', 0, 100000000)
Loan_Dur = st.slider('Loan Duration (Years)', 0, 20)
Cibil = st.slider('CIBIL Score', 0, 1000)
Assets = st.slider('Assets', 0, 10000000)

grad_s = 1 if grad == 'Graduated' else 0
emp_s = 1 if self_emp == 'Yes' else 0

if st.button('Predict Loan Status'):
    pred_data = pd.DataFrame(
        [[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
        columns=[
            'no_of_dependents', 'education', 'self_employed',
            'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'Assets'
        ]
    )

    pred_data = scaler.transform(pred_data)

    prediction = model.predict(pred_data)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

st.markdown("</div>", unsafe_allow_html=True)