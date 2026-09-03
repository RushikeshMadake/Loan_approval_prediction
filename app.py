import streamlit as st
import pandas as pd
import pickle as pk

st.set_page_config(
    page_title="SBI Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #F5F9FF 0%,
        #E3F2FD 100%
    );
}


h1 {
    color: #1A237E !important;
    text-align: center;
    font-weight: 800;
    font-size: 38px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #455A64;
    font-size: 17px;
    margin-bottom: 25px;
}


h3 {
    color: #0D47A1 !important;
    font-weight: 700;
}


.sbi-line {
    height: 5px;
    background: #FFD54F;
    border-radius: 10px;
    margin: 10px 0 25px 0;
}

.form-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 5px 20px rgba(13, 71, 161, 0.15);
    border-top: 5px solid #1A237E;
    margin-bottom: 20px;
}


label {
    color: #263238 !important;
    font-weight: 600 !important;
}


.stSlider {
    padding-bottom: 10px;
}



.stSelectbox > div > div {
    border: 1px solid #90CAF9;
    border-radius: 8px;
}



.stButton > button {
    background: linear-gradient(
        90deg,
        #1A237E,
        #0D47A1
    );

    color: white;
    border-radius: 10px;
    height: 3.2em;
    width: 100%;
    font-size: 18px;
    font-weight: 700;
    border: none;

    box-shadow: 0px 4px 10px rgba(13, 71, 161, 0.3);
}


/* BUTTON HOVER */

.stButton > button:hover {
    background: #FFD54F;
    color: #1A237E;
    border: 2px solid #1A237E;
}



.approved-box {
    background-color: #E8F5E9;
    border-left: 6px solid #2E7D32;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    color: #2E7D32;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
}


.rejected-box {
    background-color: #FFEBEE;
    border-left: 6px solid #C62828;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    color: #C62828;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
}


.footer {
    text-align: center;
    color: #607D8B;
    font-size: 13px;
    margin-top: 30px;
    padding: 15px;
}


.info-box {
    background-color: #E3F2FD;
    border-left: 5px solid #1A237E;
    padding: 12px;
    border-radius: 8px;
    color: #263238;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


model = pk.load(open("model.pkl", "rb"))
scaler = pk.load(open("scaler.pkl", "rb"))


st.title("🏦 SBI Loan Approval System")

st.markdown(
    "<div class='subtitle'>"
    "Smart Loan Eligibility Prediction System"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sbi-line'></div>",
    unsafe_allow_html=True
)



st.markdown("""
<div class="info-box">
<b>💡 Enter Applicant Details</b><br>
Provide the applicant's financial and personal information
to predict the loan approval status.
</div>
""", unsafe_allow_html=True)


st.markdown("<div class='form-card'>", unsafe_allow_html=True)

st.subheader("👤 Applicant Details")


no_of_dep = st.slider(
    "No of Dependents",
    0,
    5,
    0
)

grad = st.selectbox(
    "Education",
    ["Graduated", "Not Graduated"]
)

self_emp = st.selectbox(
    "Self Employed?",
    ["Yes", "No"]
)

Annual_Income = st.slider(
    "Annual Income (₹)",
    0,
    10000000,
    500000,
    step=10000
)

Loan_Amount = st.slider(
    "Loan Amount (₹)",
    0,
    100000000,
    1000000,
    step=100000
)


Loan_Dur = st.slider(
    "Loan Duration (Years)",
    0,
    20,
    5
)


Cibil = st.slider(
    "CIBIL Score",
    0,
    1000,
    700
)


Assets = st.slider(
    "Assets (₹)",
    0,
    10000000,
    500000,
    step=10000
)



grad_s = 1 if grad == "Graduated" else 0
emp_s = 1 if self_emp == "Yes" else 0


st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🔍 Predict Loan Status")


st.markdown("</div>", unsafe_allow_html=True)


if predict:

    pred_data = pd.DataFrame(
        [[
            no_of_dep,
            grad_s,
            emp_s,
            Annual_Income,
            Loan_Amount,
            Loan_Dur,
            Cibil,
            Assets
        ]],
        columns=[
            "no_of_dependents",
            "education",
            "self_employed",
            "income_annum",
            "loan_amount",
            "loan_term",
            "cibil_score",
            "Assets"
        ]
    )


    pred_data = scaler.transform(pred_data)



    prediction = model.predict(pred_data)



    if prediction[0] == 1:

        st.markdown("""
        <div class="approved-box">
            ✅ LOAN APPROVED
            <br>
            <span style="font-size:15px;">
            Congratulations! The applicant is eligible for the loan.
            </span>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="rejected-box">
            ❌ LOAN REJECTED
            <br>
            <span style="font-size:15px;">
            The applicant does not meet the predicted loan eligibility criteria.
            </span>
        </div>
        """, unsafe_allow_html=True)



st.markdown("""
<div class="footer">
    🏦 SBI Loan Approval Prediction System
    <br>
    Powered by Machine Learning
</div>
""", unsafe_allow_html=True)
