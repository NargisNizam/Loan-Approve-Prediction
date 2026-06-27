import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main{
    background:#f4f7fc;
}

h1{
    color:#0B5ED7;
    text-align:center;
}

h2,h3{
    color:#0B5ED7;
}

.stButton>button{
    width:100%;
    background:#0B5ED7;
    color:white;
    border-radius:10px;
    font-size:18px;
    height:50px;
}

.stButton>button:hover{
    background:#084298;
    color:white;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 0px 12px rgba(0,0,0,.15);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🏦 Loan Approval Prediction System")

st.markdown(
"""
### Machine Learning Based Loan Prediction

This application predicts whether a customer is eligible for a loan using a Random Forest Machine Learning model.
"""
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏦 Loan Prediction")

st.sidebar.success("Model Ready")

st.sidebar.markdown("---")

st.sidebar.write("### Algorithm")
st.sidebar.write("Random Forest")

st.sidebar.write("### Dataset")
st.sidebar.write("Loan Approval Dataset")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Developed using

• Python

• Streamlit

• Scikit-Learn

• Pandas
"""
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("loan_data_new.csv")

# ==========================================
# LABEL ENCODING
# ==========================================

encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Loan Status", axis=1)
y = df["Loan Status"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

st.success("✅ Model Loaded Successfully")

st.metric(
    label="Model Accuracy",
    value=f"{accuracy*100:.2f}%"
)

st.markdown("---")

# ==========================================
# USER INPUT SECTION
# ==========================================

st.header("📝 Applicant Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    education = st.selectbox(
        "Education",
        ["High School", "Bachelor", "Master", "PhD"]
    )

    income = st.number_input(
        "Person Income",
        min_value=1000,
        max_value=1000000,
        value=50000
    )

    experience = st.number_input(
        "Employee Experience",
        min_value=0,
        max_value=40,
        value=2
    )

    home = st.selectbox(
        "Home Ownership",
        ["OWN", "RENT", "MORTGAGE", "OTHER"]
    )

with col2:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=500,
        max_value=1000000,
        value=20000
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        ["PERSONAL",
         "EDUCATION",
         "MEDICAL",
         "VENTURE",
         "HOMEIMPROVEMENT",
         "DEBTCONSOLIDATION"]
    )

    interest = st.number_input(
        "Interest Rate",
        min_value=1.0,
        max_value=30.0,
        value=10.5
    )

    loan_percent = st.slider(
        "Loan Percentage",
        0.0,
        1.0,
        0.30
    )

    credit_history = st.number_input(
        "Credit History",
        min_value=0,
        max_value=30,
        value=5
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    previous_loan = st.selectbox(
        "Previous Loan",
        ["Yes", "No"]
    )

# ==========================================
# ENCODE USER INPUT
# ==========================================

gender = encoders["Gender"].transform([gender])[0]
education = encoders["Education"].transform([education])[0]
home = encoders["Home Onwership"].transform([home])[0]
loan_intent = encoders["Loan Intent"].transform([loan_intent])[0]
previous_loan = encoders["Previous Loan"].transform([previous_loan])[0]

st.markdown("")

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button(" Predict Loan Status"):

    user_data = [[
        age,
        gender,
        education,
        income,
        experience,
        home,
        loan_amount,
        loan_intent,
        interest,
        loan_percent,
        credit_history,
        credit_score,
        previous_loan
    ]]

    result = model.predict(user_data)

    st.markdown("---")

    if result[0] == 1:

        st.success(" Congratulations! Loan Approved")

        st.balloons()

    else:

        st.error(" Sorry! Loan Rejected")

        # ==========================================
# DATASET PREVIEW
# ==========================================

st.markdown("---")

st.subheader("📋 Dataset Preview")

st.dataframe(df.head())

# ==========================================
# CHARTS
# ==========================================

st.markdown("---")

st.subheader("📊 Loan Status Distribution")

fig, ax = plt.subplots(figsize=(6,4))

df["Loan Status"].value_counts().plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Loan Status")
ax.set_ylabel("Count")

st.pyplot(fig)

# ==========================================
# PROJECT SUMMARY
# ==========================================

st.markdown("---")

st.subheader("📌 Project Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset Records", len(df))
col2.metric("Features", X.shape[1])
col3.metric("Accuracy", f"{accuracy*100:.2f}%")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center; color:gray;'>

### 🏦 Loan Approval Prediction System

Developed using ❤️ with Python, Streamlit & Machine Learning

<b>Developed By: Muhammad Taqi</b>

</div>
""",
unsafe_allow_html=True
)