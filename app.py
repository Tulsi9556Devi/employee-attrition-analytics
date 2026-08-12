import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

model = joblib.load(
    "attrition_model.pkl"
)


# =========================================================
# MODEL EVALUATION
# =========================================================

features = [
    "Age",
    "MonthlyIncome",
    "OverTime",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "WorkLifeBalance",
    "StockOptionLevel",
    "JobRole",
    "MaritalStatus",
    "BusinessTravel"
]

X = df[features]

y = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# =========================================================
# BASIC DATA METRICS
# =========================================================

total = len(df)

left = (
    df["Attrition"] == "Yes"
).sum()

attrition_rate = (
    left / total
) * 100

avg_income = df["MonthlyIncome"].mean()


# =========================================================
# HEADER
# =========================================================

st.title("📊 Employee Attrition Analytics")

st.caption(
    "Data-driven employee attrition analysis "
    "and machine learning risk prediction"
)


# =========================================================
# TOP METRICS
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Employees",
    f"{total:,}"
)

c2.metric(
    "Employees Left",
    f"{left:,}"
)

c3.metric(
    "Attrition Rate",
    f"{attrition_rate:.1f}%"
)

c4.metric(
    "Model Accuracy",
    f"{accuracy * 100:.1f}%"
)

c5.metric(
    "ROC-AUC",
    f"{roc_auc:.2f}"
)


st.divider()


# =========================================================
# WORKFORCE INSIGHTS
# =========================================================

st.subheader("🔎 Workforce Insights")


# ---------- Row 1 ----------

c1, c2 = st.columns(2)


with c1:

    fig = px.histogram(
        df,
        x="OverTime",
        color="Attrition",
        barmode="group",
        title="Attrition vs Overtime"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with c2:

    fig = px.histogram(
        df,
        x="JobSatisfaction",
        color="Attrition",
        barmode="group",
        title="Attrition vs Job Satisfaction"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------- Row 2 ----------

c1, c2 = st.columns(2)


with c1:

    department_data = (
        df.groupby(
            ["Department", "Attrition"]
        )
        .size()
        .reset_index(
            name="Employees"
        )
    )

    fig = px.bar(
        department_data,
        x="Department",
        y="Employees",
        color="Attrition",
        barmode="group",
        title="Attrition by Department"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with c2:

    fig = px.box(
        df,
        x="Attrition",
        y="MonthlyIncome",
        title="Income vs Attrition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# KEY FINDING
# =========================================================

st.subheader("💡 Key Finding")


overtime_attrition = (
    df[df["OverTime"] == "Yes"]["Attrition"]
    .value_counts(normalize=True)
    .get("Yes", 0)
    * 100
)


normal_attrition = (
    df[df["OverTime"] == "No"]["Attrition"]
    .value_counts(normalize=True)
    .get("Yes", 0)
    * 100
)


st.info(
    f"Employees working overtime show an attrition rate "
    f"of approximately **{overtime_attrition:.1f}%**, "
    f"compared with **{normal_attrition:.1f}%** for employees "
    f"without overtime."
)


# =========================================================
# PREDICTION SECTION
# =========================================================

st.divider()

st.header("🤖 Attrition Risk Prediction")

st.write(
    "Enter an employee profile to estimate attrition risk."
)


# =========================================================
# INPUTS
# =========================================================

c1, c2, c3 = st.columns(3)


# ---------- Column 1 ----------

with c1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=50000,
        value=5000
    )

    years_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=50,
        value=5
    )

    years_role = st.number_input(
        "Years in Current Role",
        min_value=0,
        max_value=20,
        value=3
    )


# ---------- Column 2 ----------

with c2:

    overtime = st.selectbox(
        "Overtime",
        ["Yes", "No"]
    )

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [1, 2, 3, 4]
    )

    environment_satisfaction = st.selectbox(
        "Environment Satisfaction",
        [1, 2, 3, 4]
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [1, 2, 3, 4]
    )


# ---------- Column 3 ----------

with c3:

    job_involvement = st.selectbox(
        "Job Involvement",
        [1, 2, 3, 4]
    )

    job_level = st.selectbox(
        "Job Level",
        [1, 2, 3, 4, 5]
    )

    stock_option = st.selectbox(
        "Stock Option Level",
        [0, 1, 2, 3]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


# ---------- Additional Inputs ----------

job_role = st.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]
)


business_travel = st.selectbox(
    "Business Travel",
    [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if st.button(
    "🔮 Predict Attrition Risk",
    use_container_width=True
):

    employee = pd.DataFrame([{

        "Age": age,

        "MonthlyIncome": monthly_income,

        "OverTime": overtime,

        "JobSatisfaction": job_satisfaction,

        "EnvironmentSatisfaction":
            environment_satisfaction,

        "JobInvolvement":
            job_involvement,

        "JobLevel":
            job_level,

        "YearsAtCompany":
            years_company,

        "YearsInCurrentRole":
            years_role,

        "WorkLifeBalance":
            work_life_balance,

        "StockOptionLevel":
            stock_option,

        "JobRole":
            job_role,

        "MaritalStatus":
            marital_status,

        "BusinessTravel":
            business_travel

    }])


    probability = model.predict_proba(
        employee
    )[0][1]


    prediction = (
        "High Risk"
        if probability >= 0.5
        else "Low Risk"
    )


    st.divider()


    if prediction == "High Risk":

        st.error(
            f"### 🔴 High Attrition Risk\n\n"
            f"Estimated probability: "
            f"**{probability * 100:.1f}%**"
        )

    else:

        st.success(
            f"### 🟢 Low Attrition Risk\n\n"
            f"Estimated probability: "
            f"**{probability * 100:.1f}%**"
        )


    st.caption(
        "This is a statistical prediction based on "
        "patterns learned from the historical dataset; "
        "it is not a certainty."
    )