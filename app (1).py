import streamlit as st
import joblib
import pandas as pd


# ---------------------------------------------------------
# LOAD MODEL AND SCALER
# ---------------------------------------------------------

dt_model = joblib.load(r"D:\\PYTHON\\ML_Project\\dt_model.pkl")
scaler = joblib.load(r"D:\\PYTHON\\ML_Project\\scaler.pkl")


# ---------------------------------------------------------
# STREAMLIT PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Campus Placement Prediction",
    page_icon="🎓"
)

st.title("🎓 Campus Placement Prediction")
st.write("Enter student details to predict campus placement eligibility.")


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

cgpa = st.number_input(
    "Enter your CGPA:",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.01
)

iq = st.number_input(
    "Enter your IQ:",
    min_value=0,
    max_value=200,
    value=100,
    step=1
)

prev_sem_results = st.number_input(
    "Enter your previous semester results:",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.01
)

academic_performance = st.number_input(
    "Enter your academic performance:",
    min_value=0,
    max_value=10,
    value=7,
    step=1
)

extra_curricular_activities = st.number_input(
    "Enter your extra-curricular score:",
    min_value=0,
    max_value=10,
    value=7,
    step=1
)

communication_skills = st.number_input(
    "Enter your communication skills:",
    min_value=0,
    max_value=10,
    value=7,
    step=1
)

project_completion = st.number_input(
    "Enter your project completion:",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

internship_experience = st.selectbox(
    "Internship Experience:",
    ["No", "Yes"]
)


# ---------------------------------------------------------
# CONVERT INTERNSHIP EXPERIENCE
# ---------------------------------------------------------

if internship_experience == "Yes":
    internship_experience = 1
else:
    internship_experience = 0


# ---------------------------------------------------------
# FEATURE ENGINEERING
# Same formulas as train.py
# ---------------------------------------------------------

profile_score = (
    academic_performance
    + extra_curricular_activities
    + communication_skills
    + project_completion
)

iq_vs_cgpa = iq * cgpa

interview_readiness = profile_score + iq_vs_cgpa

total_merit = (
    profile_score
    + iq_vs_cgpa
    + interview_readiness
)


# ---------------------------------------------------------
# CREATE INPUT DATA
# EXACT SAME ORDER AS TRAINING DATA
# ---------------------------------------------------------

input_data = pd.DataFrame({
    "IQ": [iq],
    "Prev_Sem_Result": [prev_sem_results],
    "CGPA": [cgpa],
    "Academic_Performance": [academic_performance],
    "Internship_Experience": [internship_experience],
    "Extra_Curricular_Score": [extra_curricular_activities],
    "Communication_Skills": [communication_skills],
    "Projects_Completed": [project_completion],
    "profile_score": [profile_score],
    "IQ_VS_CGPA": [iq_vs_cgpa],
    "Interview_Readiness": [interview_readiness],
    "Total_Merit": [total_merit]
})


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if st.button("🔮 Predict Eligibility"):

    try:

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = dt_model.predict(input_scaled)

        # Result
        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:

            st.success(
                "🎉 Congratulations! You are eligible for the internship."
            )

        else:

            st.error(
                "❌ Sorry! You are not eligible for the internship."
            )

        # Display calculated features
        st.subheader("📈 Student Profile")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Profile Score:**", profile_score)
            st.write("**IQ vs CGPA:**", iq_vs_cgpa)

        with col2:
            st.write("**Interview Readiness:**", interview_readiness)
            st.write("**Total Merit:**", total_merit)

    except Exception as e:

        st.error("Prediction Error")
        st.exception(e)