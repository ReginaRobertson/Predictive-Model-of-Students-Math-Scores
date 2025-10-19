import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------
# Page Config & Header
# ------------------------------
st.set_page_config(page_title="🎓 Student Math Score Predictor", page_icon="📘", layout="wide")
st.title("📘 Student Final Math Score Prediction - Ghanaian Junior High School")


# --- Welcome Message ---
st.markdown("<h2 style='text-align:center; color:skyblue;'>🌟 Welcome to the Student Final Math Score Prediction App 🌟</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:white;'>Let's explore how Machine Learning can help you understand and improve student performance.</p>", unsafe_allow_html=True)


with st.expander("ℹ️ About This App", expanded=False):
    st.write("""
    This tool helps teachers and education stakeholders **predict students’ final math scores** based on key factors such as attendance, homework completion, class assessment task, age, study hour per week
and class participation.
    It provides early warnings to guide timely intervention and support.
    """)

# --- Short Guidelines ---
st.markdown("""
### 📝 Quick Guide
1. Choose how to input data — one variable per student or all variables.  
2. Enter student information.  
3. Click **🎯 Predict** to view scores and risk categories.  
4. Download results as a CSV if needed.
""")

# ------------------------------
# Load model & data
# ------------------------------
try:
    model = joblib.load('catboost_model.pkl')
    df = pd.read_excel('NEW DATA OF STUDENTS OF VRA JHS NO. 2.xlsx')
except FileNotFoundError:
    st.error("❌ Model or data file not found. Ensure both 'catboost_model.pkl' and 'NEW DATA OF STUDENTS OF VRA JHS NO. 2.xlsx' exist in this directory.")
    st.stop()

# Try to extract feature names from model
try:
    feature_names = model.feature_names_
except Exception:
    st.error("⚠️ Could not extract feature names from model. Ensure your CatBoost model is saved with feature names.")
    st.stop()

# ------------------------------
# Sidebar Configuration
# ------------------------------
st.sidebar.header("🧮 Prediction Settings")
mode = st.sidebar.radio("Select Input Mode", ["One Variable per Student", "All Variables per Student"])
n_students = st.sidebar.number_input("Input the Number of Students you want to predict for", min_value=1, max_value=100, value=1, step=1)

# ------------------------------
# Feature Mapping
# ------------------------------
features = {
    "1": "Age",
    "2": "Hours of studies per week",
    "3": "Attendance",
    "4": "Homework completion (20%)",
    "5": "Class Assessment Task (20%)",
    "6": "Class participation"
}

# ------------------------------
# Data Collection Function
# ------------------------------
def collect_student_data(n_students, all_vars=False):
    students = []
    for i in range(n_students):
        st.subheader(f"🧑‍🎓 Student {i+1}")
        student = {}
        
        if not all_vars:
            selected_feature = st.selectbox(f"Select feature for Student {i+1}", list(features.values()), key=f"feature_{i}")
            student = {
                col: df[col].mode()[0] if df[col].dtype == "object" else df[col].mean()
                for col in features.values()
            }

            if selected_feature == "Class participation":
                student[selected_feature] = st.selectbox("Class participation", ["Low", "Moderate", "High"], key=f"part_{i}")
            elif selected_feature == "Age":
                student[selected_feature] = st.number_input("Age", min_value=5, max_value=25, step=1, key=f"age_{i}")
            elif selected_feature in ["Homework completion (20%)", "Class Assessment Task (20%)"]:
                student[selected_feature] = st.number_input(f"{selected_feature} (0–20)", min_value=0.0, max_value=20.0, step=0.1, key=f"{selected_feature}_{i}")
            elif selected_feature == "Attendance":
                student[selected_feature] = st.number_input("Attendance (10 and above)", min_value=10, step=1, key=f"att_{i}")
            elif selected_feature == "Hours of studies per week":
                student[selected_feature] = st.number_input("Hours of studies per week", min_value=0.0, step=0.1, key=f"hours_{i}")

        else:
            student["Age"] = st.number_input("Age", min_value=5, max_value=25, step=1, key=f"age_{i}")
            student["Hours of studies per week"] = st.number_input("Hours of studies per week", min_value=0.0, step=0.1, key=f"hours_{i}")
            student["Attendance"] = st.number_input("Attendance (10 and above)", min_value=10, step=1, key=f"att_{i}")
            student["Homework completion (20%)"] = st.number_input("Homework completion (0–20)", min_value=0.0, max_value=20.0, step=0.1, key=f"home_{i}")
            student["Class Assessment Task (20%)"] = st.number_input("Class Assessment Task (0–20)", min_value=0.0, max_value=20.0, step=0.1, key=f"assess_{i}")
            student["Class participation"] = st.selectbox("Class participation", ["Low", "Moderate", "High"], key=f"part_{i}")
        
        students.append(student)
    return students

# ------------------------------
# Collect Data
# ------------------------------
student_data = collect_student_data(int(n_students), all_vars=(mode == "All Variables per Student"))

# ------------------------------
# Prediction
# ------------------------------
if st.button("🎯 Predict"):
    try:
        input_df = pd.DataFrame(student_data)

        # --- Feature Engineering ---
        input_df['attendance_rate_percent'] = input_df['Attendance'] / 71
        input_df['homework_ratio'] = input_df['Homework completion (20%)'] / 20
        input_df['assessment_ratio'] = input_df['Class Assessment Task (20%)'] / 20
        input_df['study_efficiency'] = input_df['Hours of studies per week'] / (input_df['Homework completion (20%)'] + 1)
        input_df['attendance_x_homework'] = input_df['attendance_rate_percent'] * input_df['homework_ratio']
        input_df['attendance_x_assessment'] = input_df['attendance_rate_percent'] * input_df['assessment_ratio']
        input_df['study_x_homework'] = input_df['study_efficiency'] * input_df['homework_ratio']

        # --- Encoding ---
        input_df = pd.get_dummies(input_df)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]
        # --- Model Prediction ---
        preds = model.predict(input_df)
        risk_labels = ["⚠️ At-Risk" if score < 50 else "✅ Not At-Risk" for score in preds]

        results = pd.DataFrame({
            "Student": [f"Student {i+1}" for i in range(len(preds))],
            "Predicted Final Math Score": preds,
            "Risk Category": risk_labels
        })


        import altair as alt

        # --- Display Results in Tabs ---
        tab1, tab2 = st.tabs(["🔢 Predictions", "📈 Visual Insights"])

        with tab1:
            st.dataframe(results.style.format({"Predicted Final Math Score": "{:.2f}"}))

        with tab2:
            st.subheader("📊 Predicted Math Scores per Student")

            chart = (
        alt.Chart(results)
        .mark_bar()
        .encode(
            x=alt.X("Student", sort=None, axis=alt.Axis(labelAngle=0, title="Students")),
            y=alt.Y("Predicted Final Math Score", title="Score"),
            color=alt.Color(
                "Risk Category",
                legend=alt.Legend(title="Risk Category"),
                scale=alt.Scale(domain=["⚠️ At-Risk", "✅ Not At-Risk"],
                               range=["#e74c3c", "#2ecc71"])  # red & green
            ),
            tooltip=["Student", "Predicted Final Math Score", "Risk Category"]
        )
        .properties(width=500, height=400)
    )
            st.altair_chart(chart, use_container_width=True)


    
         # --- Summary Insights ---
        avg_score = np.mean(preds)
        at_risk = np.sum(preds < 50)
        
       
        # --- General Encouraging Message ---
        if at_risk > 0:
            message = (
                "The student needs support."
                if at_risk == 1
                else "The students need support."
            )
            st.info(message)

        # --- Download Results ---
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv, "prediction_results.csv", "text/csv")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Developed by <b>Regina Robertson</b> | Capstone Project © 2025</p>", unsafe_allow_html=True)

