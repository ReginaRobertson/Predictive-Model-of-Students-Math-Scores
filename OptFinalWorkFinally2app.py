import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "joblib"], check=False)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# ------------------------------
# Page Config & Header
# ------------------------------
st.set_page_config(page_title="🎓 Student Math Score Predictor", page_icon="📘", layout="wide")

# --- Sidebar Navigation ---
menu = st.sidebar.radio(
    "📂 Menu",
    ["🏠 Home", "🎯 Prediction", "ℹ️ About App"]
)

st.title("📘 Student Final Math Score Prediction - Ghanaian Junior High School")

# ------------------------------
# Home Section
# ------------------------------
if menu == "🏠 Home":
    st.markdown("<h2 style='text-align:center; color:skyblue;'>🌟 Welcome to the Student Final Math Score Prediction App 🌟</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Let's explore how Machine Learning can help you understand and improve student performance.</p>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=180)
    st.markdown("""
    ### 💡 What You Can Do Here
    - Predict final math scores using real student data  
    - Identify at-risk students early  
    - Explore how academic factors influence outcomes  
    - Download results for further analysis  
    """)
    st.success("👉 Use the sidebar to go to **Prediction** or **About App** sections.")

# ------------------------------
# Prediction Section
# ------------------------------
elif menu == "🎯 Prediction":
    st.sidebar.header("🧮 Prediction Settings")
    mode = st.sidebar.radio("Select Input Mode", ["One Variable per Student", "All Variables per Student"])
    n_students = st.sidebar.number_input("Input number of students to predict for", min_value=1, max_value=100, value=1, step=1)

    # --- Load model and dataset ---
    try:
        model = joblib.load('catboost_model.pkl')
        df = pd.read_excel('NEW DATA OF STUDENTS OF VRA JHS NO. 2.xlsx')
    except FileNotFoundError:
        st.error("❌ Model or data file not found. Please upload 'catboost_model.pkl' and the dataset.")
        st.stop()

    # --- Feature name check ---
    if hasattr(model, "feature_names_") and model.feature_names_:
        feature_names = model.feature_names_
    else:
        st.warning("⚠️ Model feature names not found — using dataset column names.")
        feature_names = df.drop(columns=["Final Math Score"], errors="ignore").columns.tolist()

    # --- Feature mapping ---
    features = {
        "1": "Age",
        "2": "Hours of studies per week",
        "3": "Attendance",
        "4": "Homework completion (20%)",
        "5": "Class Assessment Task (20%)",
        "6": "Class participation"
    }

    # --- Collect Student Data ---
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
                    student[selected_feature] = st.number_input("Age", 5, 25, step=1, key=f"age_{i}")
                elif selected_feature in ["Homework completion (20%)", "Class Assessment Task (20%)"]:
                    student[selected_feature] = st.number_input(f"{selected_feature} (0–20)", 0.0, 20.0, step=0.1, key=f"{selected_feature}_{i}")
                elif selected_feature == "Attendance":
                    student[selected_feature] = st.number_input("Attendance (10+)", 10, step=1, key=f"att_{i}")
                elif selected_feature == "Hours of studies per week":
                    student[selected_feature] = st.number_input("Hours of studies per week", 0.0, step=0.1, key=f"hours_{i}")
            else:
                student["Age"] = st.number_input("Age", 5, 25, step=1, key=f"age_{i}")
                student["Hours of studies per week"] = st.number_input("Hours of studies per week", 0.0, step=0.1, key=f"hours_{i}")
                student["Attendance"] = st.number_input("Attendance (10+)", 10, step=1, key=f"att_{i}")
                student["Homework completion (20%)"] = st.number_input("Homework completion (0–20)", 0.0, 20.0, step=0.1, key=f"home_{i}")
                student["Class Assessment Task (20%)"] = st.number_input("Class Assessment Task (0–20)", 0.0, 20.0, step=0.1, key=f"assess_{i}")
                student["Class participation"] = st.selectbox("Class participation", ["Low", "Moderate", "High"], key=f"part_{i}")
            students.append(student)
        return students

    # --- Collect Data ---
    student_data = collect_student_data(int(n_students), all_vars=(mode == "All Variables per Student"))

    # --- Prediction Logic ---
    if st.button("🎯 Predict"):
        try:
            input_df = pd.DataFrame(student_data)
            input_df['attendance_rate_percent'] = input_df['Attendance'] / 71
            input_df['homework_ratio'] = input_df['Homework completion (20%)'] / 20
            input_df['assessment_ratio'] = input_df['Class Assessment Task (20%)'] / 20
            input_df['study_efficiency'] = input_df['Hours of studies per week'] / (input_df['Homework completion (20%)'] + 1)
            input_df['attendance_x_homework'] = input_df['attendance_rate_percent'] * input_df['homework_ratio']
            input_df['attendance_x_assessment'] = input_df['attendance_rate_percent'] * input_df['assessment_ratio']
            input_df['study_x_homework'] = input_df['study_efficiency'] * input_df['homework_ratio']

            # One-hot encode
            input_df = pd.get_dummies(input_df)
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_names]

            # Predict and scale 10–100
            preds = model.predict(input_df)
            preds = np.clip(preds, 10, 100)

            # Risk classification
            risk_labels = ["⚠️ At-Risk" if score < 50 else "✅ Not At-Risk" for score in preds]

            # Display Results
            results = pd.DataFrame({
                "Student": [f"Student {i+1}" for i in range(len(preds))],
                "Predicted Final Math Score (10–100)": preds,
                "Risk Category": risk_labels
            })

            tab1, tab2 = st.tabs(["🔢 Predictions", "📈 Visual Insights"])
            with tab1:
                st.dataframe(results.style.format({"Predicted Final Math Score (10–100)": "{:.2f}"}))
            with tab2:
                st.subheader("📊 Predicted Math Scores per Student")
                chart = (
                    alt.Chart(results)
                    .mark_bar()
                    .encode(
                        x=alt.X("Student", sort=None, axis=alt.Axis(labelAngle=0, title="Students")),
                        y=alt.Y("Predicted Final Math Score (10–100)", title="Score"),
                        color=alt.Color("Risk Category", legend=alt.Legend(title="Risk Category"),
                                        scale=alt.Scale(domain=["⚠️ At-Risk", "✅ Not At-Risk"],
                                                        range=["#e74c3c", "#2ecc71"]))
                    )
                    .properties(width=500, height=400)
                )
                st.altair_chart(chart, use_container_width=True)

            avg_score = np.mean(preds)
            at_risk = np.sum(preds < 50)
            if at_risk > 0:
                message = "One student needs support." if at_risk == 1 else f"{at_risk} students need support."
                st.info(message)

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", csv, "prediction_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ------------------------------
# About Section
# ------------------------------
elif menu == "ℹ️ About App":
    st.header("About This App")
    st.write("""
    This app was developed as part of a **Machine Learning Capstone Project** by **Regina Robertson (2025)**.
    It leverages the CatBoost algorithm to predict math performance for Ghanaian Junior High School students based on:
    - Age  
    - Hours of study per week  
    - Attendance  
    - Homework completion (20%)  
    - Class assessment task (20%)  
    - Class participation (Low, Moderate, High) 
    """)
    st.info("For inquiries or improvements, contact: **reginarobertson91@gmail.com**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Developed by <b>Regina Robertson</b> | Capstone Project © 2025</p>", unsafe_allow_html=True)
