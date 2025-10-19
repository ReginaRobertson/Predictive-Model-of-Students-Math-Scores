
# OptFinalWorkStableApp.py
import subprocess
import sys

# Runtime safety: try to ensure joblib is available (helps on some cloud builds)
try:
    import joblib
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "joblib"], check=False)
    import joblib

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="🎓 Student Math Score Predictor", page_icon="📘", layout="wide")

# -------------------------------
# Helper functions
# -------------------------------
def soft_global_scale(preds, target_min=10, target_max=100):
    preds = np.array(preds).astype(float).copy()
    p_min, p_max = preds.min(), preds.max()
    if p_max > p_min:
        preds = target_min + (preds - p_min) * (target_max - target_min) / (p_max - p_min)
    else:
        # all equal predictions -> map to midpoint
        preds = np.full_like(preds, (target_min + target_max) / 2.0)
    return preds

# Fixed canonical feature list used during training (engineered + dummies)
CANONICAL_FEATURES = [
    'Class Assessment Task (20%)', 'Homework completion (20%)', 'Hours of studies per week', 'Attendance', 'Age',
    'attendance_rate_percent', 'homework_ratio', 'assessment_ratio', 'study_efficiency',
    'attendance_x_homework', 'attendance_x_assessment', 'study_x_homework',
    'Class participation_High', 'Class participation_Moderate'
]

# -------------------------------
# UI layout
# -------------------------------
menu = st.sidebar.radio("Menu", ["Home", "Prediction", "About"])
st.title("📘 Student Final Math Score Prediction — Ghanaian Junior High School")

if menu == "Home":
    st.markdown("### Welcome")
    st.markdown("Use the **Prediction** page to enter student details and get a predicted final math score (displayed on a 10–100 scale).")
    st.info("Make sure `catboost_model.pkl` and (optionally) `scaler.pkl` are in the app folder. If you use a dataset for defaults, put it in the same folder too.")

elif menu == "Prediction":
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("Input mode", ["One Variable per Student", "All Variables per Student"])
    n_students = st.sidebar.number_input("Number of students", min_value=1, max_value=100, value=1, step=1)

    # --- Try to load model and optional scaler / dataset ---
    model = None
    scaler = None
    df_defaults = None

    # Load model
    try:
        model = joblib.load("optimized_catboost_model.pkl")
    except FileNotFoundError:
        st.error("Model file 'optimized_catboost_model.pkl' not found in app directory. Upload it and refresh.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Try optional scaler
    if os.path.exists("scaler.pkl"):
        try:
            scaler = joblib.load("scaler.pkl")
        except Exception:
            scaler = None
            st.warning("Found 'scaler.pkl' but failed to load it. Predictions will proceed without scaling.")

    # Try optional dataset for defaults (not required)
    if os.path.exists("NEW DATA OF STUDENTS OF VRA JHS NO. 2.xlsx"):
        try:
            df_defaults = pd.read_excel("NEW DATA OF STUDENTS OF VRA JHS NO. 2.xlsx")
        except Exception:
            df_defaults = None

    # Try to get feature names from the model, else use canonical
    feature_names_from_model = getattr(model, "feature_names_", None)
    if feature_names_from_model and len(feature_names_from_model) >= 1:
        FEATURE_ORDER = list(feature_names_from_model)
    else:
        # fallback to canonical expected features
        FEATURE_ORDER = list(CANONICAL_FEATURES)
        st.info("Model feature names not found — using canonical feature list for alignment.")

    # UI: collect student inputs
    st.header("Enter student details")
    features = {
        "1": "Age",
        "2": "Hours of studies per week",
        "3": "Attendance",
        "4": "Homework completion (20%)",
        "5": "Class Assessment Task (20%)",
        "6": "Class participation (Low, Moderate, High)"
    }

    def get_default(col):
        if df_defaults is None:
            # reasonable defaults
            defaults = {
                "Age": 14,
                "Hours of studies per week": 5.0,
                "Attendance": 60,
                "Homework completion (20%)": 15.0,
                "Class Assessment Task (20%)": 15.0,
                "Class participation": "Moderate"
            }
            return defaults.get(col)
        else:
            if col in df_defaults.columns:
                if df_defaults[col].dtype == "object":
                    return df_defaults[col].mode().iloc[0]
                else:
                    return float(df_defaults[col].mean())
            else:
                return None

    students_list = []
    for i in range(int(n_students)):
        st.subheader(f"Student {i+1}")
        student = {}
        if mode == "One Variable per Student":
            chosen = st.selectbox(f"Select variable to set for Student {i+1}", list(features.values()), key=f"choice_{i}")
            # initialize with defaults
            for col in features.values():
                student[col] = get_default(col)
            if chosen == "Class participation":
                student[chosen] = st.selectbox(f"Class participation (Student {i+1})", ["Low", "Moderate", "High"], key=f"part_{i}")
            elif chosen == "Age":
                student["Age"] = st.number_input(f"Age (Student {i+1})", min_value=5, max_value=25, value=int(get_default("Age")), key=f"age_{i}")
            elif chosen in ["Homework completion (20%)", "Class Assessment Task (20%)"]:
                student[chosen] = st.number_input(f"{chosen} (0–20) (Student {i+1})", min_value=0.0, max_value=20.0, value=float(get_default(chosen)), step=0.1, key=f"{chosen}_{i}")
            elif chosen == "Attendance":
                student["Attendance"] = st.number_input(f"Attendance (10+) (Student {i+1})", min_value=10, max_value=100, value=int(get_default("Attendance") or 60), key=f"att_{i}")
            elif chosen == "Hours of studies per week":
                student["Hours of studies per week"] = st.number_input(f"Hours of studies per week (Student {i+1})", min_value=0.0, value=float(get_default("Hours of studies per week") or 5.0), step=0.1, key=f"hours_{i}")
        else:
            # all vars
            student["Age"] = st.number_input(f"Age (Student {i+1})", min_value=5, max_value=25, value=int(get_default("Age") or 14), key=f"age_all_{i}")
            student["Hours of studies per week"] = st.number_input(f"Hours of studies per week (Student {i+1})", min_value=0.0, value=float(get_default("Hours of studies per week") or 5.0), step=0.1, key=f"hours_all_{i}")
            student["Attendance"] = st.number_input(f"Attendance (10+) (Student {i+1})", min_value=10, max_value=100, value=int(get_default("Attendance") or 60), key=f"att_all_{i}")
            student["Homework completion (20%)"] = st.number_input(f"Homework completion (0–20) (Student {i+1})", min_value=0.0, max_value=20.0, value=float(get_default("Homework completion (20%)") or 15.0), step=0.1, key=f"home_all_{i}")
            student["Class Assessment Task (20%)"] = st.number_input(f"Class Assessment Task (0–20) (Student {i+1})", min_value=0.0, max_value=20.0, value=float(get_default("Class Assessment Task (20%)") or 15.0), step=0.1, key=f"assess_all_{i}")
            student["Class participation"] = st.selectbox(f"Class participation (Student {i+1})", ["Low", "Moderate", "High"], index=1, key=f"part_all_{i}")
        students_list.append(student)

    # Predict button
    if st.button("Predict"):
        try:
            input_df = pd.DataFrame(students_list)

            # Ensure required columns exist (fill missing with defaults)
            for c in ["Age", "Hours of studies per week", "Attendance", "Homework completion (20%)", "Class Assessment Task (20%)", "Class participation"]:
                if c not in input_df.columns:
                    input_df[c] = get_default(c)

            # Feature engineering
            input_df['attendance_rate_percent'] = input_df['Attendance'] / 71
            input_df['homework_ratio'] = input_df['Homework completion (20%)'] / 20
            input_df['assessment_ratio'] = input_df['Class Assessment Task (20%)'] / 20
            input_df['study_efficiency'] = input_df['Hours of studies per week'] / (input_df['Homework completion (20%)'] + 1)
            input_df['attendance_x_homework'] = input_df['attendance_rate_percent'] * input_df['homework_ratio']
            input_df['attendance_x_assessment'] = input_df['attendance_rate_percent'] * input_df['assessment_ratio']
            input_df['study_x_homework'] = input_df['study_efficiency'] * input_df['homework_ratio']

            # One-hot encode participation (ensure columns exist)
            input_df = pd.get_dummies(input_df)
            # Guarantee presence of the participation dummies
            for col in ["Class participation_High", "Class participation_Moderate"]:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Align to FEATURE_ORDER (fallback)
            for col in FEATURE_ORDER:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Select columns in correct order
            input_aligned = input_df[FEATURE_ORDER].copy()

            # Apply scaler if available
            if scaler is not None:
                try:
                    input_transformed = scaler.transform(input_aligned)
                except Exception:
                    st.warning("Scaler exists but failed to transform inputs — proceeding without scaler.")
                    input_transformed = input_aligned.values
            else:
                input_transformed = input_aligned.values

            # Predict raw
            raw_preds = model.predict(input_transformed).astype(float)

            # Displayed prediction: rescaled 10-100 for UI
            displayed_preds = soft_global_scale(raw_preds, target_min=10, target_max=100)

            # Prepare results (only show displayed score to user)
            results = pd.DataFrame({
                "Student": [f"Student {i+1}" for i in range(len(displayed_preds))],
                "Predicted Final Math Score (10–100)": np.round(displayed_preds, 2),
            })

            # Optionally include risk based on raw prediction internally (but display only the displayed score)
            risk = ["⚠️ At-Risk" if r < 50 else "✅ Not At-Risk" for r in raw_preds]
            results["Risk Category"] = risk

            st.success("Prediction complete.")
            st.dataframe(results[["Student", "Predicted Final Math Score (10–100)"]].style.format({"Predicted Final Math Score (10–100)": "{:.2f}"}))

            # Visual
            try:
                import altair as alt
                chart = (
                    alt.Chart(results)
                    .mark_bar()
                    .encode(
                        x=alt.X("Student", sort=None),
                        y=alt.Y("Predicted Final Math Score (10–100)"),
                        color=alt.Color("Risk Category", scale=alt.Scale(domain=["⚠️ At-Risk", "✅ Not At-Risk"], range=["#e74c3c", "#2ecc71"]))
                    )
                    .properties(height=400, width="container")
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                pass

            # Download
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("Download results (CSV)", csv, "prediction_results.csv", "text/csv")

        except Exception as e:
            st.exception(f"Prediction failed: {e}")

elif menu == "About":
    st.header("About")
    st.write("""
    CatBoost-based student math score predictor.
    Developer: Regina Robertson (2025)
    """)

st.markdown("---")
st.caption("If you trained the model locally, consider saving the scaler (scaler.pkl) and the model with feature names to avoid any alignment warnings.")
