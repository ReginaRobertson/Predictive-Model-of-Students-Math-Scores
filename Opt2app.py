import streamlit as st
import pandas as pd
import joblib

# Load model and data
try:
    model = joblib.load('catboost_model.pkl')
    df = pd.read_excel('NEW DATA OF STUDENTS OF VRA JHS NO. 2.xlsx')
except FileNotFoundError:
    st.error("Model or data file not found. Please ensure 'catboost_model.pkl' and 'NEW DATA OF STUDENTS OF VRA JHS NO. 2.xlsx' are in the same directory.")
    st.stop()

# Get feature names directly from model to ensure correct order
try:
    feature_names = model.feature_names_
except Exception:
    st.error("Could not extract feature names from model. Please ensure the CatBoost model is trained and saved with feature names.")
    st.stop()

# Streamlit app title
st.title("Student Final Math Score Prediction")

# Sidebar for prediction mode selection
st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio(
    "Select input mode:",
    ["One Variable per Student", "All Variables per Student"]
)

# Input for number of students
n_students = st.number_input(
    "How many students to predict for?",
    min_value=1,
    max_value=100,
    value=1,
    step=1
)

# Features dictionary
features = {
    "1": "Age",
    "2": "Hours of studies per week",
    "3": "Attendance",
    "4": "Homework completion (20%)",
    "5": "Class Assessment Task (20%)",
    "6": "Class participation"
}

# Function to collect student data
def collect_student_data(n_students, all_vars=False):
    all_students = []
    
    for i in range(n_students):
        st.subheader(f"Student {i+1}")
        student_data = {}
        
        if not all_vars:
            # One variable mode
            selected_feature = st.selectbox(
                f"Select feature for Student {i+1}",
                options=list(features.values()),
                key=f"feature_{i}"
            )
            student_data = {
                col: df[col].mode()[0] if df[col].dtype == "object" else df[col].mean()
                for col in features.values()
            }
            
            if selected_feature == "Class participation":
                value = st.selectbox(
                    "Class participation",
                    options=["Low", "Moderate", "High"],
                    key=f"participation_{i}"
                )
                student_data[selected_feature] = value
                
            elif selected_feature == "Age":
                value = st.number_input(
                    "Age",
                    min_value=5,
                    max_value=25,
                    step=1,
                    key=f"age_{i}"
                )
                student_data[selected_feature] = int(value)
                
            elif selected_feature in ["Homework completion (20%)", "Class Assessment Task (20%)"]:
                value = st.number_input(
                    f"{selected_feature} (0-20)",
                    min_value=0.0,
                    max_value=20.0,
                    step=0.1,
                    key=f"{selected_feature}_{i}"
                )
                student_data[selected_feature] = value
                
            elif selected_feature == "Attendance":
                value = st.number_input(
                    "Attendance (10 and above)",
                    min_value=10,
                    step=1,
                    key=f"attendance_{i}"
                )
                student_data[selected_feature] = int(value)
                
            elif selected_feature == "Hours of studies per week":
                value = st.number_input(
                    "Hours of studies per week (0 and above)",
                    min_value=0.0,
                    step=0.1,
                    key=f"hours_{i}"
                )
                student_data[selected_feature] = value
                
        else:
            # All variables mode
            student_data["Age"] = st.number_input(
                "Age",
                min_value=5,
                max_value=25,
                step=1,
                key=f"age_{i}"
            )
            student_data["Age"] = int(student_data["Age"])
            
            student_data["Hours of studies per week"] = st.number_input(
                "Hours of studies per week (0 and above)",
                min_value=0.0,
                step=0.1,
                key=f"hours_{i}"
            )
            student_data["Attendance"] = st.number_input(
                "Attendance (10 and above)",
                min_value=10,
                step=1,
                key=f"attendance_{i}"
            )
            student_data["Attendance"] = int(student_data["Attendance"])
            
            student_data["Homework completion (20%)"] = st.number_input(
                "Homework completion (0-20)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                key=f"homework_{i}"
            )
            student_data["Class Assessment Task (20%)"] = st.number_input(
                "Class Assessment Task (0-20)",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                key=f"assessment_{i}"
            )
            student_data["Class participation"] = st.selectbox(
                "Class participation",
                options=["Low", "Moderate", "High"],
                key=f"participation_{i}"
            )
        
        all_students.append(student_data)
    
    return all_students

# Collect data based on mode
all_students = collect_student_data(int(n_students), all_vars=(mode == "All Variables per Student"))

# Predict button
if st.button("Predict"):
    # Convert to DataFrame
    input_df = pd.DataFrame(all_students)
    
    # Feature engineering
    input_df['attendance_rate_percent'] = input_df['Attendance'] / 71
    input_df['homework_ratio'] = input_df['Homework completion (20%)'] / 20
    input_df['assessment_ratio'] = input_df['Class Assessment Task (20%)'] / 20
    input_df['study_efficiency'] = input_df['Hours of studies per week'] / (input_df['Homework completion (20%)'] + 1)
    input_df['attendance_x_homework'] = input_df['attendance_rate_percent'] * input_df['homework_ratio']
    input_df['attendance_x_assessment'] = input_df['attendance_rate_percent'] * input_df['assessment_ratio']
    input_df['study_x_homework'] = input_df['study_efficiency'] * input_df['homework_ratio']
    
    # One-hot encoding
    input_df = pd.get_dummies(input_df)
    
    # Align with training features from model
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    
    # Predictions
    try:
        predictions = model.predict(input_df)
        
        # Display results
        results = pd.DataFrame({
            "Student": [f"Student {i+1}" for i in range(int(n_students))],
            "Predicted Final Math Score": predictions
        })
        
        st.subheader("Prediction Results")
        st.dataframe(results.style.format({"Predicted Final Math Score": "{:.2f}"}))
        
        # Save results button
        if st.button("Save Results"):
            try:
                results.to_csv("prediction_results.csv", index=False)
                st.success("Results saved to 'prediction_results.csv' in the app directory.")
            except Exception as e:
                st.error(f"Failed to save results: {str(e)}")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
