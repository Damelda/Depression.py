import streamlit as st
import joblib
import pandas as pd
import time
import altair as alt

# Load model
model = joblib.load("depression_model.pkl")
features = ['Interest', 'Sleep', 'Fatigue', 'Appetite',
            'Worthlessness', 'Concentration', 'Agitation', 'Suicidal Ideation']

# Page config
st.set_page_config(page_title="Depression Detector", page_icon="🧠", layout="centered")

# Optional: Background image (uncomment to use)
# st.markdown("""
#     <style>
#     .stApp {
#         background-image: url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d");
#         background-size: cover;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }
#     </style>
# """, unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📝 Take Assessment", "📊 About Model"])

with tab1:
    st.title("🧠 Depression Detection App")
    st.markdown("Rate each symptom from **0 (None)** to **3 (Severe)**")

    user_input = {}
    for feature in features:
        user_input[feature] = st.slider(f"{feature}", 0, 3, 1)

    if st.button("🩺 Predict", key="predict_button"):
        with st.spinner("Analyzing your symptoms..."):
            time.sleep(1.5)
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            confidence = model.predict_proba(input_df).max() * 100

            label_map = {0: "Mild", 1: "Moderate", 2: "No depression", 3: "Severe"}
            result = label_map.get(prediction, "Unknown")

            st.success(f"🔍 **Prediction:** {result}")
            st.info(f"📈 **Confidence:** {confidence:.2f}%")

    st.markdown("---")


csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

if csv_file is not None:
    try:
        batch_df = pd.read_csv(csv_file)

        # Validate columns
        missing = [col for col in features if col not in batch_df.columns]
        if missing:
            st.error(f"❌ Missing columns in uploaded file: {', '.join(missing)}")
        elif not batch_df[features].applymap(lambda x: isinstance(x, (int, float)) and 0 <= x <= 3).all().all():
            st.error("❌ All values must be between 0 and 3 (inclusive). No text or missing values.")
        else:
            # Run predictions
            preds = model.predict(batch_df[features])
            probs = model.predict_proba(batch_df[features]).max(axis=1) * 100

            label_map = {0: "Mild", 1: "Moderate", 2: "No depression", 3: "Severe"}
            results = [label_map.get(p, "Unknown") for p in preds]
            batch_df['Prediction'] = results
            batch_df['Confidence (%)'] = probs.round(2)

            st.subheader("📋 Results")

            # Style severe rows
            def highlight_severe(row):
                if row['Prediction'] == "Severe":
                    return ['background-color: #ffcccc'] * len(row)
                else:
                    return [''] * len(row)

            st.dataframe(
                batch_df.style.apply(highlight_severe, axis=1),
                use_container_width=True
            )

            # Download button
            csv_download = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_download,
                file_name="depression_predictions.csv",
                mime="text/csv"
            )

            # Chart
            st.subheader("📊 Prediction Distribution")
            chart_data = batch_df['Prediction'].value_counts().rename_axis('Class').reset_index(name='Count')

            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Class', sort='-y'),
                y='Count',
                color='Class',
            ).properties(width=500, height=350)

            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"❌ File processing error: {e}")


with tab2:
    st.title("📊 Model Information")
    st.markdown("""
    ### 🧠 Overview
    - ✅ Built using **scikit-learn**
    - 🔢 Trained on PHQ-9 symptom features
    - 🎯 Predicts 4 classes:
        - No Depression  
        - Mild  
        - Moderate  
        - Severe

    ### 📁 Batch Prediction Guide
    You can upload a CSV file for multiple predictions at once. Your file should:
    
    - Contain **exactly these 8 columns**:
        - `Interest`, `Sleep`, `Fatigue`, `Appetite`,  
          `Worthlessness`, `Concentration`, `Agitation`, `Suicidal Ideation`
    - Each column should have integer values from 0 to 3
    - Example row:  
      `2, 1, 3, 0, 1, 2, 0, 0`
    
    ✅ After upload:
    - Predictions and model confidence will appear in a table
    - A bar chart will show class distribution

    ### 🚀 Deployment
    - This app is built with **Streamlit** and hosted on **Streamlit Cloud**
    - Designed to be user-friendly and accessible
    """)


