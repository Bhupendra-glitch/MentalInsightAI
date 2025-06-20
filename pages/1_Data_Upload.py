import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import io
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_processor import DataProcessor

st.set_page_config(
    page_title="Data Upload - AI Health Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Upload & Processing")
st.markdown("Upload your health data for analysis and insights.")

# Initialize data processor
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

# Upload section
st.subheader("📁 Upload Health Data")

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing your health data",
        type=['csv'],
        help="Upload a CSV file with health metrics like sleep, exercise, mood, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.session_state.has_data = True
            
            st.success(f"✅ Data uploaded successfully! {len(df)} rows loaded.")
            
            # Display basic info about the dataset
            st.subheader("📋 Dataset Overview")
            
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("Total Records", len(df))
            
            with info_col2:
                st.metric("Columns", len(df.columns))
            
            with info_col3:
                st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Show data preview
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data quality check
            st.subheader("🔍 Data Quality Assessment")
            
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                st.write("**Missing Values:**")
                missing_data = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
                
                if missing_df['Missing Count'].sum() == 0:
                    st.success("✅ No missing values found!")
            
            with quality_col2:
                st.write("**Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Data Type': df.dtypes.values.astype(str)
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            # Process data
            if st.button("🔄 Process Data", type="primary"):
                with st.spinner("Processing data..."):
                    processed_data = st.session_state.data_processor.process_data(df)
                    st.session_state.processed_data = processed_data
                    
                    if processed_data is not None:
                        st.success("✅ Data processed successfully!")
                        st.session_state.data_processed = True
                        
                        # Show processed data summary
                        st.subheader("📊 Processed Data Summary")
                        
                        summary_col1, summary_col2 = st.columns(2)
                        
                        with summary_col1:
                            st.write("**Numerical Columns:**")
                            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns.tolist()
                            if numeric_cols:
                                st.write(", ".join(numeric_cols))
                            else:
                                st.write("No numerical columns found")
                        
                        with summary_col2:
                            st.write("**Categorical Columns:**")
                            cat_cols = processed_data.select_dtypes(include=['object']).columns.tolist()
                            if cat_cols:
                                st.write(", ".join(cat_cols))
                            else:
                                st.write("No categorical columns found")
                        
                        # Basic statistics
                        if len(numeric_cols) > 0:
                            st.subheader("📈 Statistical Summary")
                            st.dataframe(processed_data[numeric_cols].describe(), use_container_width=True)
                    else:
                        st.error("❌ Failed to process data. Please check your data format.")
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted with health-related columns.")

with upload_col2:
    st.subheader("📝 Data Format Guide")
    
    st.markdown("""
    **Expected Columns:**
    - `date`: Date of measurement
    - `sleep_hours`: Hours of sleep
    - `exercise_minutes`: Exercise duration
    - `mood_score`: Mood rating (1-10)
    - `stress_level`: Stress level (1-10)
    - `heart_rate`: Heart rate (bpm)
    - `steps`: Daily step count
    - `calories_burned`: Calories burned
    - `water_intake`: Water intake (ml)
    - `weight`: Body weight (kg)
    
    **Example Format:**
    ```
    date,sleep_hours,exercise_minutes,mood_score
    2024-01-01,7.5,30,8
    2024-01-02,6.8,45,7
    ```
    """)

# Sample data template
st.subheader("📋 Sample Data Template")

if st.button("📥 Download Sample Template"):
    # Create sample data template
    sample_data = {
        'date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
        'sleep_hours': np.random.normal(7.5, 1, 30).round(1),
        'exercise_minutes': np.random.normal(30, 15, 30).astype(int),
        'mood_score': np.random.randint(1, 11, 30),
        'stress_level': np.random.randint(1, 11, 30),
        'heart_rate': np.random.normal(70, 10, 30).astype(int),
        'steps': np.random.normal(8000, 2000, 30).astype(int),
        'calories_burned': np.random.normal(2000, 300, 30).astype(int),
        'water_intake': np.random.normal(2000, 500, 30).astype(int),
        'weight': np.random.normal(70, 5, 30).round(1)
    }
    
    sample_df = pd.DataFrame(sample_data)
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="Download Sample CSV",
        data=csv_buffer.getvalue(),
        file_name="health_data_template.csv",
        mime="text/csv"
    )
    
    st.info("📄 Sample template generated with 30 days of example health data.")

# Data validation
if st.session_state.get('has_data', False):
    st.subheader("✅ Data Validation")
    
    df = st.session_state.uploaded_data
    
    validation_col1, validation_col2 = st.columns(2)
    
    with validation_col1:
        st.write("**Data Validation Results:**")
        
        # Check for required columns
        required_cols = ['date', 'sleep_hours', 'mood_score']
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            st.warning(f"⚠️ Missing recommended columns: {', '.join(missing_required)}")
        else:
            st.success("✅ All recommended columns present")
        
        # Check data ranges
        if 'mood_score' in df.columns:
            mood_range = df['mood_score'].min(), df['mood_score'].max()
            if mood_range[0] < 1 or mood_range[1] > 10:
                st.warning("⚠️ Mood scores should be between 1-10")
            else:
                st.success("✅ Mood scores in valid range")
    
    with validation_col2:
        st.write("**Data Completeness:**")
        
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        
        for col in df.columns:
            completeness_pct = completeness[col]
            if completeness_pct >= 90:
                st.success(f"✅ {col}: {completeness_pct:.1f}%")
            elif completeness_pct >= 70:
                st.warning(f"⚠️ {col}: {completeness_pct:.1f}%")
            else:
                st.error(f"❌ {col}: {completeness_pct:.1f}%")

# Status summary
if st.session_state.get('data_processed', False):
    st.success("🎉 Data is ready for analysis! You can now proceed to other sections of the platform.")
elif st.session_state.get('has_data', False):
    st.info("📊 Data uploaded successfully. Click 'Process Data' to prepare it for analysis.")
else:
    st.info("📁 Please upload your health data to get started with the AI analytics platform.")
