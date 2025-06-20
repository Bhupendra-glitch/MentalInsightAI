import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="AI Health Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏥 AI Health Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Platform overview
    st.markdown("""
    <div class="info-box">
        <h3>🚀 Welcome to the AI Health Analytics Platform</h3>
        <p>This platform provides comprehensive health analytics with AI-powered insights including:</p>
        <ul>
            <li><strong>Behavior Analysis:</strong> Analyze patterns in health behaviors and lifestyle data</li>
            <li><strong>Personalized Recommendations:</strong> Get tailored health suggestions based on your profile</li>
            <li><strong>Predictive Analytics:</strong> Mental health predictions and early warning systems</li>
            <li><strong>Real-time Processing:</strong> Live data processing and monitoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Platform statistics
    st.subheader("📊 Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Users",
            value="0",
            help="Number of users currently using the platform"
        )
    
    with col2:
        st.metric(
            label="Data Points Processed",
            value="0",
            help="Total health data points analyzed"
        )
    
    with col3:
        st.metric(
            label="ML Models Deployed",
            value="4",
            help="Number of active machine learning models"
        )
    
    with col4:
        st.metric(
            label="Accuracy Score",
            value="0%",
            help="Average model accuracy across all predictions"
        )
    
    # Quick start guide
    st.subheader("🎯 Quick Start Guide")
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.markdown("""
        ### 📊 Data Analysis Workflow
        1. **Upload Data**: Go to the Data Upload page to import your health data
        2. **Behavior Analysis**: Analyze patterns and trends in your health behaviors
        3. **Get Recommendations**: Receive personalized health recommendations
        4. **Predictive Insights**: View mental health predictions and risk assessments
        """)
    
    with steps_col2:
        st.markdown("""
        ### 🔧 Platform Features
        - **Interactive Dashboards**: Visualize your health data with interactive charts
        - **ML-Powered Insights**: Advanced machine learning for pattern recognition
        - **Real-time Processing**: Live data analysis and monitoring
        - **Personalized Experience**: Tailored recommendations based on your profile
        """)
    
    # System status
    st.subheader("⚙️ System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.success("🟢 Data Processing Engine: Online")
    
    with status_col2:
        st.success("🟢 ML Models: Active")
    
    with status_col3:
        st.success("🟢 Recommendation Engine: Ready")
    
    # Recent activity (empty state)
    st.subheader("📈 Recent Activity")
    
    if st.session_state.get('has_data', False):
        # This would show actual recent activity if data exists
        st.info("Recent health data analysis and recommendations would appear here.")
    else:
        st.info("No recent activity. Upload your health data to get started with personalized analytics.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>AI Health Analytics Platform | Powered by Machine Learning</p>
        <p>For support or questions, please refer to the documentation in each section.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
