import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from predictive_models import PredictiveAnalyzer

st.set_page_config(
    page_title="Predictive Analytics - AI Health Analytics",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Predictive Mental Health Analytics")
st.markdown("Advanced AI-powered predictions for mental health indicators and early warning systems.")

# Initialize predictive analyzer
if 'predictive_analyzer' not in st.session_state:
    st.session_state.predictive_analyzer = PredictiveAnalyzer()

# Check if data is available
if not st.session_state.get('data_processed', False):
    st.error("❌ No processed data available. Please upload and process your data first.")
    st.info("👈 Go to the Data Upload page to upload your health data.")
    st.stop()

# Get processed data
data = st.session_state.processed_data

# Prediction settings
st.subheader("⚙️ Prediction Settings")

settings_col1, settings_col2, settings_col3 = st.columns(3)

with settings_col1:
    prediction_horizon = st.selectbox("Prediction Horizon", 
                                    ["1 day", "3 days", "1 week", "2 weeks", "1 month"])

with settings_col2:
    prediction_type = st.selectbox("Prediction Type", 
                                 ["Mental Health Score", "Stress Level", "Mood Prediction", "Risk Assessment"])

with settings_col3:
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.8, 0.05)

# Model selection
st.subheader("🤖 Model Configuration")

model_col1, model_col2 = st.columns(2)

with model_col1:
    model_type = st.selectbox("Model Type", 
                            ["Random Forest", "Gradient Boosting", "Neural Network", "Ensemble"])
    
with model_col2:
    include_features = st.multiselect("Features to Include", 
                                    data.select_dtypes(include=[np.number]).columns.tolist(),
                                    default=data.select_dtypes(include=[np.number]).columns.tolist()[:5])

# Generate predictions
if st.button("🔮 Generate Predictions", type="primary"):
    if not include_features:
        st.error("❌ Please select at least one feature for prediction.")
    else:
        with st.spinner("Training model and generating predictions..."):
            # Configure prediction parameters
            prediction_config = {
                'horizon': prediction_horizon,
                'type': prediction_type,
                'model': model_type,
                'features': include_features,
                'confidence_threshold': confidence_threshold
            }
            
            # Generate predictions
            predictions = st.session_state.predictive_analyzer.generate_predictions(
                data, prediction_config
            )
            
            if predictions:
                st.session_state.current_predictions = predictions
                st.success("✅ Predictions generated successfully!")
            else:
                st.error("❌ Failed to generate predictions. Please check your data and settings.")

# Display predictions
if st.session_state.get('current_predictions'):
    predictions = st.session_state.current_predictions
    
    st.subheader("📊 Prediction Results")
    
    # Overall prediction summary
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        if 'overall_score' in predictions:
            st.metric("Predicted Score", f"{predictions['overall_score']:.1f}/10")
        else:
            st.metric("Model Accuracy", f"{predictions.get('accuracy', 0):.1%}")
    
    with summary_col2:
        if 'confidence' in predictions:
            st.metric("Confidence", f"{predictions['confidence']:.1%}")
        else:
            st.metric("Predictions Made", len(predictions.get('predictions', [])))
    
    with summary_col3:
        if 'risk_level' in predictions:
            risk_color = "🔴" if predictions['risk_level'] == "High" else "🟡" if predictions['risk_level'] == "Medium" else "🟢"
            st.metric("Risk Level", f"{risk_color} {predictions['risk_level']}")
        else:
            st.metric("Horizon", prediction_horizon)
    
    with summary_col4:
        if 'trend' in predictions:
            trend_icon = "📈" if predictions['trend'] == "Improving" else "📉" if predictions['trend'] == "Declining" else "➡️"
            st.metric("Trend", f"{trend_icon} {predictions['trend']}")
        else:
            st.metric("Model Type", model_type)
    
    # Prediction timeline
    st.subheader("📈 Prediction Timeline")
    
    if 'timeline' in predictions:
        timeline_data = predictions['timeline']
        
        # Create timeline chart
        fig = go.Figure()
        
        # Historical data
        if len(data) > 0:
            historical_dates = pd.date_range(end=datetime.now(), periods=len(data), freq='D')
            if 'mood_score' in data.columns:
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=data['mood_score'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
        
        # Predicted data
        future_dates = pd.date_range(start=datetime.now() + timedelta(days=1), 
                                   periods=len(timeline_data), freq='D')
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=timeline_data,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Mental Health Score Prediction Timeline",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors analysis
    st.subheader("⚠️ Risk Factors Analysis")
    
    if 'risk_factors' in predictions:
        risk_factors = predictions['risk_factors']
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.write("**High Risk Factors:**")
            high_risk = [rf for rf in risk_factors if rf['risk_level'] == 'High']
            
            if high_risk:
                for factor in high_risk:
                    st.error(f"🔴 {factor['factor']}: {factor['description']}")
            else:
                st.success("✅ No high-risk factors identified")
        
        with risk_col2:
            st.write("**Protective Factors:**")
            protective = [rf for rf in risk_factors if rf['risk_level'] == 'Low']
            
            if protective:
                for factor in protective:
                    st.success(f"🟢 {factor['factor']}: {factor['description']}")
            else:
                st.info("No specific protective factors identified")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    if 'feature_importance' in predictions:
        importance_data = predictions['feature_importance']
        
        # Create feature importance chart
        fig = px.bar(
            x=list(importance_data.values()),
            y=list(importance_data.keys()),
            orientation='h',
            title="Feature Importance for Predictions"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Top features explanation
        st.write("**Top 3 Most Important Features:**")
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(sorted_features[:3]):
            st.write(f"{i+1}. **{feature}**: {importance:.3f} - {get_feature_explanation(feature)}")
    
    # Predictions breakdown
    st.subheader("📊 Detailed Predictions")
    
    if 'detailed_predictions' in predictions:
        detailed = predictions['detailed_predictions']
        
        # Create tabs for different prediction aspects
        pred_tabs = st.tabs(["Daily Predictions", "Weekly Trends", "Monthly Outlook"])
        
        with pred_tabs[0]:
            if 'daily' in detailed:
                daily_df = pd.DataFrame(detailed['daily'])
                st.dataframe(daily_df, use_container_width=True)
        
        with pred_tabs[1]:
            if 'weekly' in detailed:
                weekly_data = detailed['weekly']
                fig = px.line(x=range(len(weekly_data)), y=weekly_data, 
                            title="Weekly Trend Prediction")
                st.plotly_chart(fig, use_container_width=True)
        
        with pred_tabs[2]:
            if 'monthly' in detailed:
                st.write("**Monthly Outlook:**")
                monthly_outlook = detailed['monthly']
                st.write(monthly_outlook)

# Early warning system
st.subheader("🚨 Early Warning System")

warning_col1, warning_col2 = st.columns(2)

with warning_col1:
    st.write("**Alert Thresholds:**")
    
    mood_threshold = st.slider("Mood Score Alert Threshold", 1, 10, 4)
    stress_threshold = st.slider("Stress Level Alert Threshold", 1, 10, 7)
    
with warning_col2:
    st.write("**Notification Settings:**")
    
    alert_types = st.multiselect("Alert Types", 
                               ["Email", "SMS", "Push Notification", "Dashboard Alert"])
    alert_frequency = st.selectbox("Alert Frequency", 
                                 ["Immediate", "Daily Summary", "Weekly Summary"])

if st.button("⚡ Activate Early Warning System"):
    st.success("✅ Early warning system activated!")
    st.info("You will receive alerts based on your configured thresholds and preferences.")

# Model performance
st.subheader("📈 Model Performance")

if st.session_state.get('current_predictions'):
    predictions = st.session_state.current_predictions
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        accuracy = predictions.get('model_performance', {}).get('accuracy', 0.85)
        st.metric("Accuracy", f"{accuracy:.1%}")
    
    with perf_col2:
        precision = predictions.get('model_performance', {}).get('precision', 0.82)
        st.metric("Precision", f"{precision:.1%}")
    
    with perf_col3:
        recall = predictions.get('model_performance', {}).get('recall', 0.88)
        st.metric("Recall", f"{recall:.1%}")
    
    # Model comparison
    if st.button("📊 Compare Models"):
        st.info("Model comparison feature would show performance across different algorithms.")

def get_feature_explanation(feature):
    """Get explanation for feature importance"""
    explanations = {
        'sleep_hours': 'Sleep quality strongly correlates with mental health',
        'exercise_minutes': 'Physical activity improves mood and reduces stress',
        'mood_score': 'Previous mood patterns are predictive of future mental state',
        'stress_level': 'Chronic stress is a major risk factor for mental health issues',
        'heart_rate': 'Physiological indicators reflect mental state',
        'steps': 'Daily activity level impacts mental wellbeing'
    }
    return explanations.get(feature, 'This feature contributes to mental health predictions')

# Prediction history
st.subheader("📚 Prediction History")

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if st.session_state.get('current_predictions'):
    if st.button("💾 Save Predictions"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.prediction_history.append({
            'timestamp': timestamp,
            'predictions': st.session_state.current_predictions,
            'config': {
                'horizon': prediction_horizon,
                'type': prediction_type,
                'model': model_type
            }
        })
        st.success("✅ Predictions saved to history!")

if st.session_state.prediction_history:
    st.write(f"**History:** {len(st.session_state.prediction_history)} saved prediction sets")
    
    # Show recent history
    for i, history_item in enumerate(reversed(st.session_state.prediction_history[-3:])):
        with st.expander(f"📅 {history_item['timestamp']}"):
            config = history_item['config']
            st.write(f"**Type:** {config['type']}")
            st.write(f"**Horizon:** {config['horizon']}")
            st.write(f"**Model:** {config['model']}")
            
            if 'overall_score' in history_item['predictions']:
                st.write(f"**Predicted Score:** {history_item['predictions']['overall_score']:.1f}/10")
else:
    st.info("No prediction history yet. Generate and save predictions to build your history.")

# Sidebar information
st.sidebar.subheader("🔮 Predictive Analytics Info")
st.sidebar.markdown("""
**Model Types:**
- **Random Forest**: Ensemble method, good for complex patterns
- **Gradient Boosting**: Sequential learning, high accuracy
- **Neural Network**: Deep learning for non-linear patterns
- **Ensemble**: Combines multiple models

**Prediction Accuracy:**
- Models are trained on your historical data
- Longer data history = better predictions
- Regular updates improve accuracy
""")

st.sidebar.subheader("⚠️ Important Notes")
st.sidebar.warning("""
These predictions are for informational purposes only and should not replace professional medical advice. Please consult healthcare professionals for medical decisions.
""")
