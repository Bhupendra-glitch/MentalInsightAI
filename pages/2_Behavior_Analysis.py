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
from ml_models import BehaviorAnalyzer
from visualization import VisualizationHelper

st.set_page_config(
    page_title="Behavior Analysis - AI Health Analytics",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Behavior Analysis")
st.markdown("Analyze patterns and trends in your health behaviors using AI-powered insights.")

# Initialize components
if 'behavior_analyzer' not in st.session_state:
    st.session_state.behavior_analyzer = BehaviorAnalyzer()

if 'viz_helper' not in st.session_state:
    st.session_state.viz_helper = VisualizationHelper()

# Check if data is available
if not st.session_state.get('data_processed', False):
    st.error("❌ No processed data available. Please upload and process your data first.")
    st.info("👈 Go to the Data Upload page to upload your health data.")
    st.stop()

# Get processed data
data = st.session_state.processed_data

# Sidebar for analysis options
st.sidebar.subheader("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Overview", "Pattern Detection", "Correlation Analysis", "Clustering", "Anomaly Detection"]
)

# Main analysis section
if analysis_type == "Overview":
    st.subheader("📊 Behavioral Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 4:
        with col1:
            if 'sleep_hours' in data.columns:
                avg_sleep = data['sleep_hours'].mean()
                st.metric("Average Sleep", f"{avg_sleep:.1f} hrs")
            else:
                st.metric("Data Points", len(data))
        
        with col2:
            if 'exercise_minutes' in data.columns:
                avg_exercise = data['exercise_minutes'].mean()
                st.metric("Average Exercise", f"{avg_exercise:.0f} min")
            else:
                st.metric("Features", len(numeric_cols))
        
        with col3:
            if 'mood_score' in data.columns:
                avg_mood = data['mood_score'].mean()
                st.metric("Average Mood", f"{avg_mood:.1f}/10")
            else:
                st.metric("Avg Value", f"{data[numeric_cols[0]].mean():.1f}")
        
        with col4:
            if 'stress_level' in data.columns:
                avg_stress = data['stress_level'].mean()
                st.metric("Average Stress", f"{avg_stress:.1f}/10")
            else:
                st.metric("Max Value", f"{data[numeric_cols[0]].max():.1f}")
    
    # Behavior trends
    st.subheader("📈 Behavior Trends")
    
    if len(numeric_cols) > 0:
        # Time series plot
        fig = st.session_state.viz_helper.create_time_series_plot(data, numeric_cols[:4])
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        st.subheader("📊 Data Distributions")
        
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            if len(numeric_cols) > 0:
                fig = px.histogram(data, x=numeric_cols[0], nbins=20, 
                                 title=f"Distribution of {numeric_cols[0]}")
                st.plotly_chart(fig, use_container_width=True)
        
        with dist_col2:
            if len(numeric_cols) > 1:
                fig = px.box(data, y=numeric_cols[1], 
                           title=f"Box Plot of {numeric_cols[1]}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No numerical columns found for analysis.")

elif analysis_type == "Pattern Detection":
    st.subheader("🔍 Pattern Detection")
    
    if len(data) > 0:
        # Detect patterns using ML
        patterns = st.session_state.behavior_analyzer.detect_patterns(data)
        
        if patterns:
            st.success(f"✅ Detected {len(patterns)} behavioral patterns")
            
            # Display patterns
            for i, pattern in enumerate(patterns):
                with st.expander(f"Pattern {i+1}: {pattern['name']}"):
                    st.write(f"**Description:** {pattern['description']}")
                    st.write(f"**Confidence:** {pattern['confidence']:.2f}")
                    st.write(f"**Frequency:** {pattern['frequency']}")
                    
                    if 'visualization' in pattern:
                        st.plotly_chart(pattern['visualization'], use_container_width=True)
        else:
            st.info("No significant patterns detected in the current data.")
    else:
        st.error("Insufficient data for pattern detection.")
    
    # Pattern analysis options
    st.subheader("⚙️ Pattern Analysis Settings")
    
    pattern_col1, pattern_col2 = st.columns(2)
    
    with pattern_col1:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.1)
        
    with pattern_col2:
        lookback_days = st.slider("Lookback Period (days)", 7, 30, 14)
    
    if st.button("🔄 Reanalyze Patterns"):
        with st.spinner("Analyzing patterns..."):
            patterns = st.session_state.behavior_analyzer.detect_patterns(
                data, min_confidence=min_confidence, lookback_days=lookback_days
            )
            st.rerun()

elif analysis_type == "Correlation Analysis":
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Correlation matrix
        correlation_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Significant correlations
        st.subheader("🔍 Significant Correlations")
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'Feature 1': numeric_cols[i],
                        'Feature 2': numeric_cols[j],
                        'Correlation': corr_value,
                        'Strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })
        
        if strong_correlations:
            corr_df = pd.DataFrame(strong_correlations)
            corr_df['Correlation'] = corr_df['Correlation'].round(3)
            st.dataframe(corr_df, use_container_width=True)
            
            # Scatter plots for strong correlations
            st.subheader("📊 Correlation Scatter Plots")
            
            for corr in strong_correlations[:3]:  # Show top 3
                fig = px.scatter(data, x=corr['Feature 1'], y=corr['Feature 2'],
                               title=f"{corr['Feature 1']} vs {corr['Feature 2']} (r={corr['Correlation']:.3f})")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strong correlations found between features.")
    else:
        st.error("Need at least 2 numerical features for correlation analysis.")

elif analysis_type == "Clustering":
    st.subheader("🎯 Behavioral Clustering")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Clustering parameters
        cluster_col1, cluster_col2 = st.columns(2)
        
        with cluster_col1:
            n_clusters = st.slider("Number of Clusters", 2, 8, 3)
            
        with cluster_col2:
            selected_features = st.multiselect(
                "Select Features for Clustering",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
        
        if selected_features and st.button("🔄 Perform Clustering"):
            with st.spinner("Performing clustering analysis..."):
                cluster_results = st.session_state.behavior_analyzer.perform_clustering(
                    data[selected_features], n_clusters=n_clusters
                )
                
                if cluster_results:
                    st.success(f"✅ Successfully identified {n_clusters} behavioral clusters")
                    
                    # Add cluster labels to data
                    data_with_clusters = data.copy()
                    data_with_clusters['Cluster'] = cluster_results['labels']
                    
                    # Cluster visualization
                    if len(selected_features) >= 2:
                        fig = px.scatter(data_with_clusters, 
                                       x=selected_features[0], 
                                       y=selected_features[1],
                                       color='Cluster',
                                       title="Behavioral Clusters")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster statistics
                    st.subheader("📊 Cluster Statistics")
                    
                    cluster_stats = data_with_clusters.groupby('Cluster')[selected_features].mean()
                    st.dataframe(cluster_stats, use_container_width=True)
                    
                    # Cluster insights
                    st.subheader("💡 Cluster Insights")
                    
                    for cluster_id in range(n_clusters):
                        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
                        cluster_size = len(cluster_data)
                        
                        with st.expander(f"Cluster {cluster_id} ({cluster_size} data points)"):
                            st.write("**Characteristics:**")
                            for feature in selected_features:
                                avg_value = cluster_data[feature].mean()
                                st.write(f"- Average {feature}: {avg_value:.2f}")
                else:
                    st.error("❌ Clustering failed. Please try different parameters.")
    else:
        st.error("Need at least 2 numerical features for clustering.")

elif analysis_type == "Anomaly Detection":
    st.subheader("⚠️ Anomaly Detection")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Anomaly detection parameters
        anomaly_col1, anomaly_col2 = st.columns(2)
        
        with anomaly_col1:
            contamination = st.slider("Contamination Rate", 0.01, 0.2, 0.1, 0.01)
            
        with anomaly_col2:
            selected_features = st.multiselect(
                "Select Features for Anomaly Detection",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
        
        if selected_features and st.button("🔍 Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                anomaly_results = st.session_state.behavior_analyzer.detect_anomalies(
                    data[selected_features], contamination=contamination
                )
                
                if anomaly_results:
                    anomalies = anomaly_results['anomalies']
                    scores = anomaly_results['scores']
                    
                    st.success(f"✅ Detected {sum(anomalies)} anomalies out of {len(data)} data points")
                    
                    # Add anomaly information to data
                    data_with_anomalies = data.copy()
                    data_with_anomalies['Is_Anomaly'] = anomalies
                    data_with_anomalies['Anomaly_Score'] = scores
                    
                    # Visualize anomalies
                    if len(selected_features) >= 2:
                        fig = px.scatter(data_with_anomalies,
                                       x=selected_features[0],
                                       y=selected_features[1],
                                       color='Is_Anomaly',
                                       title="Anomaly Detection Results")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show anomalous data points
                    anomalous_data = data_with_anomalies[data_with_anomalies['Is_Anomaly']]
                    
                    if len(anomalous_data) > 0:
                        st.subheader("⚠️ Anomalous Data Points")
                        st.dataframe(anomalous_data[selected_features + ['Anomaly_Score']], use_container_width=True)
                        
                        # Anomaly insights
                        st.subheader("💡 Anomaly Insights")
                        
                        for feature in selected_features:
                            normal_mean = data_with_anomalies[~data_with_anomalies['Is_Anomaly']][feature].mean()
                            anomaly_mean = anomalous_data[feature].mean()
                            
                            st.write(f"**{feature}:**")
                            st.write(f"- Normal average: {normal_mean:.2f}")
                            st.write(f"- Anomaly average: {anomaly_mean:.2f}")
                            st.write(f"- Difference: {abs(anomaly_mean - normal_mean):.2f}")
                            st.write("")
                    else:
                        st.info("No anomalies detected with current parameters.")
                else:
                    st.error("❌ Anomaly detection failed. Please try different parameters.")
    else:
        st.error("No numerical features available for anomaly detection.")

# Analysis summary
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Analysis Summary")
st.sidebar.info(f"Dataset: {len(data)} records\nFeatures: {len(data.columns)}")

# Export results
if st.sidebar.button("📥 Export Analysis Results"):
    # Create analysis report
    report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': len(data),
        'features': data.columns.tolist(),
        'analysis_type': analysis_type
    }
    
    report_df = pd.DataFrame([report])
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    
    st.sidebar.download_button(
        label="Download Report",
        data=csv_buffer.getvalue(),
        file_name=f"behavior_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
