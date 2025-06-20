import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
from datetime import datetime, timedelta
import threading
import queue

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from data_processor import DataProcessor

st.set_page_config(
    page_title="Real-Time Processing - AI Health Analytics",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Real-Time Health Data Processing")
st.markdown("Monitor and analyze health data in real-time with live updates and instant insights.")

# Initialize components
if 'real_time_processor' not in st.session_state:
    st.session_state.real_time_processor = DataProcessor()

if 'streaming_data' not in st.session_state:
    st.session_state.streaming_data = []

if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False

if 'stream_config' not in st.session_state:
    st.session_state.stream_config = {
        'update_interval': 2,
        'data_points': 50,
        'metrics': ['heart_rate', 'stress_level', 'activity_level']
    }

# Real-time configuration
st.subheader("⚙️ Real-Time Configuration")

config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:
    update_interval = st.slider("Update Interval (seconds)", 1, 10, 2)
    st.session_state.stream_config['update_interval'] = update_interval

with config_col2:
    max_data_points = st.slider("Max Data Points", 20, 200, 50)
    st.session_state.stream_config['data_points'] = max_data_points

with config_col3:
    streaming_mode = st.selectbox("Streaming Mode", 
                                ["Simulation", "Live Data", "Hybrid"])

# Metric selection
st.subheader("📊 Metrics to Monitor")

available_metrics = [
    'heart_rate', 'stress_level', 'activity_level', 'mood_score', 
    'sleep_quality', 'energy_level', 'focus_level', 'anxiety_level'
]

selected_metrics = st.multiselect("Select Metrics", available_metrics, 
                                default=['heart_rate', 'stress_level', 'activity_level'])
st.session_state.stream_config['metrics'] = selected_metrics

# Streaming controls
st.subheader("🎮 Streaming Controls")

control_col1, control_col2, control_col3 = st.columns(3)

with control_col1:
    if st.button("▶️ Start Streaming", type="primary"):
        st.session_state.is_streaming = True
        st.success("✅ Real-time streaming started!")

with control_col2:
    if st.button("⏸️ Pause Streaming"):
        st.session_state.is_streaming = False
        st.info("⏸️ Streaming paused")

with control_col3:
    if st.button("🔄 Reset Data"):
        st.session_state.streaming_data = []
        st.success("🔄 Data reset")

# Real-time dashboard
if st.session_state.is_streaming or len(st.session_state.streaming_data) > 0:
    st.subheader("📊 Real-Time Dashboard")
    
    # Create placeholder for real-time updates
    dashboard_placeholder = st.empty()
    
    # Simulate real-time data generation
    if st.session_state.is_streaming:
        # Generate new data point
        new_data_point = generate_real_time_data(selected_metrics)
        st.session_state.streaming_data.append(new_data_point)
        
        # Limit data points
        if len(st.session_state.streaming_data) > max_data_points:
            st.session_state.streaming_data = st.session_state.streaming_data[-max_data_points:]
    
    # Display real-time dashboard
    if len(st.session_state.streaming_data) > 0:
        display_real_time_dashboard(dashboard_placeholder, selected_metrics)
        
        # Auto-refresh
        if st.session_state.is_streaming:
            time.sleep(update_interval)
            st.rerun()

# Alert system
st.subheader("🚨 Real-Time Alerts")

alert_col1, alert_col2 = st.columns(2)

with alert_col1:
    st.write("**Alert Thresholds:**")
    
    heart_rate_threshold = st.slider("Heart Rate Alert (bpm)", 60, 120, 100)
    stress_threshold = st.slider("Stress Level Alert", 1, 10, 8)
    
with alert_col2:
    st.write("**Alert Settings:**")
    
    alert_enabled = st.checkbox("Enable Alerts", value=True)
    alert_sound = st.checkbox("Sound Alerts", value=False)

# Check for alerts
if len(st.session_state.streaming_data) > 0 and alert_enabled:
    latest_data = st.session_state.streaming_data[-1]
    
    alerts = []
    
    if 'heart_rate' in latest_data and latest_data['heart_rate'] > heart_rate_threshold:
        alerts.append(f"⚠️ High heart rate detected: {latest_data['heart_rate']} bpm")
    
    if 'stress_level' in latest_data and latest_data['stress_level'] > stress_threshold:
        alerts.append(f"⚠️ High stress level detected: {latest_data['stress_level']}/10")
    
    if alerts:
        st.subheader("🚨 Active Alerts")
        for alert in alerts:
            st.error(alert)

# Data processing pipeline
st.subheader("🔄 Data Processing Pipeline")

pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(3)

with pipeline_col1:
    st.metric("Data Points Processed", len(st.session_state.streaming_data))

with pipeline_col2:
    processing_rate = len(st.session_state.streaming_data) / max(1, update_interval)
    st.metric("Processing Rate", f"{processing_rate:.1f} pts/sec")

with pipeline_col3:
    if len(st.session_state.streaming_data) > 0:
        latest_timestamp = st.session_state.streaming_data[-1]['timestamp']
        st.metric("Last Update", latest_timestamp.strftime("%H:%M:%S"))
    else:
        st.metric("Last Update", "N/A")

# Data quality monitoring
st.subheader("✅ Data Quality Monitoring")

if len(st.session_state.streaming_data) > 0:
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    
    with quality_col1:
        # Calculate data completeness
        total_expected = len(st.session_state.streaming_data) * len(selected_metrics)
        actual_data = sum(1 for point in st.session_state.streaming_data 
                         for metric in selected_metrics if metric in point)
        completeness = (actual_data / total_expected) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with quality_col2:
        # Calculate data freshness
        if st.session_state.streaming_data:
            last_update = st.session_state.streaming_data[-1]['timestamp']
            freshness = (datetime.now() - last_update).total_seconds()
            st.metric("Data Freshness", f"{freshness:.1f}s")
        else:
            st.metric("Data Freshness", "N/A")
    
    with quality_col3:
        # Calculate anomaly rate
        anomaly_rate = calculate_anomaly_rate(st.session_state.streaming_data, selected_metrics)
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

# Performance metrics
st.subheader("📈 Performance Metrics")

if len(st.session_state.streaming_data) > 10:
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        # Processing latency
        latencies = [point.get('processing_latency', 0) for point in st.session_state.streaming_data[-10:]]
        avg_latency = np.mean(latencies)
        st.metric("Avg Processing Latency", f"{avg_latency:.2f}ms")
        
        # Throughput
        throughput = len(st.session_state.streaming_data) / (time.time() - st.session_state.streaming_data[0]['timestamp'].timestamp())
        st.metric("Throughput", f"{throughput:.2f} pts/sec")
    
    with perf_col2:
        # Memory usage simulation
        memory_usage = len(st.session_state.streaming_data) * 0.1  # Simulated KB
        st.metric("Memory Usage", f"{memory_usage:.1f} KB")
        
        # CPU usage simulation
        cpu_usage = min(100, len(st.session_state.streaming_data) * 0.5)
        st.metric("CPU Usage", f"{cpu_usage:.1f}%")

# Data export
st.subheader("📥 Data Export")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("📄 Export Real-Time Data"):
        if st.session_state.streaming_data:
            df = pd.DataFrame(st.session_state.streaming_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"realtime_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data available for export")

with export_col2:
    if st.button("📊 Generate Report"):
        if st.session_state.streaming_data:
            generate_real_time_report()
        else:
            st.info("No data available for report generation")

def generate_real_time_data(metrics):
    """Generate simulated real-time health data"""
    data_point = {
        'timestamp': datetime.now(),
        'processing_latency': np.random.uniform(10, 50)  # ms
    }
    
    # Generate realistic health data
    for metric in metrics:
        if metric == 'heart_rate':
            data_point[metric] = np.random.normal(75, 15)
        elif metric == 'stress_level':
            data_point[metric] = np.random.uniform(1, 10)
        elif metric == 'activity_level':
            data_point[metric] = np.random.uniform(0, 100)
        elif metric == 'mood_score':
            data_point[metric] = np.random.uniform(1, 10)
        elif metric == 'sleep_quality':
            data_point[metric] = np.random.uniform(1, 10)
        elif metric == 'energy_level':
            data_point[metric] = np.random.uniform(1, 10)
        elif metric == 'focus_level':
            data_point[metric] = np.random.uniform(1, 10)
        elif metric == 'anxiety_level':
            data_point[metric] = np.random.uniform(1, 10)
        else:
            data_point[metric] = np.random.uniform(0, 100)
    
    return data_point

def display_real_time_dashboard(placeholder, metrics):
    """Display real-time dashboard with live updates"""
    with placeholder.container():
        if not st.session_state.streaming_data:
            st.info("No real-time data available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.streaming_data)
        
        # Current values
        st.write("**Current Values:**")
        current_col1, current_col2, current_col3, current_col4 = st.columns(4)
        
        latest_data = st.session_state.streaming_data[-1]
        
        for i, metric in enumerate(metrics[:4]):
            col = [current_col1, current_col2, current_col3, current_col4][i]
            if metric in latest_data:
                col.metric(metric.replace('_', ' ').title(), f"{latest_data[metric]:.1f}")
        
        # Time series charts
        if len(df) > 1:
            # Create subplots
            fig = make_subplots(
                rows=len(metrics), cols=1,
                subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
                vertical_spacing=0.1
            )
            
            for i, metric in enumerate(metrics):
                if metric in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df[metric],
                            mode='lines+markers',
                            name=metric,
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
            
            fig.update_layout(height=200*len(metrics), title="Real-Time Metrics")
            st.plotly_chart(fig, use_container_width=True)

def calculate_anomaly_rate(data, metrics):
    """Calculate anomaly rate in streaming data"""
    if len(data) < 10:
        return 0.0
    
    anomalies = 0
    total_points = 0
    
    for metric in metrics:
        values = [point.get(metric, 0) for point in data if metric in point]
        if len(values) > 5:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            for value in values:
                total_points += 1
                if abs(value - mean_val) > 2 * std_val:  # 2 sigma rule
                    anomalies += 1
    
    return (anomalies / max(1, total_points)) * 100

def generate_real_time_report():
    """Generate a comprehensive real-time processing report"""
    st.subheader("📊 Real-Time Processing Report")
    
    if not st.session_state.streaming_data:
        st.info("No data available for report")
        return
    
    df = pd.DataFrame(st.session_state.streaming_data)
    
    # Summary statistics
    st.write("**Summary Statistics:**")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'processing_latency' in numeric_cols:
        numeric_cols.remove('processing_latency')
    
    if numeric_cols:
        summary_stats = df[numeric_cols].describe()
        st.dataframe(summary_stats)
    
    # Time-based analysis
    st.write("**Time-Based Analysis:**")
    st.write(f"- Data collection period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    st.write(f"- Total data points: {len(df)}")
    st.write(f"- Average collection rate: {len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60):.2f} points/minute")
    
    # Performance analysis
    if 'processing_latency' in df.columns:
        st.write("**Performance Analysis:**")
        st.write(f"- Average processing latency: {df['processing_latency'].mean():.2f}ms")
        st.write(f"- Max processing latency: {df['processing_latency'].max():.2f}ms")
        st.write(f"- Min processing latency: {df['processing_latency'].min():.2f}ms")

# Sidebar information
st.sidebar.subheader("⚡ Real-Time Processing")
st.sidebar.markdown("""
**Features:**
- Live data streaming
- Real-time analytics
- Instant alerts
- Performance monitoring
- Data quality checks

**Status:**
""")

if st.session_state.is_streaming:
    st.sidebar.success("🟢 Streaming Active")
else:
    st.sidebar.info("⚪ Streaming Inactive")

st.sidebar.markdown(f"""
**Configuration:**
- Update interval: {st.session_state.stream_config['update_interval']}s
- Max data points: {st.session_state.stream_config['data_points']}
- Active metrics: {len(st.session_state.stream_config['metrics'])}
""")

# Connection status
st.sidebar.subheader("📡 Connection Status")
connection_status = "🟢 Connected" if st.session_state.is_streaming else "🔴 Disconnected"
st.sidebar.write(connection_status)

# System resources
st.sidebar.subheader("💻 System Resources")
st.sidebar.write("CPU: 🟢 Normal")
st.sidebar.write("Memory: 🟢 Normal")
st.sidebar.write("Network: 🟢 Normal")
