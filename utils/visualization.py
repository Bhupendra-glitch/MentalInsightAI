import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VisualizationHelper:
    """Comprehensive visualization utilities for health analytics"""
    
    def __init__(self):
        self.color_schemes = {
            'health': ['#2E8B57', '#32CD32', '#90EE90', '#FFD700', '#FF6347'],
            'mood': ['#4169E1', '#00BFFF', '#87CEEB', '#FFB6C1', '#FF69B4'],
            'stress': ['#228B22', '#FFD700', '#FF4500', '#DC143C', '#8B0000'],
            'sleep': ['#191970', '#4169E1', '#87CEEB', '#F0F8FF', '#FFFAF0']
        }
        
    def create_time_series_plot(self, data, columns, title="Health Metrics Over Time"):
        """Create comprehensive time series visualization"""
        try:
            # Determine date column
            date_col = None
            for col in ['date', 'timestamp', 'created_at']:
                if col in data.columns:
                    date_col = col
                    break
            
            if not date_col:
                # Create synthetic date index
                data = data.copy()
                data['date'] = pd.date_range(start='2024-01-01', periods=len(data), freq='D')
                date_col = 'date'
            
            # Create subplots
            fig = make_subplots(
                rows=len(columns), cols=1,
                subplot_titles=[col.replace('_', ' ').title() for col in columns],
                vertical_spacing=0.08,
                shared_xaxes=True
            )
            
            colors = px.colors.qualitative.Set1
            
            for i, col in enumerate(columns):
                if col in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data[date_col],
                            y=data[col],
                            mode='lines+markers',
                            name=col.replace('_', ' ').title(),
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4),
                            hovertemplate=f'<b>{col.replace("_", " ").title()}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Value: %{y:.2f}<br>' +
                                        '<extra></extra>'
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add trend line
                    if len(data) > 3:
                        z = np.polyfit(range(len(data[col].dropna())), data[col].dropna(), 1)
                        trend = np.poly1d(z)(range(len(data)))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=data[date_col],
                                y=trend,
                                mode='lines',
                                name=f'{col.replace("_", " ").title()} Trend',
                                line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                                opacity=0.6,
                                showlegend=False,
                                hoverinfo='skip'
                            ),
                            row=i+1, col=1
                        )
            
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=18)),
                height=200 * len(columns),
                showlegend=False,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=len(columns), col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error creating time series plot: {str(e)}")
            return self._create_fallback_plot("Time Series Visualization Error")
    
    def create_correlation_heatmap(self, data, title="Feature Correlations"):
        """Create interactive correlation heatmap"""
        try:
            # Select numerical columns only
            numerical_data = data.select_dtypes(include=[np.number])
            
            if numerical_data.empty:
                return self._create_fallback_plot("No numerical data available")
            
            # Calculate correlation matrix
            corr_matrix = numerical_data.corr()
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16)),
                xaxis_title="Features",
                yaxis_title="Features",
                width=600,
                height=600
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {str(e)}")
            return self._create_fallback_plot("Correlation Heatmap Error")
    
    def create_distribution_plot(self, data, column, title=None):
        """Create distribution visualization with statistics"""
        try:
            if column not in data.columns:
                return self._create_fallback_plot(f"Column {column} not found")
            
            col_data = data[column].dropna()
            
            if len(col_data) == 0:
                return self._create_fallback_plot(f"No data available for {column}")
            
            if not title:
                title = f"Distribution of {column.replace('_', ' ').title()}"
            
            # Create subplot with histogram and box plot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=["Distribution", "Box Plot"],
                vertical_spacing=0.1
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=col_data,
                    nbinsx=20,
                    name="Distribution",
                    marker_color='skyblue',
                    opacity=0.7,
                    hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add normal curve if data looks normal
            if len(col_data) > 10:
                mu, sigma = col_data.mean(), col_data.std()
                x_range = np.linspace(col_data.min(), col_data.max(), 100)
                normal_curve = ((1 / (sigma * np.sqrt(2 * np.pi))) * 
                               np.exp(-0.5 * ((x_range - mu) / sigma) ** 2))
                
                # Scale to match histogram
                normal_curve = normal_curve * len(col_data) * (col_data.max() - col_data.min()) / 20
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=normal_curve,
                        mode='lines',
                        name='Normal Curve',
                        line=dict(color='red', width=2, dash='dash'),
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    x=col_data,
                    name="Box Plot",
                    marker_color='lightgreen',
                    hovertemplate='<b>Statistics</b><br>' +
                                'Q1: %{q1}<br>' +
                                'Median: %{median}<br>' +
                                'Q3: %{q3}<br>' +
                                'Min: %{min}<br>' +
                                'Max: %{max}<br>' +
                                '<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add statistics annotation
            stats_text = (f"Mean: {col_data.mean():.2f}<br>"
                         f"Std: {col_data.std():.2f}<br>"
                         f"Min: {col_data.min():.2f}<br>"
                         f"Max: {col_data.max():.2f}")
            
            fig.add_annotation(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                xanchor="right", yanchor="top",
                showarrow=False,
                bordercolor="gray",
                borderwidth=1,
                bgcolor="white",
                font=dict(size=10)
            )
            
            fig.update_layout(
                title=dict(text=title, x=0.5),
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating distribution plot: {str(e)}")
            return self._create_fallback_plot("Distribution Plot Error")
    
    def create_scatter_matrix(self, data, columns, title="Scatter Matrix"):
        """Create interactive scatter matrix"""
        try:
            # Select only the specified columns that exist
            available_cols = [col for col in columns if col in data.columns]
            
            if len(available_cols) < 2:
                return self._create_fallback_plot("Need at least 2 columns for scatter matrix")
            
            # Create scatter matrix
            fig = ff.create_scatterplotmatrix(
                data[available_cols],
                diag='histogram',
                height=600,
                width=600
            )
            
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16))
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating scatter matrix: {str(e)}")
            return self._create_fallback_plot("Scatter Matrix Error")
    
    def create_health_dashboard(self, data):
        """Create comprehensive health dashboard"""
        try:
            # Create dashboard with multiple subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Daily Trends", "Health Metrics Distribution", 
                               "Sleep vs Exercise", "Mood vs Stress"),
                specs=[[{"secondary_y": True}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )
            
            # Plot 1: Daily trends (mood and stress)
            if 'mood_score' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        y=data['mood_score'],
                        mode='lines+markers',
                        name='Mood Score',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
            
            if 'stress_level' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        y=data['stress_level'],
                        mode='lines+markers',
                        name='Stress Level',
                        line=dict(color='red')
                    ),
                    row=1, col=1, secondary_y=True
                )
            
            # Plot 2: Health metrics distribution
            if 'sleep_hours' in data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=data['sleep_hours'],
                        name='Sleep Hours',
                        marker_color='lightblue',
                        nbinsx=15
                    ),
                    row=1, col=2
                )
            
            # Plot 3: Sleep vs Exercise scatter
            if 'sleep_hours' in data.columns and 'exercise_minutes' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['sleep_hours'],
                        y=data['exercise_minutes'],
                        mode='markers',
                        name='Sleep vs Exercise',
                        marker=dict(color='green', size=8, opacity=0.6)
                    ),
                    row=2, col=1
                )
            
            # Plot 4: Mood vs Stress scatter
            if 'mood_score' in data.columns and 'stress_level' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['mood_score'],
                        y=data['stress_level'],
                        mode='markers',
                        name='Mood vs Stress',
                        marker=dict(color='purple', size=8, opacity=0.6)
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title=dict(text="Health Analytics Dashboard", x=0.5, font=dict(size=18)),
                height=700,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating health dashboard: {str(e)}")
            return self._create_fallback_plot("Health Dashboard Error")
    
    def create_weekly_pattern_plot(self, data, column):
        """Create weekly pattern visualization"""
        try:
            if column not in data.columns:
                return self._create_fallback_plot(f"Column {column} not found")
            
            # Add weekday if date column exists
            date_col = None
            for col in ['date', 'timestamp', 'created_at']:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col:
                data = data.copy()
                data['weekday'] = pd.to_datetime(data[date_col]).dt.day_name()
            else:
                # Create synthetic weekday pattern
                data = data.copy()
                weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                data['weekday'] = [weekdays[i % 7] for i in range(len(data))]
            
            # Calculate weekly averages
            weekly_avg = data.groupby('weekday')[column].agg(['mean', 'std']).reset_index()
            
            # Order by weekday
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_avg['weekday'] = pd.Categorical(weekly_avg['weekday'], categories=weekday_order, ordered=True)
            weekly_avg = weekly_avg.sort_values('weekday')
            
            # Create bar plot with error bars
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=weekly_avg['weekday'],
                y=weekly_avg['mean'],
                error_y=dict(type='data', array=weekly_avg['std']),
                name=f'Average {column.replace("_", " ").title()}',
                marker_color='skyblue',
                hovertemplate='<b>%{x}</b><br>Average: %{y:.2f}<br>Std Dev: %{error_y.array:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Weekly Pattern: {column.replace("_", " ").title()}',
                xaxis_title='Day of Week',
                yaxis_title=column.replace('_', ' ').title(),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating weekly pattern plot: {str(e)}")
            return self._create_fallback_plot("Weekly Pattern Error")
    
    def create_progress_chart(self, data, target_column, goal_value=None):
        """Create progress tracking chart"""
        try:
            if target_column not in data.columns:
                return self._create_fallback_plot(f"Column {target_column} not found")
            
            # Calculate rolling average
            rolling_avg = data[target_column].rolling(window=7, min_periods=1).mean()
            
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                y=data[target_column],
                mode='lines+markers',
                name='Actual Values',
                line=dict(color='lightblue', width=1),
                marker=dict(size=4),
                opacity=0.6
            ))
            
            # Add rolling average
            fig.add_trace(go.Scatter(
                y=rolling_avg,
                mode='lines',
                name='7-day Average',
                line=dict(color='blue', width=3)
            ))
            
            # Add goal line if provided
            if goal_value:
                fig.add_hline(
                    y=goal_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Goal: {goal_value}"
                )
            
            # Add trend line
            if len(data) > 3:
                x_vals = list(range(len(data)))
                z = np.polyfit(x_vals, data[target_column].fillna(data[target_column].mean()), 1)
                trend_line = np.poly1d(z)(x_vals)
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(color='orange', width=2, dash='dot')
                ))
            
            fig.update_layout(
                title=f'Progress Tracking: {target_column.replace("_", " ").title()}',
                xaxis_title='Time Period',
                yaxis_title=target_column.replace('_', ' ').title(),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating progress chart: {str(e)}")
            return self._create_fallback_plot("Progress Chart Error")
    
    def create_correlation_network(self, data, threshold=0.5):
        """Create network visualization of correlations"""
        try:
            # Calculate correlations
            numerical_data = data.select_dtypes(include=[np.number])
            corr_matrix = numerical_data.corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) >= threshold:
                        strong_correlations.append({
                            'source': corr_matrix.columns[i],
                            'target': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if not strong_correlations:
                return self._create_fallback_plot("No strong correlations found")
            
            # Create network-style scatter plot
            fig = go.Figure()
            
            # Add edges (correlations)
            for corr in strong_correlations:
                # Simple visualization - could be enhanced with actual network layout
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, corr['correlation']],
                    mode='lines+text',
                    text=['', f"{corr['source']} ↔ {corr['target']}<br>r={corr['correlation']:.3f}"],
                    textposition='middle center',
                    line=dict(
                        color='red' if corr['correlation'] < 0 else 'blue',
                        width=abs(corr['correlation']) * 5
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"{corr['source']} ↔ {corr['target']}: {corr['correlation']:.3f}"
                ))
            
            fig.update_layout(
                title='Strong Feature Correlations Network',
                xaxis=dict(visible=False),
                yaxis=dict(title='Correlation Strength'),
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation network: {str(e)}")
            return self._create_fallback_plot("Correlation Network Error")
    
    def create_anomaly_plot(self, data, column, anomalies=None):
        """Create anomaly detection visualization"""
        try:
            if column not in data.columns:
                return self._create_fallback_plot(f"Column {column} not found")
            
            fig = go.Figure()
            
            # Plot normal data points
            normal_indices = list(range(len(data)))
            if anomalies is not None:
                normal_indices = [i for i in range(len(data)) if not anomalies[i]]
            
            fig.add_trace(go.Scatter(
                x=normal_indices,
                y=data[column].iloc[normal_indices],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=6)
            ))
            
            # Plot anomalies if provided
            if anomalies is not None:
                anomaly_indices = [i for i in range(len(data)) if anomalies[i]]
                if anomaly_indices:
                    fig.add_trace(go.Scatter(
                        x=anomaly_indices,
                        y=data[column].iloc[anomaly_indices],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
            
            # Add control limits (simple statistical approach)
            mean_val = data[column].mean()
            std_val = data[column].std()
            
            fig.add_hline(
                y=mean_val + 2*std_val,
                line_dash="dash",
                line_color="orange",
                annotation_text="Upper Control Limit"
            )
            
            fig.add_hline(
                y=mean_val - 2*std_val,
                line_dash="dash",
                line_color="orange",
                annotation_text="Lower Control Limit"
            )
            
            fig.add_hline(
                y=mean_val,
                line_dash="dot",
                line_color="green",
                annotation_text="Mean"
            )
            
            fig.update_layout(
                title=f'Anomaly Detection: {column.replace("_", " ").title()}',
                xaxis_title='Data Point Index',
                yaxis_title=column.replace('_', ' ').title()
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating anomaly plot: {str(e)}")
            return self._create_fallback_plot("Anomaly Plot Error")
    
    def _create_fallback_plot(self, error_message):
        """Create a fallback plot when visualization fails"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"Visualization Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="red"),
            bgcolor="lightgray",
            bordercolor="red",
            borderwidth=2
        )
        
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=400,
            height=300
        )
        
        return fig
    
    def get_color_palette(self, theme='health', n_colors=5):
        """Get color palette for specific theme"""
        colors = self.color_schemes.get(theme, self.color_schemes['health'])
        
        if n_colors <= len(colors):
            return colors[:n_colors]
        else:
            # Extend palette if needed
            extended_colors = colors * (n_colors // len(colors) + 1)
            return extended_colors[:n_colors]
    
    def create_metric_gauge(self, value, title, min_val=0, max_val=10, target=None):
        """Create gauge chart for single metric"""
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                delta={'reference': target if target else (max_val + min_val) / 2},
                gauge={
                    'axis': {'range': [min_val, max_val]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [min_val, (max_val - min_val) * 0.3 + min_val], 'color': "lightgray"},
                        {'range': [(max_val - min_val) * 0.3 + min_val, (max_val - min_val) * 0.7 + min_val], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target if target else max_val * 0.9
                    }
                }
            ))
            
            fig.update_layout(height=300)
            return fig
            
        except Exception as e:
            print(f"Error creating gauge: {str(e)}")
            return self._create_fallback_plot("Gauge Chart Error")
