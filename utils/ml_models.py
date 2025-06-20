import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BehaviorAnalyzer:
    """Machine learning models for behavior analysis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.pattern_history = []
        
    def detect_patterns(self, data, min_confidence=0.7, lookback_days=14):
        """Detect behavioral patterns in health data"""
        try:
            patterns = []
            
            # Pattern 1: Sleep-Exercise Correlation
            if 'sleep_hours' in data.columns and 'exercise_minutes' in data.columns:
                sleep_exercise_pattern = self._analyze_sleep_exercise_pattern(data)
                if sleep_exercise_pattern['confidence'] >= min_confidence:
                    patterns.append(sleep_exercise_pattern)
            
            # Pattern 2: Mood-Stress Relationship
            if 'mood_score' in data.columns and 'stress_level' in data.columns:
                mood_stress_pattern = self._analyze_mood_stress_pattern(data)
                if mood_stress_pattern['confidence'] >= min_confidence:
                    patterns.append(mood_stress_pattern)
            
            # Pattern 3: Activity Levels
            if 'steps' in data.columns or 'exercise_minutes' in data.columns:
                activity_pattern = self._analyze_activity_pattern(data)
                if activity_pattern['confidence'] >= min_confidence:
                    patterns.append(activity_pattern)
            
            # Pattern 4: Weekly Cycles
            if 'date' in data.columns:
                weekly_pattern = self._analyze_weekly_cycles(data)
                if weekly_pattern['confidence'] >= min_confidence:
                    patterns.append(weekly_pattern)
            
            # Pattern 5: Trend Analysis
            trend_patterns = self._analyze_trends(data, lookback_days)
            patterns.extend([p for p in trend_patterns if p['confidence'] >= min_confidence])
            
            return patterns
            
        except Exception as e:
            print(f"Error in pattern detection: {str(e)}")
            return []
    
    def _analyze_sleep_exercise_pattern(self, data):
        """Analyze sleep-exercise correlation pattern"""
        correlation = data['sleep_hours'].corr(data['exercise_minutes'])
        
        pattern = {
            'name': 'Sleep-Exercise Correlation',
            'description': f'Sleep and exercise show {"positive" if correlation > 0 else "negative"} correlation',
            'confidence': abs(correlation),
            'frequency': 'Daily',
            'correlation_value': correlation,
            'insight': self._get_sleep_exercise_insight(correlation)
        }
        
        # Create visualization
        fig = px.scatter(data, x='sleep_hours', y='exercise_minutes', 
                        title='Sleep vs Exercise Pattern')
        pattern['visualization'] = fig
        
        return pattern
    
    def _analyze_mood_stress_pattern(self, data):
        """Analyze mood-stress relationship pattern"""
        correlation = data['mood_score'].corr(data['stress_level'])
        
        # Calculate pattern strength
        high_stress_low_mood = len(data[(data['stress_level'] > 7) & (data['mood_score'] < 5)])
        total_records = len(data)
        
        pattern = {
            'name': 'Mood-Stress Relationship',
            'description': f'Mood and stress show {"inverse" if correlation < -0.3 else "weak"} relationship',
            'confidence': abs(correlation) if abs(correlation) > 0.3 else 0.5,
            'frequency': 'Daily',
            'correlation_value': correlation,
            'high_stress_episodes': high_stress_low_mood,
            'insight': self._get_mood_stress_insight(correlation, high_stress_low_mood, total_records)
        }
        
        return pattern
    
    def _analyze_activity_pattern(self, data):
        """Analyze activity level patterns"""
        activity_cols = [col for col in ['steps', 'exercise_minutes', 'calories_burned'] if col in data.columns]
        
        if not activity_cols:
            return {'name': 'Activity Pattern', 'confidence': 0}
        
        # Calculate activity consistency
        activity_data = data[activity_cols[0]]
        cv = activity_data.std() / activity_data.mean()  # Coefficient of variation
        
        # Pattern strength based on consistency
        confidence = max(0.5, 1 - cv) if cv < 2 else 0.3
        
        pattern = {
            'name': 'Activity Pattern',
            'description': f'Activity levels show {"consistent" if cv < 0.5 else "variable"} pattern',
            'confidence': confidence,
            'frequency': 'Daily',
            'consistency_score': 1 - cv,
            'average_activity': activity_data.mean(),
            'insight': self._get_activity_insight(cv, activity_data.mean())
        }
        
        return pattern
    
    def _analyze_weekly_cycles(self, data):
        """Analyze weekly behavioral cycles"""
        if 'date' not in data.columns:
            return {'name': 'Weekly Cycles', 'confidence': 0}
        
        # Add weekday information
        data['weekday'] = pd.to_datetime(data['date']).dt.weekday
        
        # Analyze patterns by weekday
        weekday_patterns = {}
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols[:3]:  # Analyze top 3 numerical columns
            weekday_means = data.groupby('weekday')[col].mean()
            weekday_patterns[col] = weekday_means.to_dict()
        
        # Calculate pattern strength
        pattern_strength = 0
        for col, weekday_data in weekday_patterns.items():
            values = list(weekday_data.values())
            cv = np.std(values) / np.mean(values)
            pattern_strength += cv
        
        pattern_strength = min(pattern_strength / len(weekday_patterns), 1.0)
        
        pattern = {
            'name': 'Weekly Cycles',
            'description': f'Weekly patterns show {"strong" if pattern_strength > 0.3 else "weak"} variation',
            'confidence': pattern_strength,
            'frequency': 'Weekly',
            'weekday_patterns': weekday_patterns,
            'insight': self._get_weekly_insight(weekday_patterns)
        }
        
        return pattern
    
    def _analyze_trends(self, data, lookback_days):
        """Analyze trending patterns in the data"""
        trends = []
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols[:5]:  # Analyze top 5 numerical columns
            if len(data) >= lookback_days:
                recent_data = data[col].tail(lookback_days)
                
                # Calculate trend using linear regression slope
                x = np.arange(len(recent_data))
                slope = np.polyfit(x, recent_data, 1)[0]
                
                # Normalize slope to get confidence
                confidence = min(abs(slope) / recent_data.std(), 1.0) if recent_data.std() > 0 else 0
                
                if confidence > 0.3:
                    trend = {
                        'name': f'{col.replace("_", " ").title()} Trend',
                        'description': f'{col.replace("_", " ").title()} shows {"increasing" if slope > 0 else "decreasing"} trend',
                        'confidence': confidence,
                        'frequency': f'Last {lookback_days} days',
                        'slope': slope,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'insight': self._get_trend_insight(col, slope, confidence)
                    }
                    trends.append(trend)
        
        return trends
    
    def _get_sleep_exercise_insight(self, correlation):
        """Generate insight for sleep-exercise pattern"""
        if correlation > 0.5:
            return "Strong positive correlation: More exercise tends to improve sleep quality"
        elif correlation > 0.2:
            return "Moderate positive correlation: Exercise may contribute to better sleep"
        elif correlation < -0.2:
            return "Negative correlation: High exercise might be affecting sleep"
        else:
            return "Weak correlation: Sleep and exercise patterns are relatively independent"
    
    def _get_mood_stress_insight(self, correlation, high_stress_episodes, total_records):
        """Generate insight for mood-stress pattern"""
        stress_percentage = (high_stress_episodes / total_records) * 100
        
        if correlation < -0.5:
            return f"Strong inverse relationship: High stress significantly impacts mood ({stress_percentage:.1f}% of days)"
        elif correlation < -0.3:
            return f"Moderate inverse relationship: Stress affects mood ({stress_percentage:.1f}% of days)"
        else:
            return f"Weak relationship: Stress and mood are relatively independent ({stress_percentage:.1f}% high stress days)"
    
    def _get_activity_insight(self, cv, mean_activity):
        """Generate insight for activity pattern"""
        if cv < 0.3:
            return f"Very consistent activity levels (avg: {mean_activity:.1f})"
        elif cv < 0.7:
            return f"Moderately consistent activity levels (avg: {mean_activity:.1f})"
        else:
            return f"Highly variable activity levels (avg: {mean_activity:.1f})"
    
    def _get_weekly_insight(self, weekday_patterns):
        """Generate insight for weekly patterns"""
        insights = []
        
        for metric, pattern in weekday_patterns.items():
            weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            values = [pattern.get(i, 0) for i in range(7)]
            
            max_day = weekday_names[np.argmax(values)]
            min_day = weekday_names[np.argmin(values)]
            
            insights.append(f"{metric}: Highest on {max_day}, Lowest on {min_day}")
        
        return "; ".join(insights[:2])  # Return top 2 insights
    
    def _get_trend_insight(self, column, slope, confidence):
        """Generate insight for trend patterns"""
        direction = "increasing" if slope > 0 else "decreasing"
        strength = "strong" if confidence > 0.7 else "moderate" if confidence > 0.5 else "weak"
        
        return f"{column.replace('_', ' ').title()} shows {strength} {direction} trend"
    
    def perform_clustering(self, data, n_clusters=3):
        """Perform clustering analysis on health data"""
        try:
            # Prepare data
            X = data.select_dtypes(include=[np.number]).values
            
            if X.shape[0] < n_clusters:
                return None
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score
            silhouette = silhouette_score(X_scaled, labels)
            
            result = {
                'labels': labels,
                'centroids': kmeans.cluster_centers_,
                'silhouette_score': silhouette,
                'inertia': kmeans.inertia_,
                'n_clusters': n_clusters
            }
            
            return result
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            return None
    
    def detect_anomalies(self, data, contamination=0.1):
        """Detect anomalies in health data"""
        try:
            # Prepare data
            X = data.select_dtypes(include=[np.number]).values
            
            if X.shape[0] < 10:
                return None
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Detect anomalies
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            anomaly_scores = iso_forest.decision_function(X_scaled)
            
            # Convert labels to boolean (True for anomaly)
            anomalies = anomaly_labels == -1
            
            result = {
                'anomalies': anomalies,
                'scores': anomaly_scores,
                'contamination': contamination,
                'n_anomalies': sum(anomalies)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
            return None
    
    def analyze_feature_importance(self, data, target_column):
        """Analyze feature importance for a target variable"""
        try:
            if target_column not in data.columns:
                return None
            
            # Prepare features and target
            feature_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                           if col != target_column]
            
            if len(feature_cols) == 0:
                return None
            
            X = data[feature_cols]
            y = data[target_column]
            
            # Convert to classification problem if needed
            if y.dtype in ['float64', 'int64']:
                # Convert to categories based on quartiles
                y = pd.qcut(y, q=3, labels=['Low', 'Medium', 'High'])
            
            # Train random forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = dict(zip(feature_cols, rf.feature_importances_))
            
            return importance
            
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
            return None
    
    def generate_behavior_summary(self, patterns):
        """Generate a comprehensive behavior summary"""
        if not patterns:
            return "No significant behavioral patterns detected."
        
        summary = []
        
        # Categorize patterns
        high_confidence = [p for p in patterns if p['confidence'] > 0.7]
        medium_confidence = [p for p in patterns if 0.5 <= p['confidence'] <= 0.7]
        
        if high_confidence:
            summary.append(f"Strong patterns detected: {len(high_confidence)}")
            for pattern in high_confidence[:3]:  # Top 3
                summary.append(f"• {pattern['name']}: {pattern['description']}")
        
        if medium_confidence:
            summary.append(f"Moderate patterns detected: {len(medium_confidence)}")
        
        # Add recommendations based on patterns
        recommendations = self._generate_pattern_recommendations(patterns)
        if recommendations:
            summary.append("Recommendations:")
            summary.extend([f"• {rec}" for rec in recommendations[:3]])
        
        return "\n".join(summary)
    
    def _generate_pattern_recommendations(self, patterns):
        """Generate recommendations based on detected patterns"""
        recommendations = []
        
        for pattern in patterns:
            if pattern['name'] == 'Sleep-Exercise Correlation' and pattern['confidence'] > 0.5:
                if pattern['correlation_value'] > 0:
                    recommendations.append("Continue maintaining regular exercise to support good sleep")
                else:
                    recommendations.append("Consider adjusting exercise timing to improve sleep quality")
            
            elif pattern['name'] == 'Mood-Stress Relationship' and pattern['confidence'] > 0.5:
                recommendations.append("Focus on stress management techniques to improve mood")
            
            elif 'Trend' in pattern['name'] and pattern['confidence'] > 0.6:
                if pattern['direction'] == 'decreasing':
                    recommendations.append(f"Address declining {pattern['name'].split()[0].lower()} trend")
                else:
                    recommendations.append(f"Maintain positive {pattern['name'].split()[0].lower()} trend")
        
        return recommendations
