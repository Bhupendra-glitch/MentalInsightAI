import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RecommendationEngine:
    """AI-powered recommendation engine for personalized health insights"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.user_profiles = {}
        self.recommendation_history = []
        
    def generate_recommendations(self, data, user_profile):
        """Generate personalized health recommendations based on user data and profile"""
        try:
            recommendations = []
            
            # Generate different types of recommendations
            exercise_recs = self._generate_exercise_recommendations(data, user_profile)
            nutrition_recs = self._generate_nutrition_recommendations(data, user_profile)
            sleep_recs = self._generate_sleep_recommendations(data, user_profile)
            wellness_recs = self._generate_wellness_recommendations(data, user_profile)
            
            # Combine all recommendations
            all_recs = exercise_recs + nutrition_recs + sleep_recs + wellness_recs
            
            # Filter by user preferences
            filtered_recs = self._filter_by_preferences(all_recs, user_profile)
            
            # Rank recommendations by priority
            ranked_recs = self._rank_recommendations(filtered_recs, user_profile, data)
            
            return ranked_recs[:10]  # Return top 10 recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return []
    
    def _generate_exercise_recommendations(self, data, user_profile):
        """Generate exercise-specific recommendations"""
        recommendations = []
        
        # Analyze current exercise patterns
        exercise_data = self._get_exercise_metrics(data)
        
        if not exercise_data:
            # No exercise data available - provide general recommendations
            return self._get_basic_exercise_recommendations(user_profile)
        
        avg_exercise = exercise_data.get('avg_minutes', 0)
        consistency = exercise_data.get('consistency', 0)
        
        # Low exercise recommendations
        if avg_exercise < 30:
            recommendations.append({
                'title': 'Increase Daily Physical Activity',
                'category': 'Exercise',
                'description': 'Gradually increase your daily exercise to reach recommended 30+ minutes',
                'rationale': f'Current average: {avg_exercise:.1f} minutes/day. WHO recommends 150 minutes/week.',
                'action_steps': [
                    'Start with 10-minute walks after meals',
                    'Take stairs instead of elevators',
                    'Set reminders for movement breaks',
                    'Find activities you enjoy (dancing, swimming, cycling)'
                ],
                'expected_benefits': [
                    'Improved cardiovascular health',
                    'Better mood and energy levels',
                    'Enhanced sleep quality',
                    'Weight management'
                ],
                'priority_score': 8.5,
                'difficulty': 'Beginner',
                'time_required': '10-30 minutes/day',
                'confidence': 0.9
            })
        
        # Inconsistent exercise recommendations
        if consistency < 0.6:
            recommendations.append({
                'title': 'Build Consistent Exercise Routine',
                'category': 'Exercise',
                'description': 'Develop a regular exercise schedule to improve consistency',
                'rationale': f'Exercise consistency is {consistency:.1%}. Regular activity is more beneficial than sporadic intense sessions.',
                'action_steps': [
                    'Schedule specific workout times',
                    'Start with 3 days per week',
                    'Use fitness apps for accountability',
                    'Prepare workout clothes in advance'
                ],
                'expected_benefits': [
                    'Better habit formation',
                    'Improved fitness progression',
                    'Reduced injury risk',
                    'Enhanced motivation'
                ],
                'priority_score': 7.8,
                'difficulty': 'Intermediate',
                'time_required': '30-45 minutes, 3x/week',
                'confidence': 0.85
            })
        
        # Activity level specific recommendations
        if user_profile.get('activity_level') == 'Sedentary':
            recommendations.append({
                'title': 'Combat Sedentary Lifestyle',
                'category': 'Exercise',
                'description': 'Incorporate movement throughout your day to reduce sedentary time',
                'rationale': 'Extended sitting increases health risks. Regular movement breaks are essential.',
                'action_steps': [
                    'Set hourly movement reminders',
                    'Use standing desk for part of day',
                    'Walk during phone calls',
                    'Park farther away or get off bus early'
                ],
                'expected_benefits': [
                    'Reduced back pain',
                    'Improved circulation',
                    'Better posture',
                    'Increased energy'
                ],
                'priority_score': 8.0,
                'difficulty': 'Beginner',
                'time_required': '5 minutes every hour',
                'confidence': 0.88
            })
        
        return recommendations
    
    def _generate_nutrition_recommendations(self, data, user_profile):
        """Generate nutrition-specific recommendations"""
        recommendations = []
        
        # Analyze hydration patterns
        hydration_data = self._get_hydration_metrics(data)
        
        if hydration_data and hydration_data.get('avg_intake', 0) < 2000:  # Less than 2L per day
            recommendations.append({
                'title': 'Improve Daily Hydration',
                'category': 'Nutrition',
                'description': 'Increase water intake to meet daily hydration needs',
                'rationale': f'Current average: {hydration_data.get("avg_intake", 0):.0f}ml/day. Recommended: 2000-2500ml/day.',
                'action_steps': [
                    'Carry a water bottle throughout the day',
                    'Set hydration reminders on phone',
                    'Drink water before each meal',
                    'Add fruit slices for flavor variety'
                ],
                'expected_benefits': [
                    'Better energy levels',
                    'Improved skin health',
                    'Enhanced cognitive function',
                    'Better temperature regulation'
                ],
                'priority_score': 7.5,
                'difficulty': 'Beginner',
                'time_required': 'Throughout the day',
                'confidence': 0.82
            })
        
        # Weight management recommendations
        if 'Weight Loss' in user_profile.get('health_goals', []):
            recommendations.append({
                'title': 'Sustainable Weight Management',
                'category': 'Nutrition',
                'description': 'Adopt healthy eating habits for sustainable weight loss',
                'rationale': 'Gradual, sustainable changes lead to long-term success.',
                'action_steps': [
                    'Practice portion control',
                    'Eat more vegetables and lean proteins',
                    'Reduce processed foods and added sugars',
                    'Keep a food diary for awareness'
                ],
                'expected_benefits': [
                    'Gradual, sustainable weight loss',
                    'Improved energy levels',
                    'Better overall health markers',
                    'Enhanced self-confidence'
                ],
                'priority_score': 8.2,
                'difficulty': 'Intermediate',
                'time_required': 'Ongoing lifestyle change',
                'confidence': 0.87
            })
        
        # General nutrition recommendations
        recommendations.append({
            'title': 'Optimize Nutrient Timing',
            'category': 'Nutrition',
            'description': 'Time your meals and snacks for optimal energy and recovery',
            'rationale': 'Proper nutrient timing can improve energy levels and exercise recovery.',
            'action_steps': [
                'Eat protein within 2 hours after exercise',
                'Have balanced meals every 3-4 hours',
                'Include healthy fats in each meal',
                'Avoid large meals before bedtime'
            ],
            'expected_benefits': [
                'Stable energy throughout day',
                'Better exercise recovery',
                'Improved sleep quality',
                'Enhanced nutrient absorption'
            ],
            'priority_score': 6.8,
            'difficulty': 'Intermediate',
            'time_required': 'Meal planning time',
            'confidence': 0.78
        })
        
        return recommendations
    
    def _generate_sleep_recommendations(self, data, user_profile):
        """Generate sleep-specific recommendations"""
        recommendations = []
        
        # Analyze sleep patterns
        sleep_data = self._get_sleep_metrics(data)
        
        if not sleep_data:
            return self._get_basic_sleep_recommendations()
        
        avg_sleep = sleep_data.get('avg_hours', 7)
        sleep_consistency = sleep_data.get('consistency', 0.7)
        
        # Insufficient sleep recommendations
        if avg_sleep < 7:
            recommendations.append({
                'title': 'Increase Sleep Duration',
                'category': 'Sleep',
                'description': 'Extend your sleep time to meet recommended 7-9 hours',
                'rationale': f'Current average: {avg_sleep:.1f} hours. Adults need 7-9 hours for optimal health.',
                'action_steps': [
                    'Set a consistent bedtime 30 minutes earlier',
                    'Create a relaxing pre-sleep routine',
                    'Avoid screens 1 hour before bed',
                    'Keep bedroom cool, dark, and quiet'
                ],
                'expected_benefits': [
                    'Improved cognitive function',
                    'Better mood regulation',
                    'Enhanced immune system',
                    'Better physical recovery'
                ],
                'priority_score': 9.0,
                'difficulty': 'Intermediate',
                'time_required': 'Evening routine adjustment',
                'confidence': 0.92
            })
        
        # Sleep consistency recommendations
        if sleep_consistency < 0.7:
            recommendations.append({
                'title': 'Establish Regular Sleep Schedule',
                'category': 'Sleep',
                'description': 'Create consistent sleep and wake times to improve sleep quality',
                'rationale': 'Irregular sleep patterns disrupt circadian rhythm and reduce sleep quality.',
                'action_steps': [
                    'Go to bed and wake up at same time daily',
                    'Avoid sleeping in on weekends',
                    'Use natural light exposure in morning',
                    'Limit daytime naps to 20 minutes'
                ],
                'expected_benefits': [
                    'Better sleep quality',
                    'Easier time falling asleep',
                    'More energy during day',
                    'Improved mood stability'
                ],
                'priority_score': 8.5,
                'difficulty': 'Intermediate',
                'time_required': '2-3 weeks to establish',
                'confidence': 0.89
            })
        
        # Sleep hygiene recommendations
        if 'Better Sleep' in user_profile.get('health_goals', []):
            recommendations.append({
                'title': 'Optimize Sleep Environment',
                'category': 'Sleep',
                'description': 'Create ideal conditions for restorative sleep',
                'rationale': 'Environmental factors significantly impact sleep quality.',
                'action_steps': [
                    'Maintain bedroom temperature 65-68°F (18-20°C)',
                    'Use blackout curtains or eye mask',
                    'Consider white noise machine',
                    'Invest in comfortable mattress and pillows'
                ],
                'expected_benefits': [
                    'Deeper, more restorative sleep',
                    'Fewer sleep interruptions',
                    'Easier time falling asleep',
                    'Better morning alertness'
                ],
                'priority_score': 7.8,
                'difficulty': 'Beginner',
                'time_required': 'One-time setup',
                'confidence': 0.85
            })
        
        return recommendations
    
    def _generate_wellness_recommendations(self, data, user_profile):
        """Generate wellness and mental health recommendations"""
        recommendations = []
        
        # Stress management
        stress_data = self._get_stress_metrics(data)
        if stress_data and stress_data.get('avg_level', 5) > 6:
            recommendations.append({
                'title': 'Implement Stress Management Techniques',
                'category': 'Wellness',
                'description': 'Develop effective strategies to manage and reduce stress levels',
                'rationale': f'Average stress level: {stress_data.get("avg_level", 5):.1f}/10. High stress impacts overall health.',
                'action_steps': [
                    'Practice deep breathing exercises daily',
                    'Try meditation for 10 minutes each day',
                    'Engage in regular physical activity',
                    'Maintain work-life boundaries'
                ],
                'expected_benefits': [
                    'Reduced anxiety and tension',
                    'Better emotional regulation',
                    'Improved sleep quality',
                    'Enhanced immune function'
                ],
                'priority_score': 8.8,
                'difficulty': 'Beginner',
                'time_required': '10-20 minutes/day',
                'confidence': 0.91
            })
        
        # Mood improvement
        mood_data = self._get_mood_metrics(data)
        if mood_data and mood_data.get('avg_score', 7) < 6:
            recommendations.append({
                'title': 'Boost Daily Mood and Well-being',
                'category': 'Wellness',
                'description': 'Incorporate mood-enhancing activities into your routine',
                'rationale': f'Average mood score: {mood_data.get("avg_score", 7):.1f}/10. Small changes can significantly impact mood.',
                'action_steps': [
                    'Spend time in nature or sunlight daily',
                    'Connect with friends and family regularly',
                    'Practice gratitude journaling',
                    'Engage in hobbies you enjoy'
                ],
                'expected_benefits': [
                    'Improved overall mood',
                    'Better emotional resilience',
                    'Enhanced social connections',
                    'Increased life satisfaction'
                ],
                'priority_score': 8.0,
                'difficulty': 'Beginner',
                'time_required': '15-30 minutes/day',
                'confidence': 0.86
            })
        
        # Mindfulness and mental health
        if 'Stress Reduction' in user_profile.get('health_goals', []):
            recommendations.append({
                'title': 'Develop Mindfulness Practice',
                'category': 'Wellness',
                'description': 'Build awareness and presence through mindfulness techniques',
                'rationale': 'Mindfulness practices reduce stress and improve mental well-being.',
                'action_steps': [
                    'Start with 5-minute guided meditations',
                    'Practice mindful eating at one meal daily',
                    'Use mindfulness apps for guidance',
                    'Focus on breath awareness during breaks'
                ],
                'expected_benefits': [
                    'Reduced stress and anxiety',
                    'Improved focus and concentration',
                    'Better emotional regulation',
                    'Enhanced self-awareness'
                ],
                'priority_score': 7.5,
                'difficulty': 'Beginner',
                'time_required': '5-15 minutes/day',
                'confidence': 0.83
            })
        
        return recommendations
    
    def _filter_by_preferences(self, recommendations, user_profile):
        """Filter recommendations based on user preferences"""
        filtered = []
        
        rec_type = user_profile.get('recommendation_type', 'All')
        difficulty = user_profile.get('difficulty_preference', 'Intermediate')
        time_available = user_profile.get('time_availability', 60)
        
        for rec in recommendations:
            # Filter by type
            if rec_type != 'All' and rec['category'].lower() != rec_type.lower():
                continue
            
            # Filter by difficulty (allow current level and below)
            difficulty_levels = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
            user_level = difficulty_levels.get(difficulty, 2)
            rec_level = difficulty_levels.get(rec['difficulty'], 2)
            
            if rec_level > user_level:
                continue
            
            # Filter by time availability (rough estimate)
            if 'minutes' in rec['time_required'].lower():
                try:
                    req_time = int(rec['time_required'].split()[0].split('-')[0])
                    if req_time > time_available:
                        continue
                except:
                    pass  # Skip time filtering if can't parse
            
            filtered.append(rec)
        
        return filtered
    
    def _rank_recommendations(self, recommendations, user_profile, data):
        """Rank recommendations by priority and relevance"""
        for rec in recommendations:
            # Base priority score
            priority = rec.get('priority_score', 5.0)
            
            # Adjust based on user goals
            health_goals = user_profile.get('health_goals', [])
            
            if rec['category'] == 'Exercise' and any(goal in ['Weight Loss', 'Muscle Gain'] for goal in health_goals):
                priority += 1.0
            
            if rec['category'] == 'Sleep' and 'Better Sleep' in health_goals:
                priority += 1.0
            
            if rec['category'] == 'Wellness' and 'Stress Reduction' in health_goals:
                priority += 1.0
            
            # Adjust based on data patterns
            priority += self._calculate_data_relevance(rec, data)
            
            # Adjust based on user profile
            if user_profile.get('priority') == 'Health':
                priority += 0.5
            elif user_profile.get('priority') == 'Convenience':
                if rec['difficulty'] == 'Beginner':
                    priority += 0.3
            
            rec['priority_score'] = min(priority, 10.0)  # Cap at 10
        
        # Sort by priority score
        return sorted(recommendations, key=lambda x: x['priority_score'], reverse=True)
    
    def _calculate_data_relevance(self, recommendation, data):
        """Calculate how relevant a recommendation is based on user data"""
        relevance_boost = 0.0
        
        # Check if recommendation addresses data patterns
        if recommendation['category'] == 'Exercise':
            exercise_data = self._get_exercise_metrics(data)
            if exercise_data and exercise_data.get('avg_minutes', 30) < 30:
                relevance_boost += 0.5
        
        elif recommendation['category'] == 'Sleep':
            sleep_data = self._get_sleep_metrics(data)
            if sleep_data and sleep_data.get('avg_hours', 7) < 7:
                relevance_boost += 0.5
        
        elif recommendation['category'] == 'Wellness':
            stress_data = self._get_stress_metrics(data)
            mood_data = self._get_mood_metrics(data)
            
            if stress_data and stress_data.get('avg_level', 5) > 6:
                relevance_boost += 0.3
            
            if mood_data and mood_data.get('avg_score', 7) < 6:
                relevance_boost += 0.3
        
        return relevance_boost
    
    def _get_exercise_metrics(self, data):
        """Extract exercise metrics from data"""
        if 'exercise_minutes' not in data.columns:
            return None
        
        return {
            'avg_minutes': data['exercise_minutes'].mean(),
            'consistency': (data['exercise_minutes'] > 0).mean(),
            'max_minutes': data['exercise_minutes'].max(),
            'days_active': sum(data['exercise_minutes'] > 0)
        }
    
    def _get_sleep_metrics(self, data):
        """Extract sleep metrics from data"""
        if 'sleep_hours' not in data.columns:
            return None
        
        return {
            'avg_hours': data['sleep_hours'].mean(),
            'consistency': 1 - (data['sleep_hours'].std() / data['sleep_hours'].mean()),
            'min_hours': data['sleep_hours'].min(),
            'max_hours': data['sleep_hours'].max()
        }
    
    def _get_hydration_metrics(self, data):
        """Extract hydration metrics from data"""
        if 'water_intake' not in data.columns:
            return None
        
        return {
            'avg_intake': data['water_intake'].mean(),
            'consistency': 1 - (data['water_intake'].std() / data['water_intake'].mean()),
            'days_adequate': sum(data['water_intake'] >= 2000)
        }
    
    def _get_stress_metrics(self, data):
        """Extract stress metrics from data"""
        if 'stress_level' not in data.columns:
            return None
        
        return {
            'avg_level': data['stress_level'].mean(),
            'high_stress_days': sum(data['stress_level'] > 7),
            'consistency': data['stress_level'].std()
        }
    
    def _get_mood_metrics(self, data):
        """Extract mood metrics from data"""
        if 'mood_score' not in data.columns:
            return None
        
        return {
            'avg_score': data['mood_score'].mean(),
            'low_mood_days': sum(data['mood_score'] < 5),
            'consistency': data['mood_score'].std()
        }
    
    def _get_basic_exercise_recommendations(self, user_profile):
        """Provide basic exercise recommendations when no data is available"""
        return [{
            'title': 'Start Basic Exercise Routine',
            'category': 'Exercise',
            'description': 'Begin with simple, achievable exercise goals',
            'rationale': 'Regular physical activity is fundamental for health and well-being.',
            'action_steps': [
                'Start with 10-minute daily walks',
                'Add bodyweight exercises 2-3 times per week',
                'Take stairs when possible',
                'Find physical activities you enjoy'
            ],
            'expected_benefits': [
                'Improved cardiovascular health',
                'Better mood and energy',
                'Enhanced sleep quality',
                'Stronger muscles and bones'
            ],
            'priority_score': 8.0,
            'difficulty': 'Beginner',
            'time_required': '10-30 minutes/day',
            'confidence': 0.85
        }]
    
    def _get_basic_sleep_recommendations(self):
        """Provide basic sleep recommendations when no data is available"""
        return [{
            'title': 'Establish Healthy Sleep Habits',
            'category': 'Sleep',
            'description': 'Create a foundation for quality sleep',
            'rationale': 'Good sleep is essential for physical and mental health.',
            'action_steps': [
                'Maintain consistent sleep schedule',
                'Create relaxing bedtime routine',
                'Optimize sleep environment',
                'Limit screen time before bed'
            ],
            'expected_benefits': [
                'Better sleep quality',
                'Improved daytime energy',
                'Enhanced cognitive function',
                'Better mood regulation'
            ],
            'priority_score': 8.5,
            'difficulty': 'Beginner',
            'time_required': 'Evening routine',
            'confidence': 0.88
        }]
