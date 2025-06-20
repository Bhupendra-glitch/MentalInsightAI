import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from recommendation_engine import RecommendationEngine

st.set_page_config(
    page_title="Recommendations - AI Health Analytics",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Personalized Health Recommendations")
st.markdown("Get AI-powered, personalized health recommendations based on your data and goals.")

# Initialize recommendation engine
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = RecommendationEngine()

# Check if data is available
if not st.session_state.get('data_processed', False):
    st.error("❌ No processed data available. Please upload and process your data first.")
    st.info("👈 Go to the Data Upload page to upload your health data.")
    st.stop()

# Get processed data
data = st.session_state.processed_data

# User profile setup
st.subheader("👤 User Profile")

profile_col1, profile_col2, profile_col3 = st.columns(3)

with profile_col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
with profile_col2:
    height = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)
    weight = st.number_input("Weight (kg)", min_value=40, max_value=200, value=70)
    
with profile_col3:
    activity_level = st.selectbox("Activity Level", 
                                ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    health_goals = st.multiselect("Health Goals", 
                                ["Weight Loss", "Muscle Gain", "Better Sleep", "Stress Reduction", "Improved Mood"])

# User preferences
st.subheader("⚙️ Recommendation Preferences")

pref_col1, pref_col2 = st.columns(2)

with pref_col1:
    recommendation_type = st.selectbox("Recommendation Type", 
                                     ["All", "Exercise", "Nutrition", "Sleep", "Wellness"])
    priority = st.selectbox("Priority", ["Health", "Convenience", "Efficiency"])

with pref_col2:
    time_availability = st.slider("Available Time per Day (minutes)", 0, 180, 60)
    difficulty_preference = st.selectbox("Difficulty Preference", 
                                       ["Beginner", "Intermediate", "Advanced"])

# Generate recommendations
if st.button("✨ Generate Recommendations", type="primary"):
    with st.spinner("Generating personalized recommendations..."):
        # Create user profile
        user_profile = {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'activity_level': activity_level,
            'health_goals': health_goals,
            'recommendation_type': recommendation_type,
            'priority': priority,
            'time_availability': time_availability,
            'difficulty_preference': difficulty_preference
        }
        
        # Generate recommendations
        recommendations = st.session_state.recommendation_engine.generate_recommendations(
            data, user_profile
        )
        
        if recommendations:
            st.session_state.current_recommendations = recommendations
            st.success(f"✅ Generated {len(recommendations)} personalized recommendations!")
        else:
            st.error("❌ Failed to generate recommendations. Please check your data and profile.")

# Display recommendations
if st.session_state.get('current_recommendations'):
    recommendations = st.session_state.current_recommendations
    
    st.subheader("🎯 Your Personalized Recommendations")
    
    # Filter recommendations by type
    if recommendation_type != "All":
        filtered_recommendations = [r for r in recommendations if r['category'].lower() == recommendation_type.lower()]
    else:
        filtered_recommendations = recommendations
    
    # Display recommendations in tabs
    if filtered_recommendations:
        # Create tabs for different categories
        categories = list(set([r['category'] for r in filtered_recommendations]))
        
        if len(categories) > 1:
            tabs = st.tabs(categories)
            
            for i, category in enumerate(categories):
                with tabs[i]:
                    category_recommendations = [r for r in filtered_recommendations if r['category'] == category]
                    display_recommendations(category_recommendations)
        else:
            display_recommendations(filtered_recommendations)
    else:
        st.info("No recommendations found for the selected type. Try selecting 'All' or different preferences.")

def display_recommendations(recommendations):
    """Display recommendations in a structured format"""
    for i, rec in enumerate(recommendations):
        with st.expander(f"💡 {rec['title']} - {rec['category']}", expanded=i < 3):
            
            # Recommendation details
            detail_col1, detail_col2 = st.columns([2, 1])
            
            with detail_col1:
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Rationale:** {rec['rationale']}")
                
                if 'action_steps' in rec:
                    st.write("**Action Steps:**")
                    for step in rec['action_steps']:
                        st.write(f"• {step}")
                
                if 'expected_benefits' in rec:
                    st.write("**Expected Benefits:**")
                    for benefit in rec['expected_benefits']:
                        st.write(f"• {benefit}")
            
            with detail_col2:
                st.metric("Priority Score", f"{rec['priority_score']:.1f}/10")
                st.metric("Difficulty", rec['difficulty'])
                st.metric("Time Required", rec['time_required'])
                
                if 'confidence' in rec:
                    st.metric("Confidence", f"{rec['confidence']:.0%}")
            
            # Progress tracking
            if st.button(f"📊 Track Progress", key=f"track_{i}"):
                st.info("Progress tracking feature would be implemented here.")
            
            # Feedback
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                if st.button("👍 Helpful", key=f"helpful_{i}"):
                    st.success("Thank you for your feedback!")
                    
            with feedback_col2:
                if st.button("👎 Not Helpful", key=f"not_helpful_{i}"):
                    st.info("We'll use your feedback to improve recommendations.")

# Recommendation insights
if st.session_state.get('current_recommendations'):
    st.subheader("📊 Recommendation Insights")
    
    recommendations = st.session_state.current_recommendations
    
    # Category distribution
    category_counts = {}
    for rec in recommendations:
        category = rec['category']
        category_counts[category] = category_counts.get(category, 0) + 1
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        if category_counts:
            fig = px.pie(values=list(category_counts.values()), 
                        names=list(category_counts.keys()),
                        title="Recommendations by Category")
            st.plotly_chart(fig, use_container_width=True)
    
    with insight_col2:
        # Priority distribution
        priorities = [rec['priority_score'] for rec in recommendations]
        fig = px.histogram(x=priorities, nbins=10, title="Priority Score Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Top recommendations summary
    st.subheader("🏆 Top Recommendations Summary")
    
    # Sort by priority score
    top_recommendations = sorted(recommendations, key=lambda x: x['priority_score'], reverse=True)[:5]
    
    for i, rec in enumerate(top_recommendations):
        st.write(f"**{i+1}. {rec['title']}** ({rec['category']}) - Priority: {rec['priority_score']:.1f}/10")

# Recommendation history
st.subheader("📚 Recommendation History")

if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

if st.session_state.get('current_recommendations'):
    if st.button("💾 Save Current Recommendations"):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.recommendation_history.append({
            'timestamp': timestamp,
            'recommendations': st.session_state.current_recommendations,
            'user_profile': {
                'age': age,
                'gender': gender,
                'health_goals': health_goals,
                'activity_level': activity_level
            }
        })
        st.success("✅ Recommendations saved to history!")

if st.session_state.recommendation_history:
    st.write(f"**History:** {len(st.session_state.recommendation_history)} saved recommendation sets")
    
    # Show recent history
    for i, history_item in enumerate(reversed(st.session_state.recommendation_history[-3:])):
        with st.expander(f"📅 {history_item['timestamp']}"):
            st.write(f"**Goals:** {', '.join(history_item['user_profile']['health_goals'])}")
            st.write(f"**Activity Level:** {history_item['user_profile']['activity_level']}")
            st.write(f"**Recommendations:** {len(history_item['recommendations'])}")
else:
    st.info("No recommendation history yet. Generate and save recommendations to build your history.")

# Export recommendations
if st.session_state.get('current_recommendations'):
    st.subheader("📥 Export Recommendations")
    
    if st.button("📄 Export as PDF Report"):
        st.info("PDF export feature would be implemented here.")
    
    if st.button("📧 Email Recommendations"):
        st.info("Email feature would be implemented here.")

# Recommendation tips
st.sidebar.subheader("💡 Tips for Better Recommendations")
st.sidebar.markdown("""
- **Complete your profile** with accurate information
- **Set specific health goals** for targeted recommendations
- **Provide feedback** on recommendations to improve future suggestions
- **Track your progress** to see what's working
- **Update your data regularly** for more accurate recommendations
""")

# Recommendation settings
st.sidebar.subheader("⚙️ Recommendation Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh recommendations", value=False)
notification_preferences = st.sidebar.multiselect("Notification Types", 
                                                  ["Daily Tips", "Weekly Summary", "Goal Reminders"])

if st.sidebar.button("🔄 Reset Recommendations"):
    if 'current_recommendations' in st.session_state:
        del st.session_state.current_recommendations
    st.sidebar.success("Recommendations reset!")
    st.rerun()
