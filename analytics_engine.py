"""
Advanced Analytics Engine
- Learning pattern prediction
- Personalized study schedules
- Knowledge gap identification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json

PROFILES_DIR = Path('personalized_rag/user_profiles')

class AnalyticsEngine:
    def __init__(self, username):
        self.username = username
        self.kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
        self.profile_file = PROFILES_DIR / f"{username}_profile.json"
        
    def load_data(self):
        """Load user data"""
        if not self.kb_file.exists():
            return pd.DataFrame()
        return pd.read_csv(self.kb_file)
    
    def load_profile(self):
        """Load user profile"""
        if not self.profile_file.exists():
            return {}
        with open(self.profile_file) as f:
            return json.load(f)
    
    def predict_learning_patterns(self):
        """Analyze and predict learning patterns"""
        df = self.load_data()
        if df.empty:
            return {"status": "insufficient_data"}
        
        # Time-based patterns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        
        # Peak learning hours
        hour_counts = df['hour'].value_counts().head(3)
        peak_hours = [f"{h}:00" for h in hour_counts.index]
        
        # Active days
        day_counts = df['day'].value_counts().head(3)
        active_days = day_counts.index.tolist()
        
        # Topic progression
        topics = df[df['type'] == 'query']['topic'].dropna()
        topic_freq = Counter(topics)
        trending_topics = [t for t, _ in topic_freq.most_common(3)]
        
        # Learning velocity (queries per day)
        days_active = (df['timestamp'].max() - df['timestamp'].min()).days or 1
        velocity = len(df) / days_active
        
        # Predict next session
        last_session = df['timestamp'].max()
        avg_gap = df['timestamp'].diff().mean()
        next_session = last_session + avg_gap if pd.notna(avg_gap) else last_session + timedelta(days=1)
        
        return {
            "peak_hours": peak_hours,
            "active_days": active_days,
            "trending_topics": trending_topics,
            "learning_velocity": round(velocity, 2),
            "next_predicted_session": next_session.strftime("%Y-%m-%d %H:%M"),
            "total_queries": len(df),
            "consistency_score": self._calculate_consistency(df)
        }
    
    def _calculate_consistency(self, df):
        """Calculate learning consistency (0-100)"""
        if len(df) < 7:
            return 50
        
        # Check daily activity over last 30 days
        recent = df[df['timestamp'] > datetime.now() - timedelta(days=30)]
        days_with_activity = recent['timestamp'].dt.date.nunique()
        consistency = min(100, (days_with_activity / 30) * 100)
        return round(consistency)
    
    def generate_study_schedule(self):
        """Generate personalized study schedule"""
        df = self.load_data()
        profile = self.load_profile()
        patterns = self.predict_learning_patterns()
        
        if df.empty:
            return self._default_schedule()
        
        # Get user's best learning times
        peak_hours = patterns.get('peak_hours', ['09:00', '14:00', '20:00'])
        active_days = patterns.get('active_days', ['Monday', 'Wednesday', 'Friday'])
        
        # Topic distribution
        topics = df[df['type'] == 'query']['topic'].dropna()
        topic_freq = Counter(topics)
        
        # Create schedule
        schedule = []
        for day in active_days[:5]:  # Max 5 days
            for i, (topic, count) in enumerate(topic_freq.most_common(3)):
                if i < len(peak_hours):
                    schedule.append({
                        "day": day,
                        "time": peak_hours[i],
                        "topic": topic,
                        "duration": "30-45 min",
                        "priority": "high" if count > 5 else "medium"
                    })
        
        return {
            "schedule": schedule[:10],  # Top 10 sessions
            "recommended_session_length": "30-45 minutes",
            "break_frequency": "Every 45 minutes",
            "weekly_goal": f"{len(schedule)} sessions"
        }
    
    def _default_schedule(self):
        """Default schedule for new users"""
        return {
            "schedule": [
                {"day": "Monday", "time": "09:00", "topic": "General", "duration": "30 min", "priority": "medium"},
                {"day": "Wednesday", "time": "14:00", "topic": "General", "duration": "30 min", "priority": "medium"},
                {"day": "Friday", "time": "20:00", "topic": "General", "duration": "30 min", "priority": "medium"}
            ],
            "recommended_session_length": "30 minutes",
            "break_frequency": "Every 30 minutes",
            "weekly_goal": "3 sessions"
        }
    
    def identify_knowledge_gaps(self):
        """Identify knowledge gaps and weak areas"""
        df = self.load_data()
        profile = self.load_profile()
        
        if df.empty:
            return {"status": "insufficient_data", "gaps": []}
        
        # Analyze topics
        queries = df[df['type'] == 'query']
        topic_counts = Counter(queries['topic'].dropna())
        
        # Identify repeated questions (potential gaps)
        query_texts = queries['query'].dropna().str.lower()
        repeated_queries = [q for q, count in Counter(query_texts).items() if count > 2]
        
        # Extract topics from repeated queries
        gap_topics = []
        for query in repeated_queries[:5]:
            matching = queries[queries['query'].str.lower() == query]
            if not matching.empty:
                topic = matching.iloc[0]['topic']
                gap_topics.append(topic)
        
        # Find underexplored topics
        all_topics = ['mathematics', 'physics', 'chemistry', 'biology', 'computer_science', 
                      'history', 'literature', 'geography', 'economics']
        explored = set(topic_counts.keys())
        unexplored = [t for t in all_topics if t not in explored]
        
        # Calculate mastery levels
        mastery = {}
        for topic, count in topic_counts.items():
            if count < 5:
                level = "beginner"
            elif count < 15:
                level = "intermediate"
            else:
                level = "advanced"
            mastery[topic] = {"queries": count, "level": level}
        
        # Identify weak areas (low query count but present)
        weak_areas = [topic for topic, count in topic_counts.items() if count < 5]
        
        return {
            "weak_areas": weak_areas[:5],
            "repeated_questions": repeated_queries[:5],
            "unexplored_topics": unexplored[:5],
            "mastery_levels": mastery,
            "recommendations": self._generate_recommendations(weak_areas, unexplored)
        }
    
    def _generate_recommendations(self, weak_areas, unexplored):
        """Generate learning recommendations"""
        recommendations = []
        
        if weak_areas:
            recommendations.append(f"Focus on strengthening: {', '.join(weak_areas[:3])}")
        
        if unexplored:
            recommendations.append(f"Explore new topics: {', '.join(unexplored[:3])}")
        
        recommendations.append("Review repeated questions to solidify understanding")
        recommendations.append("Maintain consistent daily practice")
        
        return recommendations
    
    def get_complete_analytics(self):
        """Get all analytics in one call"""
        return {
            "learning_patterns": self.predict_learning_patterns(),
            "study_schedule": self.generate_study_schedule(),
            "knowledge_gaps": self.identify_knowledge_gaps()
        }

def get_analytics(username):
    """Main function to get analytics"""
    engine = AnalyticsEngine(username)
    return engine.get_complete_analytics()
