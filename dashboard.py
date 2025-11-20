"""
Learning Progress Dashboard & Analytics
Tracks user progress, quiz performance, and learning patterns
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

class LearningDashboard:
    def __init__(self, user_kb_file):
        self.user_kb_file = user_kb_file
        self.kb = self.load_kb()
    
    def load_kb(self):
        """Load user's knowledge base"""
        if Path(self.user_kb_file).exists():
            return pd.read_csv(self.user_kb_file)
        return pd.DataFrame()
    
    def get_topic_mastery(self):
        """Calculate mastery level for each topic"""
        if self.kb.empty:
            return {}
        
        topic_stats = defaultdict(lambda: {'level': 'novice', 'interactions': 0, 'avg_score': 0})
        
        for _, row in self.kb.iterrows():
            topic = row['topic']
            if pd.notna(topic):
                topic_stats[topic]['interactions'] += 1
                
                # Update level based on assessments and quizzes
                if row['type'] in ['assessment', 'quiz'] and pd.notna(row['level']):
                    topic_stats[topic]['level'] = row['level']
                
                # Calculate average score
                if pd.notna(row['quiz_score']) and '/' in str(row['quiz_score']):
                    try:
                        score_parts = str(row['quiz_score']).split('/')
                        score = float(score_parts[0].split('(')[0].strip())
                        total = float(score_parts[1].split('(')[0].strip())
                        topic_stats[topic]['avg_score'] = (score / total) * 100
                    except:
                        pass
        
        return dict(topic_stats)
    
    def get_quiz_performance(self):
        """Get quiz performance history"""
        if self.kb.empty:
            return []
        
        quizzes = self.kb[self.kb['type'] == 'quiz'].copy()
        performance = []
        
        for _, row in quizzes.iterrows():
            if pd.notna(row['quiz_score']):
                try:
                    score_str = str(row['quiz_score'])
                    if '/' in score_str:
                        score_parts = score_str.split('/')
                        score = float(score_parts[0].split('(')[0].strip())
                        total = float(score_parts[1].split('(')[0].strip())
                        percentage = (score / total) * 100
                        
                        performance.append({
                            'timestamp': row['timestamp'],
                            'topic': row['topic'],
                            'score': score,
                            'total': total,
                            'percentage': percentage,
                            'level': row.get('level', 'unknown')
                        })
                except:
                    pass
        
        return performance
    
    def get_learning_streak(self):
        """Calculate learning streak (consecutive days)"""
        if self.kb.empty:
            return 0
        
        # Get unique dates
        dates = pd.to_datetime(self.kb['timestamp']).dt.date.unique()
        dates = sorted(dates, reverse=True)
        
        if not len(dates):
            return 0
        
        # Check consecutive days
        streak = 1
        for i in range(len(dates) - 1):
            diff = (dates[i] - dates[i + 1]).days
            if diff == 1:
                streak += 1
            else:
                break
        
        return streak
    
    def get_total_study_time(self):
        """Estimate total study time based on interactions"""
        if self.kb.empty:
            return 0
        
        # Estimate: query=2min, quiz=5min, assessment=3min
        time_estimates = {
            'query': 2,
            'quiz': 5,
            'assessment': 3
        }
        
        total_minutes = 0
        for _, row in self.kb.iterrows():
            entry_type = row.get('type', 'query')
            total_minutes += time_estimates.get(entry_type, 2)
        
        return total_minutes
    
    def get_weak_topics(self):
        """Identify topics that need more practice"""
        topic_mastery = self.get_topic_mastery()
        weak_topics = []
        
        for topic, stats in topic_mastery.items():
            if stats['level'] in ['novice', 'beginner'] or stats['avg_score'] < 60:
                weak_topics.append({
                    'topic': topic,
                    'level': stats['level'],
                    'score': stats['avg_score'],
                    'interactions': stats['interactions']
                })
        
        return sorted(weak_topics, key=lambda x: x['score'])
    
    def get_strong_topics(self):
        """Identify mastered topics"""
        topic_mastery = self.get_topic_mastery()
        strong_topics = []
        
        for topic, stats in topic_mastery.items():
            if stats['level'] in ['intermediate', 'advanced'] or stats['avg_score'] >= 80:
                strong_topics.append({
                    'topic': topic,
                    'level': stats['level'],
                    'score': stats['avg_score'],
                    'interactions': stats['interactions']
                })
        
        return sorted(strong_topics, key=lambda x: x['score'], reverse=True)
    
    def get_recent_activity(self, days=7):
        """Get activity in last N days"""
        if self.kb.empty:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = self.kb[pd.to_datetime(self.kb['timestamp']) >= cutoff_date]
        
        activity = []
        for _, row in recent.iterrows():
            activity.append({
                'timestamp': row['timestamp'],
                'query': row['query'][:50] + '...' if len(str(row['query'])) > 50 else row['query'],
                'topic': row['topic'],
                'type': row['type']
            })
        
        return activity
    
    def generate_report(self):
        """Generate comprehensive learning report"""
        report = {
            'total_interactions': len(self.kb),
            'total_topics': len(self.kb['topic'].unique()) if not self.kb.empty else 0,
            'learning_streak': self.get_learning_streak(),
            'study_time_minutes': self.get_total_study_time(),
            'topic_mastery': self.get_topic_mastery(),
            'quiz_performance': self.get_quiz_performance(),
            'weak_topics': self.get_weak_topics(),
            'strong_topics': self.get_strong_topics(),
            'recent_activity': self.get_recent_activity()
        }
        
        return report
    
    def display_dashboard(self):
        """Display formatted dashboard"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("📊 LEARNING PROGRESS DASHBOARD")
        print("="*60)
        
        # Overview
        print("\n📈 OVERVIEW:")
        print(f"  Total Interactions: {report['total_interactions']}")
        print(f"  Topics Explored: {report['total_topics']}")
        print(f"  Learning Streak: {report['learning_streak']} days 🔥")
        print(f"  Study Time: {report['study_time_minutes']} minutes ({report['study_time_minutes']//60}h {report['study_time_minutes']%60}m)")
        
        # Quiz Performance
        if report['quiz_performance']:
            print("\n🎯 QUIZ PERFORMANCE:")
            total_quizzes = len(report['quiz_performance'])
            avg_score = sum(q['percentage'] for q in report['quiz_performance']) / total_quizzes
            print(f"  Total Quizzes: {total_quizzes}")
            print(f"  Average Score: {avg_score:.1f}%")
            
            print("\n  Recent Quizzes:")
            for quiz in report['quiz_performance'][-3:]:
                print(f"    • {quiz['topic']}: {quiz['score']}/{quiz['total']} ({quiz['percentage']:.1f}%) - {quiz['level']}")
        
        # Strong Topics
        if report['strong_topics']:
            print("\n💪 STRONG TOPICS:")
            for topic in report['strong_topics'][:5]:
                print(f"  ✅ {topic['topic']}: {topic['level']} ({topic['score']:.0f}%)")
        
        # Weak Topics
        if report['weak_topics']:
            print("\n📚 NEEDS PRACTICE:")
            for topic in report['weak_topics'][:5]:
                print(f"  ⚠️  {topic['topic']}: {topic['level']} ({topic['score']:.0f}%)")
        
        # Recent Activity
        if report['recent_activity']:
            print("\n🕐 RECENT ACTIVITY (Last 7 Days):")
            for activity in report['recent_activity'][-5:]:
                timestamp = pd.to_datetime(activity['timestamp']).strftime('%m/%d %H:%M')
                print(f"  • [{timestamp}] {activity['type']}: {activity['query']}")
        
        print("\n" + "="*60)
        
        return report
    
    def get_recommendations(self):
        """Generate personalized recommendations"""
        report = self.generate_report()
        recommendations = []
        
        # Based on weak topics
        if report['weak_topics']:
            weak = report['weak_topics'][0]
            recommendations.append(f"📖 Focus on '{weak['topic']}' - Take a quiz to improve from {weak['level']} level")
        
        # Based on streak
        if report['learning_streak'] == 0:
            recommendations.append("🔥 Start a learning streak! Study daily to build momentum")
        elif report['learning_streak'] < 7:
            recommendations.append(f"🔥 Keep going! You're on a {report['learning_streak']}-day streak")
        
        # Based on quiz performance
        if report['quiz_performance']:
            recent_avg = sum(q['percentage'] for q in report['quiz_performance'][-3:]) / min(3, len(report['quiz_performance']))
            if recent_avg < 70:
                recommendations.append("💡 Your recent quiz scores are low - Review fundamentals before advancing")
            elif recent_avg > 85:
                recommendations.append("🚀 Great scores! Try harder difficulty levels to challenge yourself")
        
        # Based on activity
        if report['total_interactions'] < 10:
            recommendations.append("🌱 You're just getting started! Explore more topics to build your knowledge base")
        
        return recommendations
