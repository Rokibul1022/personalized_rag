"""
Adaptive Difficulty Adjustment System
Automatically adjusts quiz difficulty based on user performance
"""

import pandas as pd
from pathlib import Path
from collections import deque

class AdaptiveDifficulty:
    def __init__(self, user_kb_file, user_profile_file):
        self.user_kb_file = user_kb_file
        self.user_profile_file = user_profile_file
        self.kb = self.load_kb()
        self.profile = self.load_profile()
        
        # Difficulty thresholds
        self.DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']
        self.PROMOTION_THRESHOLD = 85  # 85%+ to level up
        self.DEMOTION_THRESHOLD = 50   # <50% to level down
        self.CONSECUTIVE_NEEDED = 3     # Need 3 consecutive good/bad scores
    
    def load_kb(self):
        """Load user's knowledge base"""
        if Path(self.user_kb_file).exists():
            return pd.read_csv(self.user_kb_file)
        return pd.DataFrame()
    
    def load_profile(self):
        """Load user profile"""
        import json
        if Path(self.user_profile_file).exists():
            with open(self.user_profile_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_profile(self):
        """Save updated profile"""
        import json
        with open(self.user_profile_file, 'w') as f:
            json.dump(self.profile, f, indent=2)
    
    def get_recent_quiz_scores(self, topic=None, n=5):
        """Get recent quiz scores (optionally filtered by topic)"""
        if self.kb.empty:
            return []
        
        quizzes = self.kb[self.kb['type'] == 'quiz'].copy()
        
        if topic:
            quizzes = quizzes[quizzes['topic'].str.contains(topic, case=False, na=False)]
        
        scores = []
        for _, row in quizzes.tail(n).iterrows():
            if pd.notna(row['quiz_score']) and '/' in str(row['quiz_score']):
                try:
                    score_str = str(row['quiz_score'])
                    score = float(score_str.split('/')[0].split('(')[0].strip())
                    total = float(score_str.split('/')[1].split('(')[0].strip())
                    percentage = (score / total) * 100
                    scores.append({
                        'percentage': percentage,
                        'topic': row['topic'],
                        'timestamp': row['timestamp']
                    })
                except:
                    pass
        
        return scores
    
    def calculate_performance_trend(self, scores):
        """Calculate if user is improving, declining, or stable"""
        if len(scores) < 2:
            return 'stable'
        
        # Compare recent half vs older half
        mid = len(scores) // 2
        older_avg = sum(s['percentage'] for s in scores[:mid]) / mid
        recent_avg = sum(s['percentage'] for s in scores[mid:]) / (len(scores) - mid)
        
        diff = recent_avg - older_avg
        
        if diff > 10:
            return 'improving'
        elif diff < -10:
            return 'declining'
        else:
            return 'stable'
    
    def should_increase_difficulty(self, topic=None):
        """Check if difficulty should be increased"""
        scores = self.get_recent_quiz_scores(topic, n=self.CONSECUTIVE_NEEDED)
        
        if len(scores) < self.CONSECUTIVE_NEEDED:
            return False
        
        # Check if all recent scores are above threshold
        high_scores = sum(1 for s in scores if s['percentage'] >= self.PROMOTION_THRESHOLD)
        
        return high_scores >= self.CONSECUTIVE_NEEDED
    
    def should_decrease_difficulty(self, topic=None):
        """Check if difficulty should be decreased"""
        scores = self.get_recent_quiz_scores(topic, n=self.CONSECUTIVE_NEEDED)
        
        if len(scores) < self.CONSECUTIVE_NEEDED:
            return False
        
        # Check if all recent scores are below threshold
        low_scores = sum(1 for s in scores if s['percentage'] < self.DEMOTION_THRESHOLD)
        
        return low_scores >= self.CONSECUTIVE_NEEDED
    
    def get_current_difficulty(self):
        """Get current difficulty level"""
        return self.profile.get('difficulty', 'medium')
    
    def adjust_difficulty(self, topic=None):
        """Adjust difficulty based on performance"""
        current_difficulty = self.get_current_difficulty()
        current_index = self.DIFFICULTY_LEVELS.index(current_difficulty)
        
        new_difficulty = current_difficulty
        reason = None
        
        # Check for promotion
        if self.should_increase_difficulty(topic):
            if current_index < len(self.DIFFICULTY_LEVELS) - 1:
                new_difficulty = self.DIFFICULTY_LEVELS[current_index + 1]
                reason = f"Promoted! {self.CONSECUTIVE_NEEDED} consecutive scores ≥{self.PROMOTION_THRESHOLD}%"
        
        # Check for demotion
        elif self.should_decrease_difficulty(topic):
            if current_index > 0:
                new_difficulty = self.DIFFICULTY_LEVELS[current_index - 1]
                reason = f"Adjusted down. {self.CONSECUTIVE_NEEDED} consecutive scores <{self.DEMOTION_THRESHOLD}%"
        
        # Update profile if changed
        if new_difficulty != current_difficulty:
            self.profile['difficulty'] = new_difficulty
            self.save_profile()
            return {
                'changed': True,
                'old': current_difficulty,
                'new': new_difficulty,
                'reason': reason
            }
        
        return {'changed': False, 'current': current_difficulty}
    
    def get_recommended_difficulty(self, topic):
        """Get recommended difficulty for a specific topic"""
        scores = self.get_recent_quiz_scores(topic, n=5)
        
        if not scores:
            return self.get_current_difficulty()
        
        avg_score = sum(s['percentage'] for s in scores) / len(scores)
        
        # Recommend based on average performance
        if avg_score >= 85:
            return 'hard'
        elif avg_score >= 60:
            return 'medium'
        else:
            return 'easy'
    
    def get_difficulty_report(self):
        """Generate difficulty adjustment report"""
        current = self.get_current_difficulty()
        scores = self.get_recent_quiz_scores(n=10)
        
        if not scores:
            return {
                'current_difficulty': current,
                'recent_performance': 'No quiz data',
                'trend': 'unknown',
                'recommendation': 'Take more quizzes to establish baseline'
            }
        
        avg_score = sum(s['percentage'] for s in scores) / len(scores)
        trend = self.calculate_performance_trend(scores)
        
        # Generate recommendation
        if trend == 'improving' and avg_score >= 80:
            recommendation = f"You're doing great! Consider trying '{self.DIFFICULTY_LEVELS[min(self.DIFFICULTY_LEVELS.index(current) + 1, 2)]}' difficulty"
        elif trend == 'declining' and avg_score < 60:
            recommendation = f"Take a break and review fundamentals. Consider '{self.DIFFICULTY_LEVELS[max(self.DIFFICULTY_LEVELS.index(current) - 1, 0)]}' difficulty"
        else:
            recommendation = f"Current difficulty '{current}' is appropriate. Keep practicing!"
        
        return {
            'current_difficulty': current,
            'recent_performance': f"{avg_score:.1f}% average",
            'trend': trend,
            'recommendation': recommendation,
            'scores': scores
        }
    
    def get_topic_specific_difficulty(self):
        """Get difficulty recommendations per topic"""
        if self.kb.empty:
            return {}
        
        topics = self.kb['topic'].unique()
        topic_difficulties = {}
        
        for topic in topics:
            if pd.notna(topic):
                recommended = self.get_recommended_difficulty(topic)
                scores = self.get_recent_quiz_scores(topic, n=3)
                
                if scores:
                    avg = sum(s['percentage'] for s in scores) / len(scores)
                    topic_difficulties[topic] = {
                        'recommended': recommended,
                        'avg_score': avg,
                        'quiz_count': len(scores)
                    }
        
        return topic_difficulties
    
    def auto_adjust_after_quiz(self, topic, score_percentage):
        """Automatically adjust difficulty after a quiz"""
        adjustment = self.adjust_difficulty(topic)
        
        if adjustment['changed']:
            message = f"\n🎯 DIFFICULTY ADJUSTED!\n"
            message += f"   {adjustment['old'].upper()} → {adjustment['new'].upper()}\n"
            message += f"   Reason: {adjustment['reason']}\n"
            return message
        
        return None
