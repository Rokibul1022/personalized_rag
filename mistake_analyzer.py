"""
Mistake Pattern Analysis System
Analyzes wrong answers, identifies patterns, and generates targeted practice
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter
import re

class MistakeAnalyzer:
    def __init__(self, user_kb_file, user_profile_file):
        self.user_kb_file = user_kb_file
        self.user_profile_file = user_profile_file
        self.kb = self.load_kb()
        self.profile = self.load_profile()
        
        # Store wrong answers
        self.wrong_answers_file = Path(str(user_kb_file).replace('.csv', '_mistakes.json'))
        self.mistakes = self.load_mistakes()
    
    def load_kb(self):
        """Load user's knowledge base"""
        if Path(self.user_kb_file).exists():
            return pd.read_csv(self.user_kb_file)
        return pd.DataFrame()
    
    def load_profile(self):
        """Load user profile"""
        if Path(self.user_profile_file).exists():
            with open(self.user_profile_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_mistakes(self):
        """Load stored mistakes"""
        if self.wrong_answers_file.exists():
            with open(self.wrong_answers_file, 'r') as f:
                return json.load(f)
        return {'mistakes': [], 'patterns': {}}
    
    def save_mistakes(self):
        """Save mistakes to file"""
        with open(self.wrong_answers_file, 'w') as f:
            json.dump(self.mistakes, f, indent=2)
    
    def record_mistake(self, question, user_answer, correct_answer, topic, difficulty):
        """Record a wrong answer"""
        mistake = {
            'timestamp': str(pd.Timestamp.now()),
            'question': question,
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'topic': topic,
            'difficulty': difficulty
        }
        
        self.mistakes['mistakes'].append(mistake)
        self.save_mistakes()
    
    def analyze_patterns(self):
        """Analyze patterns in mistakes"""
        if not self.mistakes['mistakes']:
            return {}
        
        patterns = {
            'by_topic': defaultdict(int),
            'by_difficulty': defaultdict(int),
            'common_errors': [],
            'weak_concepts': [],
            'mistake_frequency': {}
        }
        
        # Analyze by topic
        for mistake in self.mistakes['mistakes']:
            topic = mistake.get('topic', 'Unknown')
            difficulty = mistake.get('difficulty', 'medium')
            
            patterns['by_topic'][topic] += 1
            patterns['by_difficulty'][difficulty] += 1
        
        # Find most problematic topics
        if patterns['by_topic']:
            sorted_topics = sorted(patterns['by_topic'].items(), key=lambda x: x[1], reverse=True)
            patterns['weak_concepts'] = [{'topic': t, 'mistakes': c} for t, c in sorted_topics[:5]]
        
        # Analyze answer patterns
        answer_types = self.classify_answer_types()
        patterns['common_errors'] = answer_types
        
        return patterns
    
    def classify_answer_types(self):
        """Classify types of wrong answers"""
        error_types = {
            'blank': 0,           # No answer
            'partial': 0,         # Partially correct
            'misconception': 0,   # Wrong concept
            'calculation': 0      # Calculation error
        }
        
        for mistake in self.mistakes['mistakes']:
            user_ans = str(mistake.get('user_answer', '')).lower().strip()
            
            if not user_ans or user_ans in ['idk', "i don't know", 'dont know', '']:
                error_types['blank'] += 1
            elif len(user_ans) < 5:
                error_types['partial'] += 1
            else:
                error_types['misconception'] += 1
        
        return [{'type': k, 'count': v} for k, v in error_types.items() if v > 0]
    
    def get_weak_topics(self):
        """Get topics with most mistakes"""
        patterns = self.analyze_patterns()
        return patterns.get('weak_concepts', [])
    
    def get_mistake_summary(self):
        """Get summary of mistakes"""
        total_mistakes = len(self.mistakes['mistakes'])
        
        if total_mistakes == 0:
            return {
                'total': 0,
                'message': 'No mistakes recorded yet. Take some quizzes!'
            }
        
        patterns = self.analyze_patterns()
        
        # Get recent mistakes
        recent = self.mistakes['mistakes'][-5:]
        
        return {
            'total': total_mistakes,
            'by_topic': dict(patterns['by_topic']),
            'by_difficulty': dict(patterns['by_difficulty']),
            'weak_concepts': patterns['weak_concepts'],
            'common_errors': patterns['common_errors'],
            'recent_mistakes': recent
        }
    
    def generate_targeted_practice(self, topic, num_questions=5):
        """Generate practice questions for weak areas"""
        # Get mistakes for this topic
        topic_mistakes = [m for m in self.mistakes['mistakes'] if m.get('topic') == topic]
        
        if not topic_mistakes:
            return None
        
        # Analyze what types of questions were missed
        missed_concepts = []
        for mistake in topic_mistakes[-10:]:  # Last 10 mistakes
            question = mistake.get('question', '')
            missed_concepts.append(question)
        
        return {
            'topic': topic,
            'mistake_count': len(topic_mistakes),
            'focus_areas': missed_concepts[:3],
            'recommended_questions': num_questions
        }
    
    def get_improvement_suggestions(self):
        """Generate personalized improvement suggestions"""
        suggestions = []
        patterns = self.analyze_patterns()
        
        # Based on weak topics
        weak_topics = patterns.get('weak_concepts', [])
        if weak_topics:
            top_weak = weak_topics[0]
            suggestions.append({
                'type': 'focus_topic',
                'message': f"Focus on '{top_weak['topic']}' - you have {top_weak['mistakes']} mistakes here",
                'action': f"Take a practice quiz on {top_weak['topic']}"
            })
        
        # Based on error types
        error_types = patterns.get('common_errors', [])
        for error in error_types:
            if error['type'] == 'blank' and error['count'] > 3:
                suggestions.append({
                    'type': 'answer_strategy',
                    'message': f"You left {error['count']} questions blank",
                    'action': "Try to answer even if unsure - eliminate wrong options first"
                })
            elif error['type'] == 'misconception' and error['count'] > 5:
                suggestions.append({
                    'type': 'concept_review',
                    'message': f"You have {error['count']} conceptual errors",
                    'action': "Review fundamental concepts before taking more quizzes"
                })
        
        # Based on difficulty
        if patterns['by_difficulty'].get('hard', 0) > patterns['by_difficulty'].get('easy', 0):
            suggestions.append({
                'type': 'difficulty_adjustment',
                'message': "Most mistakes are on hard questions",
                'action': "Practice medium difficulty first to build confidence"
            })
        
        return suggestions
    
    def get_misconception_report(self):
        """Identify common misconceptions"""
        misconceptions = []
        
        # Group similar mistakes
        topic_mistakes = defaultdict(list)
        for mistake in self.mistakes['mistakes']:
            topic = mistake.get('topic', 'Unknown')
            topic_mistakes[topic].append(mistake)
        
        # Analyze each topic
        for topic, mistakes in topic_mistakes.items():
            if len(mistakes) >= 2:
                misconceptions.append({
                    'topic': topic,
                    'frequency': len(mistakes),
                    'examples': [m['question'][:50] + '...' for m in mistakes[:2]]
                })
        
        return sorted(misconceptions, key=lambda x: x['frequency'], reverse=True)
    
    def calculate_improvement_rate(self):
        """Calculate if user is improving over time"""
        if len(self.mistakes['mistakes']) < 10:
            return None
        
        # Split into older and recent
        mid = len(self.mistakes['mistakes']) // 2
        older_mistakes = self.mistakes['mistakes'][:mid]
        recent_mistakes = self.mistakes['mistakes'][mid:]
        
        # Count mistakes per quiz (approximate)
        older_rate = len(older_mistakes) / max(mid / 5, 1)  # Assume 5 questions per quiz
        recent_rate = len(recent_mistakes) / max((len(self.mistakes['mistakes']) - mid) / 5, 1)
        
        improvement = ((older_rate - recent_rate) / older_rate * 100) if older_rate > 0 else 0
        
        return {
            'older_mistake_rate': older_rate,
            'recent_mistake_rate': recent_rate,
            'improvement_percentage': improvement,
            'trend': 'improving' if improvement > 10 else 'stable' if improvement > -10 else 'declining'
        }
    
    def display_mistake_report(self):
        """Display comprehensive mistake analysis"""
        summary = self.get_mistake_summary()
        
        print("\n" + "="*60)
        print("🔍 MISTAKE PATTERN ANALYSIS")
        print("="*60)
        
        if summary['total'] == 0:
            print("\n✅ No mistakes recorded yet!")
            print("   Take some quizzes to start tracking your learning patterns.")
            print("="*60 + "\n")
            return
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Mistakes: {summary['total']}")
        
        # By topic
        if summary['by_topic']:
            print(f"\n📚 MISTAKES BY TOPIC:")
            for topic, count in sorted(summary['by_topic'].items(), key=lambda x: x[1], reverse=True)[:5]:
                topic_short = topic[:40] + '...' if len(topic) > 40 else topic
                bar = '█' * min(count, 20)
                print(f"  {topic_short:45} {bar} {count}")
        
        # Weak concepts
        if summary['weak_concepts']:
            print(f"\n⚠️  WEAK AREAS (Need Practice):")
            for concept in summary['weak_concepts'][:3]:
                print(f"  • {concept['topic']}: {concept['mistakes']} mistakes")
        
        # Error types
        if summary['common_errors']:
            print(f"\n🎯 ERROR PATTERNS:")
            for error in summary['common_errors']:
                error_names = {
                    'blank': 'Left blank / "I don\'t know"',
                    'partial': 'Incomplete answers',
                    'misconception': 'Conceptual errors',
                    'calculation': 'Calculation mistakes'
                }
                print(f"  • {error_names.get(error['type'], error['type'])}: {error['count']} times")
        
        # Improvement suggestions
        suggestions = self.get_improvement_suggestions()
        if suggestions:
            print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
            for i, sug in enumerate(suggestions[:3], 1):
                print(f"  {i}. {sug['message']}")
                print(f"     → {sug['action']}")
        
        # Improvement rate
        improvement = self.calculate_improvement_rate()
        if improvement:
            print(f"\n📈 IMPROVEMENT TREND:")
            trend_emoji = '📈' if improvement['trend'] == 'improving' else '📉' if improvement['trend'] == 'declining' else '➡️'
            print(f"  {trend_emoji} {improvement['trend'].upper()}")
            if improvement['improvement_percentage'] > 0:
                print(f"  You're making {improvement['improvement_percentage']:.1f}% fewer mistakes!")
            elif improvement['improvement_percentage'] < 0:
                print(f"  Mistakes increased by {abs(improvement['improvement_percentage']):.1f}%")
        
        # Recent mistakes
        if summary['recent_mistakes']:
            print(f"\n🕐 RECENT MISTAKES:")
            for mistake in summary['recent_mistakes'][-3:]:
                q_short = mistake['question'][:50] + '...' if len(mistake['question']) > 50 else mistake['question']
                print(f"  • {mistake['topic']}: {q_short}")
        
        print("\n" + "="*60 + "\n")
