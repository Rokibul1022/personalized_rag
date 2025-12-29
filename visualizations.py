"""
Visualization Module for Learning Analytics
Creates charts and graphs for progress tracking
"""

import pandas as pd
from pathlib import Path

def create_progress_chart(user_kb_file, output_file='progress_chart.txt'):
    """Create ASCII progress chart"""
    if not Path(user_kb_file).exists():
        return
    
    kb = pd.read_csv(user_kb_file)
    
    # Quiz performance over time
    quizzes = kb[kb['type'] == 'quiz'].copy()
    
    if quizzes.empty:
        print("No quiz data available for visualization")
        return
    
    print("\n📊 QUIZ PERFORMANCE TREND")
    print("="*50)
    
    for idx, row in quizzes.iterrows():
        if pd.notna(row['quiz_score']) and '/' in str(row['quiz_score']):
            try:
                score_str = str(row['quiz_score'])
                score = float(score_str.split('/')[0].split('(')[0].strip())
                total = float(score_str.split('/')[1].split('(')[0].strip())
                percentage = int((score / total) * 100)
                
                # Create bar
                bar_length = percentage // 2
                bar = '█' * bar_length + '░' * (50 - bar_length)
                
                topic_short = row['topic'][:20] + '...' if len(str(row['topic'])) > 20 else row['topic']
                print(f"{topic_short:25} |{bar}| {percentage}%")
            except:
                pass
    
    print("="*50)

def create_topic_mastery_chart(user_kb_file):
    """Create topic mastery visualization"""
    if not Path(user_kb_file).exists():
        return
    
    kb = pd.read_csv(user_kb_file)
    
    # Get unique topics with levels
    topic_levels = {}
    for _, row in kb.iterrows():
        if pd.notna(row['topic']) and pd.notna(row['level']):
            topic = row['topic']
            level = row['level']
            topic_levels[topic] = level
    
    if not topic_levels:
        print("No mastery data available")
        return
    
    print("\n🎯 TOPIC MASTERY LEVELS")
    print("="*50)
    
    level_colors = {
        'novice': '🔴',
        'beginner': '🟡',
        'intermediate': '🟢',
        'advanced': '🔵'
    }
    
    for topic, level in sorted(topic_levels.items(), key=lambda x: x[1]):
        icon = level_colors.get(level, '⚪')
        topic_short = topic[:30] + '...' if len(topic) > 30 else topic
        print(f"{icon} {topic_short:35} {level.upper()}")
    
    print("="*50)
    print("Legend: 🔴 Novice | 🟡 Beginner | 🟢 Intermediate | 🔵 Advanced")

def create_activity_heatmap(user_kb_file):
    """Create activity heatmap (last 30 days)"""
    if not Path(user_kb_file).exists():
        return
    
    kb = pd.read_csv(user_kb_file)
    
    if kb.empty:
        return
    
    # Convert timestamps
    kb['date'] = pd.to_datetime(kb['timestamp']).dt.date
    
    # Count activities per day
    activity_counts = kb.groupby('date').size().to_dict()
    
    # Get last 30 days
    from datetime import datetime, timedelta
    today = datetime.now().date()
    dates = [(today - timedelta(days=i)) for i in range(29, -1, -1)]
    
    print("\n📅 ACTIVITY HEATMAP (Last 30 Days)")
    print("="*50)
    
    # Create heatmap
    week_data = []
    for i, date in enumerate(dates):
        count = activity_counts.get(date, 0)
        
        # Intensity levels
        if count == 0:
            symbol = '⬜'
        elif count <= 2:
            symbol = '🟩'
        elif count <= 5:
            symbol = '🟨'
        else:
            symbol = '🟥'
        
        week_data.append(symbol)
        
        # Print week by week
        if (i + 1) % 7 == 0 or i == len(dates) - 1:
            print(' '.join(week_data))
            week_data = []
    
    print("="*50)
    print("Legend: ⬜ No activity | 🟩 Light | 🟨 Moderate | 🟥 Heavy")

def generate_all_visualizations(user_kb_file):
    """Generate all visualizations"""
    create_topic_mastery_chart(user_kb_file)
    create_progress_chart(user_kb_file)
    create_activity_heatmap(user_kb_file)
