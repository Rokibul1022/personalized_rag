import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

class CollaborativeLearning:
    def __init__(self, user_profiles_dir="user_profiles"):
        self.profiles_dir = user_profiles_dir
        self.study_groups_file = os.path.join(user_profiles_dir, "study_groups.json")
        self.collaboration_log = os.path.join(user_profiles_dir, "collaboration_log.json")
        self._init_files()
    
    def _init_files(self):
        """Initialize collaboration files if they don't exist"""
        if not os.path.exists(self.study_groups_file):
            with open(self.study_groups_file, 'w') as f:
                json.dump({"groups": []}, f, indent=2)
        
        if not os.path.exists(self.collaboration_log):
            with open(self.collaboration_log, 'w') as f:
                json.dump({"collaborations": []}, f, indent=2)
    
    def get_all_users(self):
        """Get all registered users from profile directory"""
        users = []
        for file in os.listdir(self.profiles_dir):
            if file.endswith('.json') and file not in ['study_groups.json', 'collaboration_log.json']:
                username = file.replace('.json', '')
                profile_path = os.path.join(self.profiles_dir, file)
                kb_path = os.path.join(self.profiles_dir, f"{username}_knowledge_base.csv")
                
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                
                # Get user stats
                stats = self._get_user_stats(username, kb_path)
                
                users.append({
                    'username': username,
                    'profile': profile,
                    'stats': stats,
                    'online': True  # In real system, track actual online status
                })
        
        return users
    
    def _get_user_stats(self, username, kb_path):
        """Get user learning statistics"""
        if not os.path.exists(kb_path):
            return {'total_interactions': 0, 'topics_explored': 0, 'avg_quiz_score': 0}
        
        df = pd.read_csv(kb_path)
        quiz_scores = []
        
        for _, row in df.iterrows():
            if row['type'] == 'quiz' and pd.notna(row['quiz_score']):
                score_str = str(row['quiz_score'])
                if '/' in score_str:
                    correct, total = score_str.split('(')[0].strip().split('/')
                    quiz_scores.append((int(correct) / int(total)) * 100)
        
        topics = df[df['type'].isin(['query', 'assessment'])]['topic'].nunique() if 'topic' in df.columns else 0
        
        return {
            'total_interactions': len(df),
            'topics_explored': topics,
            'avg_quiz_score': round(sum(quiz_scores) / len(quiz_scores), 1) if quiz_scores else 0
        }
    
    def display_all_users(self):
        """Display all available users in a formatted way"""
        users = self.get_all_users()
        
        print("\n" + "="*80)
        print("🌐 COLLABORATIVE LEARNING - AVAILABLE USERS".center(80))
        print("="*80 + "\n")
        
        if not users:
            print("No users found in the system.")
            return []
        
        for i, user in enumerate(users, 1):
            profile = user['profile']
            stats = user['stats']
            status = "🟢 Online" if user['online'] else "⚫ Offline"
            
            name = profile.get('name', user.get('username', 'Unknown'))
            print(f"{i}. {status} {name.upper()}")
            print(f"   └─ Age: {profile.get('age', 'N/A')} | Grade: {profile.get('grade', 'N/A')}")
            print(f"   └─ Interests: {profile.get('favorite_topics', 'N/A')}")
            print(f"   └─ Weak Areas: {profile.get('weak_topics', 'N/A')}")
            print(f"   └─ Learning Style: {profile.get('learning_style', 'N/A')}")
            print(f"   └─ Stats: {stats['topics_explored']} topics | {stats['total_interactions']} interactions | {stats['avg_quiz_score']}% avg quiz")
            print()
        
        return users
    
    def find_study_partners(self, current_user, criteria='interests'):
        """Find compatible study partners based on criteria"""
        all_users = self.get_all_users()
        current_profile = None
        partners = []
        
        # Get current user profile
        for user in all_users:
            if user['username'].lower() == current_user.lower():
                current_profile = user
                break
        
        if not current_profile:
            return []
        
        # Find matches
        for user in all_users:
            if user['username'].lower() == current_user.lower():
                continue
            
            match_score = 0
            reasons = []
            
            if criteria == 'interests':
                # Match by favorite topics
                if current_profile['profile'].get('favorite_topics', '').lower() in user['profile'].get('favorite_topics', '').lower():
                    match_score += 3
                    reasons.append("shared interests")
                
                # Match by grade level
                if current_profile['profile'].get('grade', '') == user['profile'].get('grade', ''):
                    match_score += 2
                    reasons.append("same grade")
            
            elif criteria == 'complement':
                # Match strengths with weaknesses
                current_weak = current_profile['profile'].get('weak_topics', '').lower()
                partner_strong = user['profile'].get('favorite_topics', '').lower()
                
                if current_weak and partner_strong and current_weak in partner_strong:
                    match_score += 3
                    reasons.append("can help with your weak areas")
            
            if match_score > 0:
                partners.append({
                    'user': user,
                    'match_score': match_score,
                    'reasons': reasons
                })
        
        # Sort by match score
        partners.sort(key=lambda x: x['match_score'], reverse=True)
        return partners
    
    def create_study_group(self, creator, members, topic, description=""):
        """Create a new study group"""
        with open(self.study_groups_file, 'r') as f:
            data = json.load(f)
        
        group_id = len(data['groups']) + 1
        group = {
            'id': group_id,
            'creator': creator,
            'members': members,
            'topic': topic,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'messages': []
        }
        
        data['groups'].append(group)
        
        with open(self.study_groups_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return group_id
    
    def get_study_groups(self, username=None):
        """Get all study groups or groups for specific user"""
        with open(self.study_groups_file, 'r') as f:
            data = json.load(f)
        
        if username:
            return [g for g in data['groups'] if username in g['members'] or g['creator'] == username]
        
        return data['groups']
    
    def join_study_group(self, group_id, username):
        """Join an existing study group"""
        with open(self.study_groups_file, 'r') as f:
            data = json.load(f)
        
        for group in data['groups']:
            if group['id'] == group_id:
                if username not in group['members']:
                    group['members'].append(username)
                    
                    with open(self.study_groups_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    return True
        
        return False
    
    def share_knowledge(self, from_user, to_user, topic, content):
        """Share knowledge between users"""
        with open(self.collaboration_log, 'r') as f:
            data = json.load(f)
        
        collaboration = {
            'from': from_user,
            'to': to_user,
            'topic': topic,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        data['collaborations'].append(collaboration)
        
        with open(self.collaboration_log, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_shared_knowledge(self, username):
        """Get knowledge shared with a user"""
        with open(self.collaboration_log, 'r') as f:
            data = json.load(f)
        
        return [c for c in data['collaborations'] if c['to'] == username]
    
    def display_study_groups(self, username=None):
        """Display study groups"""
        groups = self.get_study_groups(username)
        
        if not groups:
            print("\n📚 No study groups found.")
            return
        
        print("\n" + "="*80)
        print("📚 STUDY GROUPS".center(80))
        print("="*80 + "\n")
        
        for group in groups:
            print(f"Group #{group['id']}: {group['topic']}")
            print(f"   Creator: {group['creator']}")
            print(f"   Members: {', '.join(group['members'])}")
            print(f"   Description: {group['description']}")
            print(f"   Created: {group['created_at'][:10]}")
            print()
    
    def recommend_collaborators(self, current_user):
        """Recommend best study partners"""
        print("\n" + "="*80)
        print("🤝 RECOMMENDED STUDY PARTNERS".center(80))
        print("="*80 + "\n")
        
        # Find by shared interests
        interest_matches = self.find_study_partners(current_user, 'interests')
        
        # Find complementary partners
        complement_matches = self.find_study_partners(current_user, 'complement')
        
        if interest_matches:
            print("📌 Based on Shared Interests:\n")
            for match in interest_matches[:3]:
                user = match['user']
                profile = user['profile']
                name = profile.get('name', user.get('username', 'Unknown'))
                print(f"   • {name} - {', '.join(match['reasons'])}")
                print(f"     Topics: {profile.get('favorite_topics', 'N/A')}")
                print()
        
        if complement_matches:
            print("📌 Can Help With Your Weak Areas:\n")
            for match in complement_matches[:3]:
                user = match['user']
                profile = user['profile']
                name = profile.get('name', user.get('username', 'Unknown'))
                print(f"   • {name} - {', '.join(match['reasons'])}")
                print(f"     Strong in: {profile.get('favorite_topics', 'N/A')}")
                print()
