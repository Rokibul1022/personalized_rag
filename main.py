import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = Path('datasets')
PROFILES_DIR = Path('user_profiles')
PROFILES_DIR.mkdir(exist_ok=True)

class SimpleRAGSystem:
    def __init__(self):
        self.kb = self.load_knowledge_base()
        self.retriever = self.build_retriever()
        
    def load_knowledge_base(self):
        csv_files = list(DATA_DIR.glob('*.csv'))
        if not csv_files:
            return pd.DataFrame()
        
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    text_parts = []
                    for col in df.columns:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            text_parts.append(f"{col}: {row[col]}")
                    
                    if text_parts:
                        content = " | ".join(text_parts)
                        all_data.append({
                            'content': content,
                            'source': csv_file.name,
                            'topic': row.get('topic', 'General')
                        })
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return pd.DataFrame(all_data)
    
    def build_retriever(self):
        if self.kb.empty:
            return None
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
            stop_words='english'
        )
        
        normalized_content = [self.normalize_text(content) for content in self.kb['content']]
        self.doc_matrix = self.vectorizer.fit_transform(normalized_content)
        return True
    
    def normalize_text(self, text):
        import re
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def retrieve_documents(self, query, top_k=3):
        if not self.retriever:
            return []
        
        # Query expansion
        query_terms = self.normalize_text(query).split()
        expanded_terms = []
        
        term_expansions = {
            'quantum': ['quantum', 'atomic', 'electron', 'orbital', 'numbers'],
            'chemical': ['chemical', 'chemistry', 'bond', 'molecule', 'reaction'],
            'bonding': ['bonding', 'bond', 'ionic', 'covalent', 'molecular'],
            'photosynthesis': ['photosynthesis', 'plant', 'chlorophyll', 'glucose'],
            'calculus': ['calculus', 'derivative', 'integral', 'limit'],
            'newton': ['newton', 'force', 'motion', 'physics', 'law'],
            'vector': ['vector', 'magnitude', 'direction', 'component', 'dot', 'cross'],
            'matrix': ['matrix', 'determinant', 'inverse', 'eigenvalue', 'linear'],
            'algebra': ['algebra', 'equation', 'variable', 'polynomial', 'function'],
            'geometry': ['geometry', 'triangle', 'circle', 'angle', 'area', 'volume']
        }
        
        for term in query_terms:
            expanded_terms.append(term)
            if term in term_expansions:
                expanded_terms.extend(term_expansions[term])
        
        expanded_query = ' '.join(expanded_terms)
        query_vector = self.vectorizer.transform([expanded_query])
        
        similarity_scores = cosine_similarity(query_vector, self.doc_matrix).flatten()
        
        # Boost exact matches
        for i, doc_content in enumerate(self.kb['content']):
            doc_lower = doc_content.lower()
            for term in query_terms:
                if term in doc_lower:
                    similarity_scores[i] *= 1.5
        
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.05:
                content = self.kb.iloc[idx]['content']
                topic = self.extract_topic(content, query)
                results.append({
                    'content': content,
                    'topic': topic,
                    'score': float(similarity_scores[idx])
                })
        
        return results
    
    def extract_topic(self, content, query):
        content_lower = content.lower()
        query_lower = query.lower()
        
        if 'quantum' in query_lower and ('quantum' in content_lower or 'electron' in content_lower):
            return 'Quantum Numbers'
        elif 'bond' in query_lower and ('bond' in content_lower or 'ionic' in content_lower):
            return 'Chemical Bonding'
        elif 'photosynthesis' in query_lower and 'photosynthesis' in content_lower:
            return 'Photosynthesis'
        elif 'calculus' in query_lower and ('calculus' in content_lower or 'derivative' in content_lower):
            return 'Calculus'
        elif 'newton' in query_lower and ('newton' in content_lower or 'force' in content_lower):
            return 'Newton\'s Laws'
        
        return 'General Topic'
    
    def generate_response(self, query, profile, retrieved_docs):
        if not retrieved_docs:
            return f"Hi {profile['name']}! I couldn't find specific information about '{query}'. Try rephrasing your question."
        
        response_parts = [f"Hi {profile['name']}! 📚"]
        
        if 'visual' in profile.get('learning_style', '').lower():
            response_parts.append("Since you prefer visual learning, here's a clear explanation:")
        
        response_parts.append(f"\n🎯 **About {query.title()}:**")
        
        # Extract and format content more comprehensively
        all_content = []
        for doc in retrieved_docs:
            content = doc['content']
            if '|' in content:
                parts = content.split('|')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        # Skip metadata fields
                        if key not in ['no', 'example', 'source', 'file'] and value:
                            all_content.append((key, value))
        
        # Group similar content
        content_dict = {}
        for key, value in all_content:
            if key in content_dict:
                if value not in content_dict[key]:
                    content_dict[key] += f"; {value}"
            else:
                content_dict[key] = value
        
        # Add main content
        for key, value in list(content_dict.items())[:4]:  # Show top 4 pieces of info
            if len(value) > 10:  # Only show substantial content
                response_parts.append(f"• **{key.title()}**: {value}")
        
        # Add examples if available
        examples = [v for k, v in content_dict.items() if 'example' in k.lower()]
        if examples:
            response_parts.append(f"\n📝 **Examples**: {examples[0]}")
        
        response_parts.append("\n💡 **Study Tips:**")
        if any(topic.strip().lower() in query.lower() for topic in profile.get('weak_topics', '').split(',')):
            response_parts.append("• This is a challenging topic for you - break it into smaller parts")
            response_parts.append("• Practice with simple examples first")
        
        if profile.get('difficulty') == 'challenging':
            response_parts.append("• Try solving complex problems to master the concept")
        elif profile.get('difficulty') == 'easy':
            response_parts.append("• Focus on understanding the basic concepts first")
        
        # Add learning style specific tips
        learning_style = profile.get('learning_style', '').lower()
        if 'visual' in learning_style:
            response_parts.append("• Draw diagrams or charts to visualize the concept")
        elif 'hands' in learning_style:
            response_parts.append("• Try hands-on experiments or practical applications")
        
        return "\n".join(response_parts)
    
    def generate_quiz(self, topic, profile, num_questions=3):
        """Generate quiz questions for ANY topic dynamically"""
        # Generic question templates that work for any topic
        question_templates = [
            f"What is the main concept of {topic}?",
            f"Explain the key principles of {topic}.",
            f"How would you apply {topic} in real-world scenarios?",
            f"What are the important components or elements of {topic}?",
            f"Describe the relationship between different aspects of {topic}.",
            f"What are common misconceptions about {topic}?",
            f"How does {topic} relate to other concepts you've learned?",
            f"What are the practical applications of {topic}?",
            f"Explain {topic} in your own words.",
            f"What challenges might you face when learning {topic}?"
        ]
        
        # Select random questions based on num_questions
        selected = random.sample(question_templates, min(num_questions, len(question_templates)))
        return selected

class ChatInterface:
    def __init__(self):
        self.rag_system = SimpleRAGSystem()
        self.current_user = None
        self.conversation_context = {
            'current_topic': None,
            'follow_up_count': 0,
            'max_follow_ups': 10,
            'topic_content': None
        }
        
    def collect_profile(self):
        print("🎓 Welcome to Adaptive Learning Assistant!")
        print("Let's create your personalized learning profile:\n")
        
        profile = {}
        questions = [
            ('name', 'What is your name?'),
            ('age', 'How old are you?'),
            ('grade', 'What grade/class are you in?'),
            ('favorite_topics', 'What subjects do you enjoy? (comma separated)'),
            ('weak_topics', 'Which topics are challenging for you? (comma separated)'),
            ('learning_style', 'How do you prefer to learn? (visual/auditory/hands-on)'),
            ('difficulty', 'Do you prefer easy, medium, or challenging problems?'),
            ('goals', 'What are your learning goals?')
        ]
        
        for key, question in questions:
            answer = input(f"📝 {question}: ").strip()
            profile[key] = answer if answer else "Not specified"
        
        return profile
    
    def save_profile(self, profile):
        profile_file = PROFILES_DIR / f"{profile['name'].replace(' ', '_')}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f"✅ Profile saved for {profile['name']}")
    
    def load_profile(self, name):
        profile_file = PROFILES_DIR / f"{name.replace(' ', '_')}.json"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return json.load(f)
        return None
    
    def chat_loop(self):
        print("\n🤖 RAG Assistant is ready!")
        print("Commands: 'quiz' for practice questions, 'profile' to see your info, 'new topic' to reset, 'quit' to exit\n")
        
        while True:
            user_input = input(f"💬 {self.current_user['name']}: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Happy learning! See you next time!")
                break
            elif user_input.lower() == 'profile':
                print(f"\n👤 Your Profile:")
                for key, value in self.current_user.items():
                    print(f"  {key.title()}: {value}")
                print()
                continue
            elif user_input.lower() == 'new topic':
                self.conversation_context = {
                    'current_topic': None,
                    'follow_up_count': 0,
                    'max_follow_ups': 10,
                    'topic_content': None
                }
                print("✅ Context reset. Ask me about a new topic!\n")
                continue
            elif user_input.lower() == 'quiz':
                self.generate_quiz_session()
                continue
            elif not user_input:
                continue
            
            # Track follow-up context
            if self.conversation_context['current_topic']:
                self.conversation_context['follow_up_count'] += 1
                if self.conversation_context['follow_up_count'] <= self.conversation_context['max_follow_ups']:
                    print(f"\n💡 Follow-up {self.conversation_context['follow_up_count']}/{self.conversation_context['max_follow_ups']} on: {self.conversation_context['current_topic']}")
                else:
                    print("\n📚 Reached follow-up limit. Searching for new context...")
                    self.conversation_context['follow_up_count'] = 0
            else:
                print("\n🔍 Searching knowledge base...")
            
            # Retrieve documents
            retrieved_docs = self.rag_system.retrieve_documents(user_input)
            
            # Update context for new topics
            if retrieved_docs and self.conversation_context['follow_up_count'] == 0:
                self.conversation_context['current_topic'] = retrieved_docs[0].get('topic', user_input)
                self.conversation_context['topic_content'] = retrieved_docs[0].get('content', '')
            
            # Generate response
            response = self.rag_system.generate_response(user_input, self.current_user, retrieved_docs)
            
            print("\n" + "="*60)
            print(response)
            print("="*60 + "\n")
            
            # Offer quiz based on user's query
            quiz_offer = input("🎯 Would you like a quiz on this topic? (y/n): ").lower()
            if quiz_offer == 'y':
                topic = self.conversation_context['current_topic'] or user_input
                self.run_quiz(topic)
    
    def generate_quiz_session(self):
        print("\n📚 Quiz Generator")
        print("Enter any topic you want to practice (e.g., arrays, photosynthesis, calculus, etc.)")
        
        topic = input("\n🎯 Topic: ").strip()
        
        if not topic:
            print("❌ Please enter a valid topic!")
            return
        
        try:
            num_questions = int(input("📝 Number of questions (1-10): ").strip() or "3")
            num_questions = max(1, min(10, num_questions))  # Clamp between 1-10
        except ValueError:
            num_questions = 3
            print("Using default: 3 questions")
        
        self.run_quiz(topic, num_questions)
    
    def run_quiz(self, topic, num_questions=3):
        print(f"\n⏳ Generating {num_questions} questions about '{topic}'...")
        questions = self.rag_system.generate_quiz(topic, self.current_user, num_questions)
        score = 0
        
        print(f"\n🎯 Quiz: {topic.title()}")
        print("-" * 40)
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            answer = input("Your answer: ").strip()
            
            if answer:
                print("✅ Good effort!")
                score += 1
            else:
                print("❌ Try to answer next time!")
        
        print(f"\n📊 Quiz Complete! Score: {score}/{len(questions)}")
        
        if score == len(questions):
            print("🎉 Excellent work! You've mastered this topic!")
        elif score > len(questions) // 2:
            print("👍 Good job! Keep practicing to improve!")
        else:
            print("📖 Review the material and try again!")
    
    def start(self):
        print("🚀 Starting Adaptive Learning Assistant...")
        
        # Check for existing user
        name = input("Enter your name (or 'new' for new user): ").strip()
        
        if name.lower() == 'new':
            profile = self.collect_profile()
            self.save_profile(profile)
            self.current_user = profile
        else:
            profile = self.load_profile(name)
            if profile:
                print(f"✅ Welcome back, {profile['name']}!")
                self.current_user = profile
            else:
                print("User not found. Creating new profile...")
                profile = self.collect_profile()
                self.save_profile(profile)
                self.current_user = profile
        
        # Start chat
        self.chat_loop()

if __name__ == "__main__":
    print("🤖 Choose your RAG system:")
    print("1. Basic RAG (Local only)")
    print("2. AI-Powered RAG (with Gemini API)")
    
    choice = input("Select option (1 or 2): ").strip()
    
    try:
        if choice == '2':
            # Use Gemini-powered RAG
            from rag_api import ChatInterface as GeminiChatInterface
            chat = GeminiChatInterface()
        else:
            # Use basic RAG
            chat = ChatInterface()
        
        chat.start()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy learning!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your setup and try again.")