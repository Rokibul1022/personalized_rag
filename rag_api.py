import json
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyCT5iExKJOaHUDUR8rfHNtxvkIc1tFiD2Y"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent"

DATA_DIR = Path('datasets')
PROFILES_DIR = Path('user_profiles')
PROFILES_DIR.mkdir(exist_ok=True)

class GeminiRAGSystem:
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
        
        query_normalized = self.normalize_text(query)
        query_vector = self.vectorizer.transform([query_normalized])
        similarity_scores = cosine_similarity(query_vector, self.doc_matrix).flatten()
        
        # Boost exact matches
        query_terms = query_normalized.split()
        for i, doc_content in enumerate(self.kb['content']):
            doc_lower = doc_content.lower()
            for term in query_terms:
                if term in doc_lower:
                    similarity_scores[i] *= 1.5
        
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.05:
                results.append({
                    'content': self.kb.iloc[idx]['content'],
                    'topic': self.kb.iloc[idx]['topic'],
                    'score': float(similarity_scores[idx])
                })
        
        return results
    
    def call_gemini_api(self, prompt):
        """Call Gemini API with the given prompt"""
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Sorry, I couldn't generate a response."
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return "Sorry, there was an error connecting to the AI service."
                
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "Sorry, I'm having trouble connecting to the AI service right now."
    
    def generate_response(self, query, profile, retrieved_docs):
        """Generate response using Gemini API with retrieved context"""
        
        # Build context from retrieved documents
        context = ""
        if retrieved_docs:
            context = "Relevant information from knowledge base:\n"
            for i, doc in enumerate(retrieved_docs[:2], 1):
                context += f"{i}. {doc['content']}\n"
        
        # Build personalized prompt
        prompt = f"""You are a personalized learning assistant. Here's the student's profile:
- Name: {profile.get('name', 'Student')}
- Age: {profile.get('age', 'Not specified')}
- Grade: {profile.get('grade', 'Not specified')}
- Learning Style: {profile.get('learning_style', 'Not specified')}
- Favorite Topics: {profile.get('favorite_topics', 'Not specified')}
- Weak Topics: {profile.get('weak_topics', 'Not specified')}
- Difficulty Preference: {profile.get('difficulty', 'medium')}
- Goals: {profile.get('goals', 'Not specified')}

{context}

Student Question: {query}

Please provide a personalized educational response that:
1. Addresses the student by name
2. Explains the concept clearly based on their learning style
3. Uses appropriate difficulty level
4. Includes practical examples
5. Provides study tips specific to their profile
6. If this relates to their weak topics, offer extra encouragement and simpler explanations
7. Keep the response educational and engaging

Format your response with emojis and clear sections like:
🎯 **Explanation:**
💡 **Examples:**
📚 **Study Tips:**
🎯 **Practice Suggestions:**"""

        return self.call_gemini_api(prompt)
    
    def generate_quiz(self, topic, profile):
        """Generate quiz questions using Gemini API"""
        prompt = f"""Create 3 educational quiz questions about "{topic}" for a student with this profile:
- Name: {profile.get('name', 'Student')}
- Grade: {profile.get('grade', 'Not specified')}
- Difficulty Preference: {profile.get('difficulty', 'medium')}

Make the questions appropriate for their level. Format as:
Q1: [question]
Q2: [question]  
Q3: [question]

Keep questions clear and educational."""

        response = self.call_gemini_api(prompt)
        
        # Parse questions from response
        questions = []
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('Q'):
                question = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                questions.append(question)
        
        return questions[:3] if questions else [
            f"What is the main concept of {topic}?",
            f"How would you apply {topic} in real life?",
            f"What are the key components of {topic}?"
        ]

class ChatInterface:
    def __init__(self):
        self.rag_system = GeminiRAGSystem()
        self.current_user = None
        
    def collect_profile(self):
        print("🎓 Welcome to AI-Powered Learning Assistant!")
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
        print("\n🤖 AI Learning Assistant is ready!")
        print("Commands: 'quiz' for practice questions, 'profile' to see your info, 'quit' to exit\n")
        
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
            elif user_input.lower() == 'quiz':
                self.generate_quiz_session()
                continue
            elif not user_input:
                continue
            
            print("\n🔍 Searching knowledge base and generating AI response...")
            
            # Retrieve documents
            retrieved_docs = self.rag_system.retrieve_documents(user_input)
            
            # Generate AI response
            response = self.rag_system.generate_response(user_input, self.current_user, retrieved_docs)
            
            print("\n" + "="*60)
            print(response)
            print("="*60 + "\n")
            
            # Offer quiz
            if retrieved_docs:
                quiz_offer = input("🎯 Would you like a quiz on this topic? (y/n): ").lower()
                if quiz_offer == 'y':
                    topic = retrieved_docs[0]['topic']
                    self.run_quiz(topic)
    
    def generate_quiz_session(self):
        topics = ['mathematics', 'physics', 'chemistry', 'biology', 'computer science']
        print("\n📚 Available quiz topics:")
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic.title()}")
        
        try:
            choice = int(input("\nSelect topic (1-5): ")) - 1
            if 0 <= choice < len(topics):
                self.run_quiz(topics[choice])
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")
    
    def run_quiz(self, topic):
        questions = self.rag_system.generate_quiz(topic, self.current_user)
        score = 0
        
        print(f"\n🎯 AI-Generated Quiz: {topic.title()}")
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
            print("🎉 Perfect! You're mastering this topic!")
        elif score > len(questions) // 2:
            print("👍 Great job! Keep practicing!")
        else:
            print("📖 Review the material and try again!")
    
    def start(self):
        print("🚀 Starting AI-Powered Learning Assistant...")
        
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
    try:
        chat = ChatInterface()
        chat.start()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy learning!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your setup and try again.")