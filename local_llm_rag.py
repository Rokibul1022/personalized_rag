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
MODELS_DIR = Path('local_models')
PROFILES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

class LocalLLMRAGSystem:
    def __init__(self, model_path=None, user_name=None):
        self.user_name = user_name
        self.user_kb_file = PROFILES_DIR / f"{user_name}_knowledge_base.csv" if user_name else None
        
        self.kb = self.load_knowledge_base()
        self.user_kb = self.load_user_knowledge_base()
        self.retriever = self.build_retriever()
        self.llm = self.load_local_llm(model_path)
        
        # Intelligent RAG components
        self.memory = self.load_memory()
        self.decision_patterns = self.load_decision_patterns()
        self.response_templates = self.load_response_templates()
        self.learning_weights = self.initialize_learning_weights()
        
    def load_local_llm(self, model_path):
        """Load local LLM model using Ollama or Transformers"""
        try:
            # Option 1: Using Ollama (recommended for local models)
            import requests
            
            # Test if Ollama is running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    print(" Ollama detected - using local LLM")
                    return "ollama"
            except:
                pass
            
            # Option 2: Using Transformers (for downloaded models)
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                if model_path and Path(model_path).exists():
                    print(f"📦 Loading model from {model_path}")
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    return {"tokenizer": tokenizer, "model": model, "type": "transformers"}
                else:
                    # Use a small model as fallback
                    print("📦 Loading default small model...")
                    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
                    return {"tokenizer": tokenizer, "model": model, "type": "transformers"}
                    
            except ImportError:
                print(" Transformers not installed. Install with: pip install transformers torch")
                return None
                
        except Exception as e:
            print(f" Error loading LLM: {e}")
            return None
    
    def call_ollama(self, prompt, model="deepseek-r1:1.5b"):
        """Call Ollama API for local LLM inference with DeepSeek-R1"""
        import requests
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 512
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return "Error: Could not generate response"
                
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
    def call_transformers(self, prompt, max_length=512):
        """Call Transformers model for local inference"""
        try:
            tokenizer = self.llm["tokenizer"]
            model = self.llm["model"]
            
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from response
            response = response[len(prompt):].strip()
            
            return response if response else "I need more context to provide a helpful response."
            
        except Exception as e:
            return f"Error with local model: {e}"
    
    def load_user_knowledge_base(self):
        """Load user's personal knowledge base"""
        if not self.user_kb_file or not self.user_kb_file.exists():
            # Create new user KB with columns
            df = pd.DataFrame(columns=['timestamp', 'query', 'topic', 'response', 'quiz_score', 'level', 'type'])
            if self.user_kb_file:
                df.to_csv(self.user_kb_file, index=False)
                print(f"💾 Created knowledge base: {self.user_kb_file.name}")
            return df
        
        return pd.read_csv(self.user_kb_file)
    
    def save_to_user_kb(self, query, topic, response='', quiz_score=None, level=None, entry_type='query'):
        """Save entry to user's knowledge base"""
        if not self.user_kb_file:
            return
        
        new_entry = pd.DataFrame([{
            'timestamp': pd.Timestamp.now(),
            'query': query,
            'topic': topic,
            'response': response[:500] if response else '',
            'quiz_score': quiz_score,
            'level': level,
            'type': entry_type
        }])
        
        self.user_kb = pd.concat([self.user_kb, new_entry], ignore_index=True)
        self.user_kb.to_csv(self.user_kb_file, index=False)
    
    def is_topic_in_user_kb(self, topic):
        """Check if topic exists in user's knowledge base"""
        if self.user_kb.empty:
            return False
        
        topic_lower = topic.lower()
        topic_terms = set(self.normalize_text(topic).split())
        
        for existing_topic in self.user_kb['topic'].dropna():
            existing_lower = str(existing_topic).lower()
            existing_terms = set(self.normalize_text(existing_topic).split())
            
            # Check for word containment or overlap
            for term in topic_terms:
                if len(term) > 3:  # Only meaningful words
                    for existing_term in existing_terms:
                        if term in existing_term or existing_term in term:
                            return True
            
            # Check direct containment
            if topic_lower in existing_lower or existing_lower in topic_lower:
                return True
        
        return False
    
    def get_user_topic_level(self, topic):
        """Get user's level for a specific topic"""
        if self.user_kb.empty:
            return None
        
        topic_entries = self.user_kb[self.user_kb['topic'].str.contains(topic, case=False, na=False)]
        if not topic_entries.empty:
            return topic_entries.iloc[-1]['level']
        return None
    
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
        # Combine global KB and user KB
        combined_kb = self.kb.copy()
        
        if not self.user_kb.empty:
            # Add user KB entries to combined KB
            for _, row in self.user_kb.iterrows():
                if pd.notna(row['query']) and pd.notna(row['topic']):
                    content = f"topic: {row['topic']} | query: {row['query']} | response: {row['response']}"
                    combined_kb = pd.concat([combined_kb, pd.DataFrame([{
                        'content': content,
                        'source': 'user_kb',
                        'topic': row['topic']
                    }])], ignore_index=True)
        
        if combined_kb.empty:
            return None
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
        
        normalized_content = [self.normalize_text(content) for content in combined_kb['content']]
        self.doc_matrix = self.vectorizer.fit_transform(normalized_content)
        self.kb = combined_kb
        return True
    
    def normalize_text(self, text):
        import re
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def retrieve_documents(self, query, top_k=5):
        if not self.retriever:
            return []
        
        # Expand query with synonyms and related terms
        expanded_query = self.expand_query(query)
        query_normalized = self.normalize_text(expanded_query)
        query_vector = self.vectorizer.transform([query_normalized])
        similarity_scores = cosine_similarity(query_vector, self.doc_matrix).flatten()
        
        # Advanced scoring with keyword matching
        query_terms = query_normalized.split()
        for i, doc_content in enumerate(self.kb['content']):
            doc_lower = doc_content.lower()
            
            # Exact phrase match - highest boost
            if query_normalized in doc_lower:
                similarity_scores[i] *= 3.0
            
            # Multi-word match boost
            matched_terms = sum(1 for term in query_terms if len(term) > 2 and term in doc_lower)
            if matched_terms > 0:
                similarity_scores[i] *= (1 + matched_terms * 0.3)
        
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0.01:
                results.append({
                    'content': self.kb.iloc[idx]['content'],
                    'topic': self.kb.iloc[idx]['topic'],
                    'score': float(similarity_scores[idx])
                })
        
        return results
    
    def expand_query(self, query):
        """Expand query with related terms for better retrieval"""
        query_lower = query.lower()
        expansions = {
            'data structure': 'data structure algorithm array list tree graph stack queue',
            'algorithm': 'algorithm sorting searching complexity time space',
            'data': 'data information storage structure database',
            'real life': 'real life application practical use case example',
            'software': 'software application program system development',
        }
        
        for key, expansion in expansions.items():
            if key in query_lower:
                query += ' ' + expansion
        
        return query
    
    def intelligent_retrieve(self, query, profile):
        base_results = self.retrieve_documents(query, top_k=5)
        
        # Check if retrieval quality is poor
        if not base_results:
            print(" No documents found - using LLM")
            return None
        
        # Check semantic relevance - do topics actually match the query?
        is_relevant = self.check_semantic_relevance(query, base_results)
        if not is_relevant:
            print(f" Topics don't match query (found: {base_results[0]['topic']}) - using LLM")
            return None
        
        # Check score threshold
        if base_results[0]['score'] < 0.5:
            print(f" Low relevance score ({base_results[0]['score']:.3f}) - using LLM")
            return None
        
        user_id = profile.get('name', 'unknown')
        
        if user_id in self.memory['user_preferences']:
            user_prefs = self.memory['user_preferences'][user_id]
            for result in base_results:
                topic = result['topic'].lower()
                if topic in user_prefs.get('successful_topics', []):
                    result['score'] *= 1.3
                elif topic in user_prefs.get('difficult_topics', []):
                    result['score'] *= 0.8
        
        return sorted(base_results, key=lambda x: x['score'], reverse=True)[:3]
    
    def check_semantic_relevance(self, query, results):
        """Check if retrieved topics are actually relevant to the query"""
        query_lower = query.lower()
        query_keywords = set(self.normalize_text(query).split())
        
        # Get top result topic
        top_topic = results[0]['topic'].lower()
        top_content = results[0]['content'].lower()
        
        # Check if any significant query words appear in topic or content
        significant_words = [w for w in query_keywords if len(w) > 3]
        
        if not significant_words:
            return True  # Short query, trust the score
        
        # Check if at least one significant word matches
        matches = 0
        for word in significant_words:
            if word in top_topic or word in top_content:
                matches += 1
        
        # Need at least 30% of significant words to match
        relevance_ratio = matches / len(significant_words)
        return relevance_ratio >= 0.3
    
    def intelligent_response_generation(self, query, profile, retrieved_docs):
        user_id = profile.get('name', 'Student')
        learning_style = profile.get('learning_style', 'general').lower()
        difficulty = profile.get('difficulty', 'medium')
        
        # This should not be called if retrieved_docs is None
        # The check is done in intelligent_retrieve
        
        processed_content = self.process_content_intelligently(retrieved_docs, difficulty)
        
        response = f"Hi {user_id}! \n\n **About {query.title()}:**\n{processed_content['main']}"
        
        study_tips = self.generate_personalized_tips(query, profile)
        response += f"\n\n **Personalized Tips:**\n{study_tips}"
        
        return response
    
    def process_content_intelligently(self, retrieved_docs, difficulty):
        all_content = []
        for doc in retrieved_docs[:3]:
            content = doc['content']
            if '|' in content:
                parts = content.split('|')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        if key.strip().lower() not in ['no', 'example', 'source']:
                            all_content.append(f"• **{key.strip()}**: {value.strip()}")
        
        main_content = '\n'.join(all_content[:4])
        return {'main': main_content}
    
    def generate_personalized_tips(self, query, profile):
        user_id = profile.get('name', 'unknown')
        tips = []
        
        if user_id in self.memory['user_preferences']:
            weak_topics = profile.get('weak_topics', '').lower()
            if any(topic in query.lower() for topic in weak_topics.split(',')):
                tips.append("• This topic needs extra attention - break it into smaller parts")
        
        learning_style = profile.get('learning_style', '').lower()
        if 'visual' in learning_style:
            tips.append("• Draw diagrams and use colors to highlight key concepts")
        elif 'hands' in learning_style:
            tips.append("• Find practical applications and experiments")
        
        return '\n'.join(tips) if tips else "• Practice regularly and ask questions when stuck"
    
    def generate_llm_response(self, query, profile):
        """Use LLM directly when retrieval fails and add to knowledge base"""
        user_id = profile.get('name', 'Student')
        learning_style = profile.get('learning_style', 'general')
        difficulty = profile.get('difficulty', 'medium')
        
        if not self.llm:
            return (f"Hi {user_id}! I don't have specific information about '{query}' in my knowledge base. Try asking about topics in mathematics, physics, chemistry, or computer science.", False)
        
        print(f"🤖 Generating answer using DeepSeek-R1...")
        
        prompt = f"""You are a personalized educational assistant helping {user_id}.

Student Profile:
- Learning Style: {learning_style}
- Difficulty Level: {difficulty}
- Weak Topics: {profile.get('weak_topics', 'None')}

Student Question: {query}

Provide a clear, educational response that:
1. Directly answers the question
2. Uses examples and analogies
3. Matches the {difficulty} difficulty level
4. Adapts to {learning_style} learning style

Keep response under 300 words."""
        
        if self.llm == "ollama":
            llm_response = self.call_ollama(prompt)
        else:
            llm_response = self.call_transformers(prompt)
        
        # Add LLM response to knowledge base
        self.add_to_knowledge_base(query, llm_response)
        
        # Format the response
        response = f"Hi {user_id}! \n\n **AI Response:**\n{llm_response}\n\n **Personalized Tips:**\n"
        
        # Add personalized tips
        if 'hands' in learning_style.lower():
            response += "• Try implementing this concept with code examples\n"
        elif 'visual' in learning_style.lower():
            response += "• Draw diagrams to visualize this concept\n"
        else:
            response += "• Practice with examples to reinforce understanding\n"
        
        return response, True  # Return flag indicating new topic
    
    def generate_fallback_response(self, query, profile):
        user_id = profile.get('name', 'Student')
        return f"Hi {user_id}! I don't have specific information about '{query}'. Try rephrasing or asking about related topics."
    
    def generate_response(self, query, profile, retrieved_docs):
        smart_docs = self.intelligent_retrieve(query, profile)
        
        # If retrieval failed or poor quality, use LLM directly
        if smart_docs is None:
            return self.generate_llm_response(query, profile)
        
        print(f" Found relevant content: '{smart_docs[0]['topic']}' (score: {smart_docs[0]['score']:.3f})")
        intelligent_response = self.intelligent_response_generation(query, profile, smart_docs)
        
        return intelligent_response, False
    
    def generate_quiz(self, topic, difficulty='medium', num_questions=5):
        """Generate quiz questions using DeepSeek-R1"""
        if not self.llm:
            return [{'question': f'What is {topic}?', 'options': ['A) Option 1', 'B) Option 2', 'C) Option 3', 'D) Option 4'], 'correct': 'A'}]
        
        # Simpler prompt that works better
        prompt = f"""Generate {num_questions} multiple choice quiz questions about "{topic}" at {difficulty} difficulty level.

Format each question EXACTLY like this:

Q1: [Your question here]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]
Correct: A

Q2: [Next question]
A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]
Correct: B

Generate all {num_questions} questions now:"""

        if self.llm == "ollama":
            response = self.call_ollama(prompt)
            print(f"\n🔍 DEBUG - LLM Response length: {len(response)} chars")
        else:
            response = "Q1: What is the main concept?\nA) Concept A\nB) Concept B\nC) Concept C\nD) Concept D\nCorrect: A"
        
        questions = self.parse_quiz_response(response)
        
        if not questions:
            print(" Failed to parse LLM response, generating fallback questions...")
            # Generate fallback questions
            return self.generate_fallback_quiz(topic, num_questions)
        
        return questions
    
    def generate_fallback_quiz(self, topic, num_questions):
        """Generate simple fallback quiz questions with randomized answers"""
        questions = []
        question_templates = [
            f'What is a key concept in {topic}?',
            f'Which technique is commonly used in {topic}?',
            f'What is an important principle of {topic}?',
            f'Which approach is fundamental to {topic}?',
            f'What defines the core of {topic}?'
        ]
        
        for i in range(min(num_questions, 5)):
            correct_answer = random.choice(['A', 'B', 'C', 'D'])
            questions.append({
                'question': question_templates[i % len(question_templates)],
                'options': [
                    'A) Fundamental principle',
                    'B) Advanced technique',
                    'C) Basic operation',
                    'D) Complex algorithm'
                ],
                'correct': correct_answer
            })
        return questions
    
    def parse_quiz_response(self, response):
        questions = []
        lines = response.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_question and current_question.get('options'):
                    # Randomize correct answer if not set
                    if not current_question.get('correct'):
                        current_question['correct'] = random.choice(['A', 'B', 'C', 'D'])
                    questions.append(current_question)
                current_question = {
                    'question': line.split(':', 1)[1].strip(),
                    'options': [],
                    'correct': None
                }
            elif line.startswith(('A)', 'B)', 'C)', 'D)')) and current_question:
                current_question['options'].append(line)
            elif 'correct' in line.lower() and ':' in line and current_question:
                # Extract correct answer more carefully
                correct_part = line.split(':', 1)[1].strip().upper()
                # Get first letter that's A, B, C, or D
                for char in correct_part:
                    if char in ['A', 'B', 'C', 'D']:
                        current_question['correct'] = char
                        break
        
        if current_question and current_question.get('options'):
            if not current_question.get('correct'):
                # Randomize if still not found
                current_question['correct'] = random.choice(['A', 'B', 'C', 'D'])
            questions.append(current_question)
        
        return questions
    
    def learn_from_interaction(self, query, profile, response, feedback):
        user_id = profile.get('name', 'unknown')
        
        if user_id not in self.memory['user_preferences']:
            self.memory['user_preferences'][user_id] = {
                'successful_topics': [],
                'difficult_topics': [],
                'preferred_style': profile.get('learning_style', 'general'),
                'interaction_count': 0
            }
        
        user_prefs = self.memory['user_preferences'][user_id]
        user_prefs['interaction_count'] += 1
        
        if feedback == 'good':
            topic = self.extract_topic_from_query(query)
            if topic and topic not in user_prefs['successful_topics']:
                user_prefs['successful_topics'].append(topic)
            
            pattern_key = f"{profile.get('difficulty', 'medium')}_{profile.get('learning_style', 'general')}"
            if pattern_key not in self.decision_patterns['success_indicators']:
                self.decision_patterns['success_indicators'][pattern_key] = 0
            self.decision_patterns['success_indicators'][pattern_key] += 1
            
        elif feedback == 'bad':
            topic = self.extract_topic_from_query(query)
            if topic and topic not in user_prefs['difficult_topics']:
                user_prefs['difficult_topics'].append(topic)
        
        self.save_memory()
        print(f"🧠 RAG learned from interaction: {feedback} feedback on {query[:30]}...")
    
    def extract_topic_from_query(self, query):
        query_lower = query.lower()
        topics = ['vector', 'matrix', 'calculus', 'physics', 'chemistry', 'biology', 'quantum', 'machine learning']
        for topic in topics:
            if topic in query_lower:
                return topic
        return None
    
    def fine_tune_on_interactions(self, interactions_file="user_interactions.json"):
        """Fine-tune the model based on user interactions"""
        
        if not Path(interactions_file).exists():
            print("No interaction data found for fine-tuning")
            return
        
        with open(interactions_file, 'r') as f:
            interactions = json.load(f)
        
        print(f" Found {len(interactions)} interactions for fine-tuning")
        
        # Create training data from interactions
        training_data = []
        for interaction in interactions:
            if interaction.get('feedback') == 'good':
                training_data.append({
                    'input': interaction['query'],
                    'output': interaction['response'],
                    'profile': interaction['profile']
                })
        
        print(f" Prepared {len(training_data)} training examples")
        
        # Save training data for fine-tuning
        with open(MODELS_DIR / 'training_data.json', 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(" Training data saved. Use this data to fine-tune your model externally.")
        
        return training_data
    
    def load_memory(self):
        memory_file = MODELS_DIR / 'rag_memory.json'
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                return json.load(f)
        return {
            'successful_patterns': {},
            'user_preferences': {},
            'topic_expertise': {},
            'response_quality': {},
            'learning_history': []
        }
    
    def save_memory(self):
        memory_file = MODELS_DIR / 'rag_memory.json'
        with open(memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def load_decision_patterns(self):
        patterns_file = MODELS_DIR / 'decision_patterns.json'
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        return {
            'query_classification': {},
            'difficulty_mapping': {},
            'style_preferences': {},
            'success_indicators': {}
        }
    
    def load_response_templates(self):
        return {
            'explanation': {
                'visual': "Let me show you {topic} with clear examples:\n{content}\n\nVisual Tips: {visual_tips}",
                'hands-on': "Let's explore {topic} through practical examples:\n{content}\n\nTry This: {practical_tips}",
                'auditory': "Listen to this explanation of {topic}:\n{content}\n\nDiscussion Points: {discussion_tips}"
            },
            'difficulty': {
                'easy': "Hi {name}! Let's start with the basics of {topic}:\n{simple_content}",
                'medium': "Hello {name}! Here's a solid explanation of {topic}:\n{content}",
                'hard': "Hi {name}! Let's dive deep into {topic}:\n{advanced_content}"
            }
        }
    
    def add_to_knowledge_base(self, query, llm_response):
        """Add LLM-generated content to knowledge base"""
        # Clean the response for storage
        clean_response = llm_response.replace('\n', ' ').replace('|', '-')[:500]
        
        new_entry = {
            'content': f"topic: {query} | description: {clean_response} | keywords: {query.lower()}",
            'source': 'llm_generated',
            'topic': query
        }
        
        # Add to dataframe
        new_df = pd.DataFrame([new_entry])
        self.kb = pd.concat([self.kb, new_df], ignore_index=True)
        
        # Rebuild retriever with new data
        if not self.kb.empty:
            normalized_content = [self.normalize_text(content) for content in self.kb['content']]
            self.doc_matrix = self.vectorizer.fit_transform(normalized_content)
            print(f" Knowledge base updated: {len(self.kb)} total entries")
        
        # Save to CSV
        llm_kb_file = DATA_DIR / 'llm_generated_knowledge.csv'
        try:
            if llm_kb_file.exists():
                existing_df = pd.read_csv(llm_kb_file)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(llm_kb_file, index=False)
            else:
                new_df.to_csv(llm_kb_file, index=False)
            print(f"💾 Saved to {llm_kb_file.name}")
        except Exception as e:
            print(f" Could not save to CSV: {e}")
    
    def generate_assessment_quiz(self, topic):
        """Generate 3 easy questions to assess user's knowledge on unknown topic"""
        if not self.llm:
            return None
        
        prompt = f"""Generate exactly 3 EASY quiz questions about "{topic}" to assess basic understanding.

Format each question exactly like this:
Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: A

Make questions simple and fundamental."""
        
        if self.llm == "ollama":
            response = self.call_ollama(prompt)
        else:
            response = "Q1: What is the basic concept?\nA) Concept A\nB) Concept B\nC) Concept C\nD) Concept D\nCorrect: A"
        
        return self.parse_quiz_response(response)
    
    def initialize_learning_weights(self):
        return {
            'retrieval_relevance': 0.4,
            'user_history': 0.3,
            'difficulty_match': 0.2,
            'style_preference': 0.1
        }

class ChatInterface:
    def __init__(self, model_path=None):
        print(" Initializing DeepSeek-R1 RAG System...")
        self.rag_system = None
        self.current_user = None
        self.model_path = model_path
        self.interactions = []
        self.conversation_context = {
            'current_topic': None,
            'follow_up_count': 0,
            'max_follow_ups': 10,
            'topic_content': None,
            'last_response': None
        }
        
    def collect_profile(self):
        print(" Welcome to DeepSeek-R1 Personalized Learning Assistant!")
        print(" Powered by DeepSeek-R1:1.5b - Advanced reasoning model")
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
            answer = input(f" {question}: ").strip()
            profile[key] = answer if answer else "Not specified"
        
        return profile
    
    def save_profile(self, profile):
        profile_file = PROFILES_DIR / f"{profile['name'].replace(' ', '_')}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f" Profile saved for {profile['name']}")
    
    def load_profile(self, name):
        profile_file = PROFILES_DIR / f"{name.replace(' ', '_')}.json"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_interaction(self, query, response, feedback=None):
        """Save interaction for fine-tuning"""
        interaction = {
            'query': query,
            'response': response,
            'profile': self.current_user,
            'feedback': feedback,
            'timestamp': str(pd.Timestamp.now())
        }
        self.interactions.append(interaction)
        
        # Save to file
        with open('user_interactions.json', 'w') as f:
            json.dump(self.interactions, f, indent=2)
    
    def extract_topic_from_query(self, query):
        """Extract main topic from user query"""
        query_lower = query.lower()
        
        # Common question patterns
        patterns = [
            'what is ',
            'what are ',
            'tell me about ',
            'explain ',
            'define ',
            'describe '
        ]
        
        for pattern in patterns:
            if pattern in query_lower:
                topic = query_lower.split(pattern)[1].strip()
                # Remove trailing question marks and extra words
                topic = topic.replace('?', '').split(' in ')[0].split(' on ')[0].strip()
                return topic
        
        return None
    
    def is_same_topic(self, user_input, current_topic):
        """Check if user input is about the same topic as current context"""
        if not current_topic:
            return False
        
        # Normalize both
        user_lower = user_input.lower()
        topic_lower = current_topic.lower()
        
        # Extract key terms from both
        user_terms = set(self.rag_system.normalize_text(user_input).split())
        topic_terms = set(self.rag_system.normalize_text(current_topic).split())
        
        # Remove common words
        stop_words = {'what', 'is', 'the', 'how', 'why', 'tell', 'me', 'about', 'in', 'of', 'and', 'or', 'to', 'a', 'an', 'difference', 'between'}
        user_terms = user_terms - stop_words
        topic_terms = topic_terms - stop_words
        
        # Check for word containment (e.g., "regression" in "linear regression")
        for topic_term in topic_terms:
            if len(topic_term) > 3:  # Only meaningful words
                for user_term in user_terms:
                    if topic_term in user_term or user_term in topic_term:
                        return True
        
        # Check overlap - need at least 20% common terms
        if not user_terms or not topic_terms:
            return False
        
        common_terms = user_terms & topic_terms
        similarity = len(common_terms) / max(len(user_terms), len(topic_terms))
        
        return similarity >= 0.2
    
    def is_follow_up_question(self, user_input):
        """Detect if this is a follow-up question"""
        follow_up_indicators = ['how', 'why', 'what about', 'can you explain', 'tell me more', 
                                'elaborate', 'example', 'more details', 'clarify', 'also', 'what is',
                                'what are', 'explain', 'describe', 'tell me']
        user_lower = user_input.lower()
        return any(indicator in user_lower for indicator in follow_up_indicators)
    
    def generate_llm_follow_up_response(self, user_input):
        """Generate response using LLM with conversation context for follow-ups"""
        prompt = f"""You are a personalized learning assistant.

Student Profile:
- Name: {self.current_user.get('name', 'Student')}
- Grade: {self.current_user.get('grade', 'Not specified')}
- Learning Style: {self.current_user.get('learning_style', 'Not specified')}
- Difficulty: {self.current_user.get('difficulty', 'medium')}

Previous Topic: {self.conversation_context['current_topic']}
Previous Discussion: {self.conversation_context['last_response'][:500] if self.conversation_context['last_response'] else 'None'}

Follow-up Question: {user_input}

Provide a clear, educational answer that:
1. Directly addresses their follow-up question
2. Builds on the previous topic context
3. Uses their learning style and difficulty level
4. Includes examples if helpful
5. Keep it concise (under 300 words)

Answer:"""
        
        if self.rag_system.llm == "ollama":
            return self.rag_system.call_ollama(prompt)
        elif isinstance(self.rag_system.llm, dict):
            return self.rag_system.call_transformers(prompt)
        else:
            return f"Follow-up on {self.conversation_context['current_topic']}: {user_input}"
    
    def chat_loop(self):
        print("\n DeepSeek-R1 Assistant is ready!")
        print("Commands: 'quiz' for practice, 'finetune' to improve, 'profile' for info, 'new topic' to reset, 'quit' to exit\n")
        
        while True:
            user_input = input(f" {self.current_user['name']}: ").strip()
            
            if user_input.lower() == 'quit':
                print(" Happy learning! See you next time!")
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
                    'topic_content': None,
                    'last_response': None
                }
                print(" Context reset. Ask me about a new topic!\n")
                continue
            elif user_input.lower() == 'quiz':
                self.generate_quiz_session()
                continue
            elif user_input.lower() == 'finetune':
                print(" Analyzing RAG memory and learning patterns...")
                self.show_rag_intelligence()
                self.rag_system.fine_tune_on_interactions()
                continue
            elif not user_input:
                continue
            
            # Check for special requests (quiz, preparation, etc.) - always use LLM
            quiz_keywords = ['quiz', 'preparation', 'prepare', 'practice questions', 'test', 'exam', 'assessment']
            is_quiz_request = any(keyword in user_input.lower() for keyword in quiz_keywords)
            
            if is_quiz_request:
                # Use LLM for quiz/preparation requests
                print("\n🎯 Quiz/Preparation request detected - Using DeepSeek-R1...")
                response = self.generate_llm_follow_up_response(user_input)
                self.conversation_context['current_topic'] = user_input
                self.conversation_context['follow_up_count'] = 0
                self.conversation_context['last_response'] = response
                is_new_topic = False
            else:
                # Check if this is a follow-up question on the SAME topic
                is_same_topic = self.is_same_topic(user_input, self.conversation_context['current_topic'])
                has_follow_up_pattern = self.is_follow_up_question(user_input)
                
                is_follow_up = (self.conversation_context['current_topic'] is not None and 
                               is_same_topic and
                               has_follow_up_pattern and
                               self.conversation_context['follow_up_count'] < self.conversation_context['max_follow_ups'])
                
                if is_follow_up:
                    # Use LLM for follow-up questions
                    print(f"\n Follow-up {self.conversation_context['follow_up_count'] + 1}/{self.conversation_context['max_follow_ups']} on: {self.conversation_context['current_topic']}")
                    print(" Using DeepSeek-R1 with context...")
                    
                    response = self.generate_llm_follow_up_response(user_input)
                    self.conversation_context['follow_up_count'] += 1
                    self.conversation_context['last_response'] = response
                    is_new_topic = False
                else:
                    # New topic or exceeded follow-up limit - use RAG
                    if self.conversation_context['follow_up_count'] >= self.conversation_context['max_follow_ups']:
                        print("\n Reached follow-up limit. Searching knowledge base for new context...")
                    else:
                        print("\n Searching knowledge base and generating response...")
                    
                    # Retrieve documents
                    retrieved_docs = self.rag_system.retrieve_documents(user_input)
                    
                    # Extract topic from query itself, not just from retrieved docs
                    query_topic = self.extract_topic_from_query(user_input)
                    topic_name = query_topic if query_topic else (retrieved_docs[0].get('topic', user_input) if retrieved_docs else user_input)
                    
                    # Check if topic is new to user AND not a follow-up
                    is_new_to_user = not self.rag_system.is_topic_in_user_kb(topic_name)
                    is_related_to_current = self.is_same_topic(user_input, self.conversation_context['current_topic'])
                    
                    # Only assess if truly new AND not related to current conversation
                    if is_new_to_user and not is_related_to_current and retrieved_docs:
                        user_level = self.assess_new_topic(topic_name)
                        print(f"✅ Assessment complete! Continuing with your question...\n")
                    
                    # Always use LLM with KB context for better responses
                    if retrieved_docs:
                        print("🤖 Generating enhanced answer using DeepSeek-R1 with knowledge base context...")
                        
                        # Build context from KB
                        kb_context = "\n".join([f"- {doc['content'][:300]}" for doc in retrieved_docs[:2]])
                        
                        # Get user's level for this topic if available
                        user_level = self.rag_system.get_user_topic_level(topic_name) or self.current_user.get('difficulty', 'medium')
                        
                        prompt = f"""You are a personalized learning assistant.

Student Profile:
- Name: {self.current_user.get('name', 'Student')}
- Grade: {self.current_user.get('grade', 'Not specified')}
- Learning Style: {self.current_user.get('learning_style', 'Not specified')}
- Current Level on this topic: {user_level}

Knowledge Base Context:
{kb_context}

Student Question: {user_input}

Provide a clear, educational answer that:
1. Uses the knowledge base context as reference
2. Explains in simple terms appropriate for {user_level} level
3. Includes practical examples
4. Matches their learning style: {self.current_user.get('learning_style', 'general')}
5. Keep it concise and engaging (under 400 words)

Answer:"""
                        
                        if self.rag_system.llm == "ollama":
                            response = self.rag_system.call_ollama(prompt)
                        elif isinstance(self.rag_system.llm, dict):
                            response = self.rag_system.call_transformers(prompt)
                        else:
                            response, is_new_topic = self.rag_system.generate_response(user_input, self.current_user, retrieved_docs)
                        
                        is_new_topic = False
                    else:
                        # No KB context, use LLM directly
                        response, is_new_topic = self.rag_system.generate_llm_response(user_input, self.current_user)
                    
                    # Save query to user KB
                    self.rag_system.save_to_user_kb(
                        query=user_input,
                        topic=topic_name,
                        response=response,
                        entry_type='query'
                    )
                    
                    # Update conversation context with the ACTUAL topic from this query
                    self.conversation_context['current_topic'] = topic_name
                    self.conversation_context['topic_content'] = response
                    self.conversation_context['follow_up_count'] = 0
                    self.conversation_context['last_response'] = response
            
            print("\n" + "="*60)
            print(response)
            print("="*60 + "\n")
            
            # If new topic, offer assessment quiz
            if is_new_topic:
                take_quiz = input(" I notice this is a new topic for you. Would you like to take a quick 3-question quiz to assess your understanding? (yes/no): ").lower()
                if take_quiz == 'yes':
                    self.run_assessment_quiz(user_input)
            
            # Get feedback for RAG learning
            feedback = input(" Was this response helpful? (good/bad/skip): ").lower()
            if feedback in ['good', 'bad']:
                self.save_interaction(user_input, response, feedback)
                self.rag_system.learn_from_interaction(user_input, self.current_user, response, feedback)
                print(" RAG system learned from your feedback!")
    
    def assess_new_topic(self, topic):
        """Assess user's knowledge on a new topic with 3 questions"""
        print(f"\n🎯 New topic detected: {topic}")
        print("📝 Let me assess your current understanding with 3 quick questions...\n")
        
        # Generate 3 assessment questions
        prompt = f"""Generate exactly 3 simple assessment questions about "{topic}" to evaluate basic understanding.

Format:
Q1: [Question]
Q2: [Question]
Q3: [Question]

Make questions clear and fundamental."""
        
        if self.rag_system.llm == "ollama":
            response = self.rag_system.call_ollama(prompt)
        else:
            response = f"Q1: What is {topic}?\nQ2: What are key concepts in {topic}?\nQ3: How is {topic} used?"
        
        # Parse questions
        questions = []
        for line in response.split('\n'):
            if line.strip().startswith('Q'):
                q = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                questions.append(q)
        
        # Ask questions and collect answers
        answers = []
        for i, q in enumerate(questions[:3], 1):
            print(f"Q{i}: {q}")
            ans = input("👉 Your answer: ").strip()
            answers.append(ans)
        
        # Evaluate level based on answers
        answered = sum(1 for a in answers if len(a) > 10)
        if answered >= 2:
            level = "intermediate"
            feedback = "👍 Good understanding! You have intermediate knowledge."
        elif answered == 1:
            level = "beginner"
            feedback = "📚 Basic understanding. You're at beginner level."
        else:
            level = "novice"
            feedback = "🌱 Starting fresh! You're at novice level."
        
        print(f"\n{feedback}")
        print(f"💾 Saving assessment to your knowledge base...\n")
        
        # Save to user KB
        self.rag_system.save_to_user_kb(
            query=f"Assessment: {topic}",
            topic=topic,
            response=f"Questions: {questions[:3]}, Answers: {answers}",
            quiz_score=f"{answered}/3",
            level=level,
            entry_type='assessment'
        )
        
        # Rebuild retriever with new data
        self.rag_system.build_retriever()
        
        return level
    
    def start(self):
        print(" Starting DeepSeek-R1 Learning Assistant...")
        print(" Using advanced reasoning capabilities for personalized education")
        print("💾 NEW: Personal knowledge base - All your learning tracked!")
        print("💡 NEW: Intelligent follow-up handling - up to 10 contextual follow-ups per topic!")
        print("   Type 'new topic' to reset context anytime.\n")
        
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
        
        # Initialize RAG with user name
        self.rag_system = LocalLLMRAGSystem(self.model_path, self.current_user['name'])
        
        # Load existing interactions
        if Path('user_interactions.json').exists():
            with open('user_interactions.json', 'r') as f:
                self.interactions = json.load(f)
        
        # Start chat
        self.chat_loop()
    
    def generate_quiz_session(self):
        # Main topics with expanded subtopics
        topics = {
            'mathematics': ['Algebra', 'Calculus', 'Geometry', 'Statistics', 'Linear Algebra', 'Trigonometry', 'Number Theory', 'Probability'],
            'physics': ['Mechanics', 'Thermodynamics', 'Electromagnetism', 'Optics', 'Quantum Physics', 'Relativity', 'Waves', 'Kinematics'],
            'chemistry': ['Organic Chemistry', 'Inorganic Chemistry', 'Physical Chemistry', 'Biochemistry', 'Analytical Chemistry', 'Chemical Bonding', 'Reactions'],
            'biology': ['Cell Biology', 'Genetics', 'Ecology', 'Evolution', 'Anatomy', 'Microbiology', 'Physiology', 'Botany'],
            'computer science': ['Arrays', 'Linked Lists', 'Stacks', 'Queues', 'Trees', 'Graphs', 'Hash Tables', 'Sorting', 'Searching', 'Dynamic Programming', 'Recursion', 'OOP', 'Databases', 'Operating Systems', 'Networks', 'AI/ML', 'Web Development'],
            'vectors': ['Vector Operations', 'Dot Product', 'Cross Product', 'Vector Spaces', 'Applications', 'Magnitude', 'Direction'],
            'quantum mechanics': ['Wave Functions', 'Operators', 'Uncertainty Principle', 'Quantum States', 'Entanglement', 'Superposition']
        }
        
        print("\n Quiz Topics:")
        topic_list = list(topics.keys())
        for i, topic in enumerate(topic_list, 1):
            print(f"  {i}. {topic.title()}")
        
        try:
            # Main topic selection
            topic_input = input("\n Select topic (1-7 or name): ").strip()
            
            # Handle both number and name input (case-insensitive)
            if topic_input.isdigit():
                topic_choice = int(topic_input) - 1
                if 0 <= topic_choice < len(topic_list):
                    selected_main_topic = topic_list[topic_choice]
                else:
                    print(" Invalid choice!")
                    return
            else:
                # Try to match by name
                topic_lower = topic_input.lower()
                if topic_lower in topics:
                    selected_main_topic = topic_lower
                else:
                    print(" Topic not found!")
                    return
            
            # Show subtopics
            subtopics = topics[selected_main_topic]
            print(f"\n {selected_main_topic.title()} - Subtopics:")
            for i, subtopic in enumerate(subtopics, 1):
                print(f"  {i}. {subtopic}")
            print(f"  {len(subtopics) + 1}. Custom Topic (Enter your own)")
            
            # Subtopic selection
            subtopic_input = input(f"\n Select subtopic (1-{len(subtopics) + 1} or name): ").strip()
            
            if subtopic_input.isdigit():
                subtopic_choice = int(subtopic_input) - 1
                if subtopic_choice == len(subtopics):
                    # Custom topic option
                    custom_topic = input("\n Enter your custom topic: ").strip()
                    if custom_topic:
                        selected_subtopic = custom_topic
                    else:
                        print(" No topic entered!")
                        return
                elif 0 <= subtopic_choice < len(subtopics):
                    selected_subtopic = subtopics[subtopic_choice]
                else:
                    print(" Invalid choice!")
                    return
            else:
                # Try to match by name (case-insensitive)
                matched = [st for st in subtopics if st.lower() == subtopic_input.lower()]
                if matched:
                    selected_subtopic = matched[0]
                else:
                    # Treat as custom topic
                    print(f"ℹ '{subtopic_input}' not in list, using as custom topic")
                    selected_subtopic = subtopic_input
            
            # Difficulty selection
            print("\n🎯 Difficulty:")
            print("  1. Easy")
            print("  2. Medium")
            print("  3. Hard")
            
            diff_input = input("\n Select (1-3 or easy/medium/hard): ").strip().lower()
            
            if diff_input.isdigit():
                difficulty = {1: 'easy', 2: 'medium', 3: 'hard'}.get(int(diff_input), 'medium')
            elif diff_input in ['easy', 'medium', 'hard']:
                difficulty = diff_input
            else:
                difficulty = 'medium'
                print(" Invalid input, using medium difficulty")
            
            # Number of questions
            print("\n📝 Questions:")
            print("  1. Quick (3)")
            print("  2. Standard (5)")
            print("  3. Long (10)")
            
            num_input = input("\n Select (1-3 or number): ").strip()
            
            if num_input.isdigit():
                num_val = int(num_input)
                if num_val in [1, 2, 3]:
                    num_questions = {1: 3, 2: 5, 3: 10}[num_val]
                elif 1 <= num_val <= 20:
                    num_questions = num_val
                else:
                    num_questions = 5
                    print(" Invalid input, using 5 questions")
            else:
                num_questions = 5
            
            # Run quiz with full topic name
            full_topic = f"{selected_main_topic} - {selected_subtopic}"
            self.run_advanced_quiz(full_topic, difficulty, num_questions)
            
        except KeyboardInterrupt:
            print("\n Quiz cancelled")
        except ValueError:
            print(" Invalid input!")
        except Exception as e:
            print(f" Error: {e}")
    
    def run_assessment_quiz(self, topic):
        """Run quick 3-question assessment quiz for new topics"""
        print(f"\n Generating assessment quiz on '{topic}'...")
        questions = self.rag_system.generate_assessment_quiz(topic)
        
        if not questions or len(questions) == 0:
            print(" Could not generate quiz questions.")
            return
        
        score = 0
        total_questions = min(len(questions), 3)
        
        print(f"\n Quick Assessment - {topic}")
        print("=" * 50)
        
        for i, q in enumerate(questions[:3], 1):
            print(f"\nQuestion {i}/{total_questions}:")
            print(f" {q['question']}")
            
            if q.get('options'):
                for option in q['options']:
                    print(f"   {option}")
                
                user_answer = input("\n Your answer (A/B/C/D): ").strip().upper()
                correct_answer = str(q.get('correct', 'A')).upper()
                
                if user_answer == correct_answer:
                    print(" Correct!")
                    score += 1
                else:
                    print(f" Incorrect. Correct: {correct_answer}")
        
        percentage = (score / total_questions) * 100
        
        print("\n" + "=" * 50)
        print(f" Assessment Results: {score}/{total_questions} ({percentage:.1f}%)")
        
        # Determine level
        if percentage >= 70:
            level = "intermediate"
            print(" You have good understanding of this topic!")
        else:
            level = "beginner"
            print(" This topic needs more practice. I'll provide extra support!")
        
        # Save to user KB
        self.rag_system.save_to_user_kb(
            query=f"Assessment: {topic}",
            topic=topic,
            response="Quick assessment quiz",
            quiz_score=f"{score}/{total_questions} ({percentage:.1f}%)",
            level=level,
            entry_type='assessment'
        )
        
        # Rebuild retriever
        self.rag_system.build_retriever()
        self.rag_system.save_memory()
    
    def run_advanced_quiz(self, topic, difficulty, num_questions):
        print(f"\n Generating {difficulty.upper()} quiz on {topic.title()}...")
        questions = self.rag_system.generate_quiz(topic, difficulty, num_questions)
        
        if not questions:
            print(" Could not generate quiz questions. Please try again.")
            return
        
        score = 0
        total_questions = len(questions)
        
        print(f"\n {topic.title()} Quiz - {difficulty.upper()} Level")
        print("=" * 50)
        
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}/{total_questions}:")
            print(f" {q['question']}")
            
            if q.get('options'):
                for option in q['options']:
                    print(f"   {option}")
                
                user_answer = input("\n Your answer (A/B/C/D): ").strip().upper()
                correct_answer = str(q.get('correct', 'A')).upper()
                
                if user_answer == correct_answer:
                    print(" Correct! Well done!")
                    score += 1
                else:
                    print(f" Incorrect. The correct answer is: {correct_answer}")
            else:
                user_answer = input("👉 Your answer: ").strip()
                if user_answer:
                    print(" Good effort!")
                    score += 0.5
                else:
                    print(" No answer provided.")
        
        percentage = (score / total_questions) * 100
        
        print("\n" + "=" * 50)
        print(f" Quiz Results: {score}/{total_questions} ({percentage:.1f}%)")
        
        if percentage >= 90:
            print(" Outstanding! You've mastered this topic!")
            level = "advanced"
        elif percentage >= 70:
            print(" Great job! You have a solid understanding!")
            level = "intermediate"
        elif percentage >= 50:
            print(" Good effort! Review the topics you missed.")
            level = "beginner"
        else:
            print(" Keep studying! Focus on the fundamentals.")
            level = "novice"
        
        # Save quiz results to user KB
        print("💾 Saving quiz results to your knowledge base...")
        self.rag_system.save_to_user_kb(
            query=f"Quiz: {topic} ({difficulty})",
            topic=topic,
            response=f"{num_questions} questions, {difficulty} difficulty",
            quiz_score=f"{score}/{total_questions} ({percentage:.1f}%)",
            level=level,
            entry_type='quiz'
        )
        
        # Rebuild retriever with new data
        self.rag_system.build_retriever()
    
    def show_rag_intelligence(self):
        memory = self.rag_system.memory
        patterns = self.rag_system.decision_patterns
        
        print("\n RAG Intelligence Report:")
        print("=" * 40)
        
        # Show knowledge base stats
        print(f" Knowledge Base: {len(self.rag_system.kb)} entries")
        llm_generated = len(self.rag_system.kb[self.rag_system.kb['source'] == 'llm_generated'])
        if llm_generated > 0:
            print(f"   LLM-generated: {llm_generated} entries")
        
        if memory['user_preferences']:
            print(f"\n Users tracked: {len(memory['user_preferences'])}")
            for user, prefs in memory['user_preferences'].items():
                print(f"  {user}: {prefs['interaction_count']} interactions")
                if prefs['successful_topics']:
                    print(f"    Strong in: {', '.join(prefs['successful_topics'][:3])}")
                if prefs['difficult_topics']:
                    print(f"    Needs help: {', '.join(prefs['difficult_topics'][:3])}")
        
        if patterns['success_indicators']:
            print(f"\n Success Patterns:")
            for pattern, count in patterns['success_indicators'].items():
                print(f"  {pattern}: {count} successful interactions")
        
        weights = self.rag_system.learning_weights
        print(f"\n Learning Weights:")
        for weight, value in weights.items():
            print(f"  {weight}: {value:.2f}")
        
        print("\n RAG is getting smarter with each interaction!")

if __name__ == "__main__":
    print(" DeepSeek-R1 RAG Learning System")
    print(" Using DeepSeek-R1:1.5b model for personalized education")
    print()
    print(" NEW FEATURES:")
    print("  • LLM responses automatically added to knowledge base")
    print("  • Auto-quiz for new topics to assess your understanding")
    print("  • RAG learns from every interaction")
    print()
    print("Setup Instructions:")
    print("1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("2. Pull DeepSeek model: ollama pull deepseek-r1:1.5b")
    print("3. Start Ollama: ollama serve")
    print()
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            deepseek_available = any('deepseek-r1' in model.get('name', '') for model in models)
            
            if deepseek_available:
                print(" DeepSeek-R1:1.5b model detected and ready!")
                print(" Knowledge will be saved and expanded automatically!")
            else:
                print(" DeepSeek-R1:1.5b not found. Run: ollama pull deepseek-r1:1.5b")
                print("Available models:", [m.get('name') for m in models])
        else:
            print(" Ollama not running. Start with: ollama serve")
    except:
        print(" Cannot connect to Ollama. Please install and start Ollama first.")
    
    print()
    input("Press Enter to continue...")
    
    try:
        chat = ChatInterface()
        chat.start()
    except KeyboardInterrupt:
        print("\n\n Goodbye! Happy learning!")
    except Exception as e:
        print(f" Error: {e}")
        print("Please check your DeepSeek-R1 setup and try again.")