from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
from pathlib import Path
import json
import sys
import os
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import subprocess
import tempfile

# Add personalized_rag to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'personalized_rag'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'personalized_rag'))

from personalized_rag.local_llm_rag import LocalLLMRAGSystem
from personalized_rag.collaborative_learning import CollaborativeLearning
from personalized_rag.dashboard import LearningDashboard
from personalized_rag.mistake_analyzer import MistakeAnalyzer
from personalized_rag.exam_prep import ExamPreparation
from auto_learning import init_auto_learning
from web_search import WebSearcher
from analytics_engine import get_analytics

app = Flask(__name__)
app.secret_key = 'deepseek-rag-secret-key-2025'
CORS(app)

PROFILES_DIR = Path('personalized_rag/user_profiles')
PROFILES_DIR.mkdir(exist_ok=True)

# Store RAG instances per user
rag_instances = {}

# ============= AUTHENTICATION =============

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/.well-known/appspecific/<path:filename>')
def chrome_devtools(filename):
    # Return empty 204 to suppress Chrome DevTools json requests
    return '', 204

@app.route('/favicon.ico')
def favicon():
    # Return 204 No Content to suppress favicon 404 errors
    return '', 204

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    
    if not username:
        return jsonify({'success': False, 'error': 'Username required'}), 400
    
    profile_file = PROFILES_DIR / f"{username}.json"
    
    if profile_file.exists():
        return jsonify({'success': False, 'error': 'Username already exists'}), 400
    
    # Create profile
    profile = {
        'name': data.get('name', username),
        'age': data.get('age', ''),
        'grade': data.get('grade', ''),
        'favorite_topics': data.get('favorite_topics', ''),
        'weak_topics': data.get('weak_topics', ''),
        'learning_style': data.get('learning_style', 'general'),
        'difficulty': data.get('difficulty', 'medium'),
        'goals': data.get('goals', '')
    }
    
    with open(profile_file, 'w') as f:
        json.dump(profile, f, indent=2)
    
    # Create knowledge base
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    with open(kb_file, 'w') as f:
        f.write('timestamp,query,topic,response,quiz_score,level,type\n')
    
    return jsonify({'success': True, 'username': username})

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    
    if not username:
        return jsonify({'success': False, 'error': 'Username required'}), 400
    
    profile_file = PROFILES_DIR / f"{username}.json"
    
    if not profile_file.exists():
        return jsonify({'success': False, 'error': 'User not found'}), 404
    
    with open(profile_file, 'r') as f:
        profile = json.load(f)
    
    session['username'] = username
    session['profile'] = profile
    
    return jsonify({'success': True, 'username': username, 'profile': profile})

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    username = session.get('username')
    if username and username in rag_instances:
        del rag_instances[username]
    session.clear()
    return jsonify({'success': True})

@app.route('/api/auth/session')
def check_session():
    if 'username' in session:
        return jsonify({'logged_in': True, 'username': session['username'], 'profile': session.get('profile', {})})
    return jsonify({'logged_in': False})

@app.route('/api/favorites/toggle', methods=['POST'])
def toggle_favorite():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    data = request.json
    topic = data.get('topic')
    
    fav_file = PROFILES_DIR / f"{username}_favorites.json"
    
    if fav_file.exists():
        with open(fav_file, 'r') as f:
            favorites = json.load(f)
    else:
        favorites = {'topics': []}
    
    if topic in favorites['topics']:
        favorites['topics'].remove(topic)
        is_favorite = False
    else:
        favorites['topics'].append(topic)
        is_favorite = True
    
    with open(fav_file, 'w') as f:
        json.dump(favorites, f)
    
    return jsonify({'is_favorite': is_favorite, 'favorites': favorites['topics']})

@app.route('/api/favorites')
def get_favorites():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    fav_file = PROFILES_DIR / f"{username}_favorites.json"
    
    if fav_file.exists():
        with open(fav_file, 'r') as f:
            favorites = json.load(f)
        return jsonify(favorites)
    
    return jsonify({'topics': []})

@app.route('/api/chat/export', methods=['GET'])
def export_chat():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    if not kb_file.exists():
        return jsonify({'error': 'No chat history'}), 404
    
    import pandas as pd
    df = pd.read_csv(kb_file)
    queries = df[df['type'] == 'query']
    
    # Create simple text export
    export_text = f"Chat History - {username}\n"
    export_text += "=" * 50 + "\n\n"
    
    for _, row in queries.iterrows():
        export_text += f"Q: {row['query']}\n"
        export_text += f"A: {row['response'][:500]}...\n"
        export_text += f"Topic: {row['topic']}\n"
        export_text += f"Date: {row['timestamp']}\n"
        export_text += "-" * 50 + "\n\n"
    
    return jsonify({'content': export_text})

# ============= CHAT =============

@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('chat.html')

@app.route('/practice')
def practice_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('practice.html')

def get_rag_instance(username, model=None):
    if username not in rag_instances:
        # Get user's preferred model or use default
        if model is None:
            model_file = PROFILES_DIR / f"{username}_model.json"
            if model_file.exists():
                with open(model_file, 'r') as f:
                    model = json.load(f).get('model', 'deepseek-r1:1.5b')
            else:
                model = 'deepseek-r1:1.5b'
        rag_instances[username] = LocalLLMRAGSystem(user_name=username, use_external_sources=True, llm_model=model)
    return rag_instances[username]

def extract_topic_from_query(query):
    """Extract main topic from user query"""
    query_lower = query.lower().strip()
    
    # Remove common question patterns
    patterns = [
        'what is ', 'what are ', 'tell me about ', 'explain ', 
        'how does ', 'how do ', 'how is ', 'how ', 'describe ',
        'define ', 'can you explain ', 'tell me '
    ]
    
    for pattern in patterns:
        if query_lower.startswith(pattern):
            query_lower = query_lower[len(pattern):]
            break
    
    # Remove question marks and extra words
    query_lower = query_lower.replace('?', '').strip()
    
    # Remove common verbs that don't define topics
    query_lower = query_lower.replace(' works', '').replace(' work', '')
    
    # Take first meaningful part (before 'in', 'on', 'for', etc.)
    for separator in [' in ', ' on ', ' for ', ' with ', ' using ']:
        if separator in query_lower:
            query_lower = query_lower.split(separator)[0]
            break
    
    return query_lower.strip()

@app.route('/api/chat/history')
def chat_history():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    history = []
    if kb_file.exists():
        import pandas as pd
        df = pd.read_csv(kb_file)
        all_queries = df[(df['type'] == 'query') | (df['type'] == 'web_search') | (df['type'] == 'images')]
        
        for _, row in all_queries.iterrows():
            history.append({
                'query': str(row['query']),
                'response': str(row['response'])
            })
    
    return jsonify({'history': history})

@app.route('/api/chat/save', methods=['POST'])
def save_chat():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    data = request.json
    query = data.get('query', '')
    response = data.get('response', '')
    
    if not query or not response:
        return jsonify({'error': 'Query and response required'}), 400
    
    rag = get_rag_instance(username)
    
    # Determine entry type
    if query.startswith('[IMAGES]'):
        entry_type = 'images'
        topic = 'images'
    else:
        topic = extract_topic_from_query(query)
        entry_type = 'web_search'
    
    rag.save_to_user_kb(
        query=query,
        topic=topic,
        response=response,
        entry_type=entry_type
    )
    
    return jsonify({'success': True})

@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    message = data.get('message', '').strip()
    username = session['username']
    profile = session.get('profile', {})
    
    if not message:
        return jsonify({'error': 'Message required'}), 400
    
    rag = get_rag_instance(username)
    
    # Extract topic from query using helper function
    topic = extract_topic_from_query(message)
    
    # Check if topic is NEW before generating response
    is_new_topic = not rag.is_topic_in_user_kb(topic)
    
    # Debug logging
    print(f"[DEBUG] Query: {message}")
    print(f"[DEBUG] Extracted topic: {topic}")
    print(f"[DEBUG] Is new topic: {is_new_topic}")
    
    # Check if this is a follow-up question (but only if NOT a new topic)
    is_follow_up = (not is_new_topic) and rag.is_follow_up_question(message)
    
    # ALWAYS use LLM for better responses (with KB context if available)
    if is_follow_up:
        # Follow-up: Use LLM with context
        response_tuple = rag.generate_llm_response(message, profile, is_follow_up=True)
        if isinstance(response_tuple, tuple):
            response, _ = response_tuple
        else:
            response = response_tuple
    elif is_new_topic:
        # New topic: Use LLM with full profile context
        age = profile.get('age', 'Not specified')
        age_instruction = f"Explain for a {age}-year-old using very simple words" if age.isdigit() and int(age) <= 12 else f"Explain for age {age}"
        
        prompt = f"""You are a friendly educational assistant. {age_instruction}.

Student Profile:
- Name: {profile.get('name', 'Student')}
- Age: {age}
- Grade: {profile.get('grade', 'Not specified')}
- Favorite Topics: {profile.get('favorite_topics', 'None')}
- Weak Topics: {profile.get('weak_topics', 'None')}
- Learning Style: {profile.get('learning_style', 'general')}
- Difficulty: {profile.get('difficulty', 'medium')}
- Goals: {profile.get('goals', 'Not specified')}

Question: {message}

Provide a clear explanation that matches their age, grade, learning style, and difficulty level. Use examples related to their favorite topics when possible. Keep it under 250 words."""
        
        response = rag.call_ollama(prompt)
    else:
        # Known topic: Use LLM with KB context and full profile
        retrieved_docs = rag.retrieve_documents(message)
        age = profile.get('age', 'Not specified')
        age_instruction = f"Explain for a {age}-year-old using simple words" if age.isdigit() and int(age) <= 12 else f"Explain for age {age}"
        
        kb_context = "\n".join([f"- {doc['content'][:200]}" for doc in retrieved_docs[:2]]) if retrieved_docs else ""
        
        prompt = f"""You are a friendly educational assistant. {age_instruction}.

Student Profile:
- Age: {age}, Grade: {profile.get('grade', 'Not specified')}
- Favorite Topics: {profile.get('favorite_topics', 'None')}
- Weak Topics: {profile.get('weak_topics', 'None')}
- Learning Style: {profile.get('learning_style', 'general')}
- Difficulty: {profile.get('difficulty', 'medium')}

Context: {kb_context}

Question: {message}

Provide a clear explanation matching their profile. Keep under 250 words."""
        
        response = rag.call_ollama(prompt)
    
    # Get external resources BEFORE saving
    external_resources = None
    external_links_text = ''
    if rag.use_external_sources and rag.external:
        external_resources = rag.external.get_external_resources(message)
        
        # Format external links as text to save in KB
        if external_resources:
            if external_resources.get('pdfs'):
                external_links_text += '\n\n📚 Recommended Articles:\n'
                for pdf in external_resources['pdfs']:
                    external_links_text += f"- {pdf['title']}: {pdf['url']}\n"
            if external_resources.get('videos'):
                external_links_text += '\n🎥 Recommended Videos:\n'
                for video in external_resources['videos']:
                    external_links_text += f"- {video['title']} by {video['channel']}: {video['url']}\n"
    
    # Save to KB with external links included
    full_response = response + external_links_text
    rag.save_to_user_kb(query=message, topic=topic, response=full_response, entry_type='query')
    
    result = {
        'response': response,
        'external_resources': external_resources,
        'is_new_topic': is_new_topic,
        'topic': topic
    }
    
    print(f"[DEBUG] Returning is_new_topic: {is_new_topic}")
    
    return jsonify(result)

# ============= QUIZ =============

@app.route('/quiz')
def quiz_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('quiz.html')

@app.route('/api/quiz/topics')
def get_quiz_topics():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Return hardcoded topics like in RAG system
    topics = {
        'mathematics': ['Algebra', 'Calculus', 'Geometry', 'Statistics', 'Linear Algebra', 'Trigonometry', 'Number Theory', 'Probability'],
        'physics': ['Mechanics', 'Thermodynamics', 'Electromagnetism', 'Optics', 'Quantum Physics', 'Relativity', 'Waves', 'Kinematics'],
        'chemistry': ['Organic Chemistry', 'Inorganic Chemistry', 'Physical Chemistry', 'Biochemistry', 'Analytical Chemistry', 'Chemical Bonding', 'Reactions'],
        'biology': ['Cell Biology', 'Genetics', 'Ecology', 'Evolution', 'Anatomy', 'Microbiology', 'Physiology', 'Botany'],
        'computer science': ['Arrays', 'Linked Lists', 'Stacks', 'Queues', 'Trees', 'Graphs', 'Hash Tables', 'Sorting', 'Searching', 'Dynamic Programming', 'Recursion', 'OOP', 'Databases', 'Operating Systems', 'Networks', 'AI/ML', 'Web Development'],
        'vectors': ['Vector Operations', 'Dot Product', 'Cross Product', 'Vector Spaces', 'Applications', 'Magnitude', 'Direction'],
        'quantum mechanics': ['Wave Functions', 'Operators', 'Uncertainty Principle', 'Quantum States', 'Entanglement', 'Superposition']
    }
    
    return jsonify({'topics': topics})

@app.route('/api/quiz/generate', methods=['POST'])
def generate_quiz():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'medium')
    num_questions = data.get('num_questions', 5)
    
    username = session['username']
    rag = get_rag_instance(username)
    
    questions = rag.generate_quiz(topic, difficulty, num_questions)
    
    # Save generated quiz to KB immediately (full data)
    import json
    import pandas as pd
    quiz_data = json.dumps({'questions': questions, 'generated': True})
    rag.save_to_user_kb(
        query=f"Generated Quiz: {topic} ({difficulty})",
        topic=topic,
        response=quiz_data,  # Save full quiz data
        quiz_score='Not taken',
        level='pending',
        entry_type='quiz'
    )
    
    return jsonify({'questions': questions, 'topic': topic, 'difficulty': difficulty})

@app.route('/api/quiz/history')
def get_quiz_history():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    history = []
    if kb_file.exists():
        import pandas as pd
        import json
        import re
        df = pd.read_csv(kb_file)
        quiz_entries = df[df['type'] == 'quiz']
        
        for idx, row in quiz_entries.iterrows():
            quiz_data = None
            try:
                quiz_data = json.loads(row['response']) if pd.notna(row['response']) else None
            except:
                # Try to parse old format: "10 questions, hard difficulty"
                response_str = str(row['response'])
                if 'questions' in response_str and 'difficulty' in response_str:
                    match = re.search(r'(\d+)\s+questions,\s+(\w+)\s+difficulty', response_str)
                    if match:
                        num_q = int(match.group(1))
                        diff = match.group(2)
                        # Create placeholder quiz data for old entries
                        quiz_data = {
                            'questions': [{
                                'question': f'Question {i+1} (Historical data - details not available)',
                                'options': ['A) Option A', 'B) Option B', 'C) Option C', 'D) Option D'],
                                'correct': 'A'
                            } for i in range(num_q)],
                            'generated': False,
                            'legacy': True
                        }
            
            # Extract difficulty from query if available
            difficulty = 'medium'
            query_str = str(row.get('query', ''))
            if '(easy)' in query_str:
                difficulty = 'easy'
            elif '(medium)' in query_str:
                difficulty = 'medium'
            elif '(hard)' in query_str:
                difficulty = 'hard'
            
            history.append({
                'id': int(idx),
                'topic': str(row['topic']),
                'score': str(row['quiz_score']),
                'difficulty': difficulty,
                'timestamp': str(row['timestamp']),
                'quiz_data': quiz_data
            })
    
    return jsonify({'history': history})

@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'medium')
    answers = data.get('answers', [])
    questions = data.get('questions', [])
    
    username = session['username']
    rag = get_rag_instance(username)
    
    # Calculate score
    score = 0
    total = len(questions)
    results = []
    
    for i, q in enumerate(questions):
        user_answer = answers[i] if i < len(answers) else ''
        correct_answer = q.get('correct', 'A')
        is_correct = user_answer.upper() == correct_answer.upper()
        
        if is_correct:
            score += 1
        
        results.append({
            'question': q['question'],
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct
        })
    
    percentage = (score / total * 100) if total > 0 else 0
    
    # Determine level
    if percentage >= 90:
        level = 'advanced'
    elif percentage >= 70:
        level = 'intermediate'
    elif percentage >= 50:
        level = 'beginner'
    else:
        level = 'novice'
    
    # Save quiz with questions to KB (full data, not truncated)
    import json
    quiz_data = json.dumps({'questions': questions, 'answers': answers, 'results': results})
    rag.save_to_user_kb(
        query=f"Quiz: {topic} ({difficulty})",
        topic=topic,
        response=quiz_data,  # Save full quiz data
        quiz_score=f"{score}/{total} ({percentage:.1f}%)",
        level=level,
        entry_type='quiz'
    )
    
    return jsonify({
        'score': score,
        'total': total,
        'percentage': percentage,
        'level': level,
        'results': results
    })

# ============= EXAMS =============

@app.route('/exam')
def exam_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('exam.html')

@app.route('/api/exam/list', methods=['GET'])
def get_exams():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        username = session['username']
        profile_file = PROFILES_DIR / f"{username}.json"
        kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
        
        # Create exam_prep instance with LLM caller
        rag = get_rag_instance(username)
        exam_prep = ExamPreparation(str(kb_file), str(profile_file), rag.call_ollama)
        
        # Get all exams (not just upcoming)
        all_exams = exam_prep.get_all_exams()
        
        # Add days_until for each exam
        for exam in all_exams:
            try:
                exam_date = datetime.fromisoformat(exam['date'].replace('Z', '+00:00'))
                if exam_date.tzinfo is not None:
                    exam_date = exam_date.replace(tzinfo=None)
                days_until = (exam_date - datetime.now()).days
                exam['days_until'] = days_until
                exam['is_past'] = days_until < 0
            except:
                exam['days_until'] = 0
                exam['is_past'] = False
        
        return jsonify({'exams': all_exams})
    except Exception as e:
        print(f"Error in get_exams: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'exams': [], 'error': str(e)}), 500

@app.route('/api/exam/add', methods=['POST'])
def add_exam():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    profile_file = PROFILES_DIR / f"{username}.json"
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    data = request.json
    exam_name = data.get('exam_name', '')
    exam_date = data.get('exam_date', '')
    topics = data.get('topics', [])
    difficulty = data.get('difficulty', 'medium')
    
    if not exam_name or not exam_date or not topics:
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    rag = get_rag_instance(username)
    exam_prep = ExamPreparation(str(kb_file), str(profile_file), rag.call_ollama)
    
    exam_id = exam_prep.add_exam(exam_name, exam_date, topics, difficulty)
    
    # Auto-generate study plan immediately
    try:
        with open(profile_file, 'r') as f:
            profile = json.load(f)
        
        study_plan = exam_prep.generate_study_plan(exam_id, profile)
        
        if study_plan:
            return jsonify({'success': True, 'exam_id': exam_id, 'study_plan': study_plan})
    except Exception as e:
        print(f"Error auto-generating study plan: {e}")
    
    return jsonify({'success': True, 'exam_id': exam_id})

@app.route('/api/exam/<int:exam_id>/plan', methods=['POST'])
def generate_exam_plan(exam_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        username = session['username']
        profile_file = PROFILES_DIR / f"{username}.json"
        kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
        
        with open(profile_file, 'r') as f:
            profile = json.load(f)
        
        rag = get_rag_instance(username)
        exam_prep = ExamPreparation(str(kb_file), str(profile_file), rag.call_ollama)
        
        # Generate study plan
        study_plan = exam_prep.generate_study_plan(exam_id, profile)
        
        if not study_plan:
            return jsonify({'success': False, 'error': 'Exam not found'}), 404
        
        return jsonify({'success': True, 'study_plan': study_plan})
    except Exception as e:
        print(f"Error in generate_exam_plan: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/plan', methods=['GET'])
def get_exam_plan(exam_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        username = session['username']
        profile_file = PROFILES_DIR / f"{username}.json"
        kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
        
        rag = get_rag_instance(username)
        exam_prep = ExamPreparation(str(kb_file), str(profile_file), rag.call_ollama)
        
        exams = exam_prep.get_all_exams()
        exam = next((e for e in exams if e['id'] == exam_id), None)
        
        if not exam:
            return jsonify({'success': False, 'error': 'Exam not found'}), 404
        
        return jsonify({
            'success': True,
            'exam': exam,
            'exam_name': exam.get('name'),
            'exam_date': exam.get('date'),
            'topics': exam.get('topics'),
            'difficulty': exam.get('difficulty'),
            'study_plan': exam.get('study_plan'),
            'has_plan': exam.get('study_plan') is not None
        })
    except Exception as e:
        print(f"Error in get_exam_plan: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exam/<int:exam_id>/progress', methods=['POST'])
def mark_exam_progress(exam_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    profile_file = PROFILES_DIR / f"{username}.json"
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    data = request.json
    day_completed = data.get('day', 1)
    notes = data.get('notes', '')
    
    rag = get_rag_instance(username)
    exam_prep = ExamPreparation(str(kb_file), str(profile_file), rag.call_ollama)
    
    success = exam_prep.mark_progress(exam_id, day_completed, notes)
    
    if not success:
        return jsonify({'success': False, 'error': 'Exam not found'}), 404
    
    return jsonify({'success': True, 'message': 'Progress marked'})

@app.route('/api/exam/<int:exam_id>/delete', methods=['DELETE'])
def delete_exam(exam_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    profile_file = PROFILES_DIR / f"{username}.json"
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    rag = get_rag_instance(username)
    exam_prep = ExamPreparation(str(kb_file), str(profile_file), rag.call_ollama)
    
    exams_data = exam_prep.get_all_exams()
    exams = [e for e in exams_data if e['id'] != exam_id]
    
    # Save updated exams
    exams_file = exam_prep.exams_file
    with open(exams_file, 'w') as f:
        json.dump({'exams': exams}, f, indent=2)
    
    return jsonify({'success': True, 'message': 'Exam deleted'})

# ============= STATS =============

@app.route('/stats')
def stats_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('stats.html')

@app.route('/api/stats/dashboard')
def get_dashboard():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    dashboard = LearningDashboard(kb_file)
    stats = dashboard.get_stats_dict()
    
    # Add advanced analytics
    analytics = get_analytics(username)
    stats['analytics'] = analytics
    
    return jsonify(stats)

# ============= PROFILE =============

@app.route('/profile')
def profile_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('profile.html')

@app.route('/api/profile')
def get_profile():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    return jsonify(session.get('profile', {}))

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    profile_file = PROFILES_DIR / f"{username}.json"
    
    data = request.json
    
    with open(profile_file, 'r') as f:
        profile = json.load(f)
    
    profile.update(data)
    
    with open(profile_file, 'w') as f:
        json.dump(profile, f, indent=2)
    
    session['profile'] = profile
    
    return jsonify({'success': True, 'profile': profile})

# ============= COLLABORATION =============

@app.route('/collab')
def collab_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('collab.html')

@app.route('/analytics')
def analytics_page():
    if 'username' not in session:
        return redirect(url_for('login_page'))
    return render_template('analytics.html')

@app.route('/api/collab/users')
def get_users():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    collab = CollaborativeLearning(str(PROFILES_DIR))
    users = collab.get_all_users()
    
    return jsonify({'users': users})

@app.route('/api/collab/groups')
def get_groups():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    collab = CollaborativeLearning(str(PROFILES_DIR))
    groups = collab.get_study_groups(username)
    
    return jsonify({'groups': groups})

@app.route('/api/collab/groups', methods=['POST'])
def create_group():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    username = session['username']
    
    collab = CollaborativeLearning(str(PROFILES_DIR))
    group_id = collab.create_study_group(
        creator=username,
        members=data.get('members', [username]),
        topic=data.get('topic', ''),
        description=data.get('description', '')
    )
    
    return jsonify({'success': True, 'group_id': group_id})

@app.route('/api/collab/groups/<int:group_id>/messages', methods=['GET'])
def get_group_messages(group_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    messages_file = PROFILES_DIR / f"group_{group_id}_messages.json"
    
    if not messages_file.exists():
        return jsonify({'messages': []})
    
    with open(messages_file, 'r') as f:
        data = json.load(f)
    
    return jsonify({'messages': data.get('messages', [])})

@app.route('/api/collab/groups/<int:group_id>/messages', methods=['POST'])
def send_group_message(group_id):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    message = data.get('message', '').strip()
    username = session['username']
    
    if not message:
        return jsonify({'error': 'Message required'}), 400
    
    messages_file = PROFILES_DIR / f"group_{group_id}_messages.json"
    
    # Load existing messages
    if messages_file.exists():
        with open(messages_file, 'r') as f:
            messages_data = json.load(f)
    else:
        messages_data = {'messages': []}
    
    # Add new message
    new_message = {
        'id': len(messages_data['messages']) + 1,
        'username': username,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    messages_data['messages'].append(new_message)
    
    # Save messages
    with open(messages_file, 'w') as f:
        json.dump(messages_data, f, indent=2)
    
    return jsonify({'success': True, 'message': new_message})

@app.route('/api/review/<topic>', methods=['POST'])
def review_topic(topic):
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    review_file = PROFILES_DIR / f"{username}_reviews.json"
    
    if review_file.exists():
        with open(review_file, 'r') as f:
            reviews = json.load(f)
    else:
        reviews = {'topics': {}}
    
    reviews['topics'][topic] = {
        'last_reviewed': datetime.now().isoformat(),
        'review_count': reviews['topics'].get(topic, {}).get('review_count', 0) + 1
    }
    
    with open(review_file, 'w') as f:
        json.dump(reviews, f)
    
    return jsonify({'success': True, 'topic': topic})

@app.route('/api/theme/toggle', methods=['POST'])
def toggle_theme():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    theme_file = PROFILES_DIR / f"{username}_theme.json"
    
    if theme_file.exists():
        with open(theme_file, 'r') as f:
            theme_data = json.load(f)
    else:
        theme_data = {'dark_mode': False}
    
    theme_data['dark_mode'] = not theme_data.get('dark_mode', False)
    
    with open(theme_file, 'w') as f:
        json.dump(theme_data, f)
    
    return jsonify(theme_data)

@app.route('/api/theme')
def get_theme():
    if 'username' not in session:
        return jsonify({'dark_mode': False})
    
    username = session['username']
    theme_file = PROFILES_DIR / f"{username}_theme.json"
    
    if theme_file.exists():
        with open(theme_file, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({'dark_mode': False})

@app.route('/api/image/analyze', methods=['POST'])
def analyze_image():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Remove data URL prefix
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Try OCR extraction
        try:
            import pytesseract
            extracted_text = pytesseract.image_to_string(image)
        except:
            extracted_text = "[OCR not available - install tesseract]"
        
        if not extracted_text.strip():
            extracted_text = "[No text detected in image]"
        
        # Get user profile for personalized response
        username = session['username']
        profile = session.get('profile', {})
        
        # Generate AI analysis
        rag = get_rag_instance(username)
        
        age = profile.get('age', 'Not specified')
        age_instruction = f"Explain for a {age}-year-old using simple words" if age.isdigit() and int(age) <= 12 else f"Explain for age {age}"
        
        prompt = f"""You are a helpful homework assistant. {age_instruction}.

Student Profile:
- Age: {age}
- Grade: {profile.get('grade', 'Not specified')}
- Learning Style: {profile.get('learning_style', 'general')}

Extracted text from image:
{extracted_text}

Provide a clear explanation or solution:
- Identify what the problem/question is asking
- Explain the concept step-by-step
- Show the solution process
- Use age-appropriate language

Keep it under 300 words."""
        
        response = rag.call_ollama(prompt)
        
        return jsonify({
            'extracted_text': extracted_text,
            'analysis': response
        })
        
    except Exception as e:
        print(f"Image analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List available Ollama models"""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_list = [{'name': m['name'], 'size': m.get('size', 0)} for m in models]
            return jsonify({'models': model_list})
        return jsonify({'models': []})
    except:
        return jsonify({'models': []})

@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Select LLM model for user"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    data = request.json
    model = data.get('model', 'deepseek-r1:1.5b')
    
    # Save model preference
    model_file = PROFILES_DIR / f"{username}_model.json"
    with open(model_file, 'w') as f:
        json.dump({'model': model}, f)
    
    # Reinitialize RAG with new model
    if username in rag_instances:
        del rag_instances[username]
    
    return jsonify({'success': True, 'model': model})

@app.route('/api/models/current', methods=['GET'])
def get_current_model():
    """Get current model for user"""
    if 'username' not in session:
        return jsonify({'model': 'deepseek-r1:1.5b'})
    
    username = session['username']
    model_file = PROFILES_DIR / f"{username}_model.json"
    
    if model_file.exists():
        with open(model_file, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({'model': 'deepseek-r1:1.5b'})

@app.route('/api/finetune/start', methods=['POST'])
def start_finetuning():
    """Trigger model fine-tuning"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    model = data.get('model', 'Qwen/Qwen2.5-3B-Instruct')
    
    # Check if enough training data
    total_examples = 0
    for kb_file in PROFILES_DIR.glob('*_knowledge_base.csv'):
        try:
            import pandas as pd
            df = pd.read_csv(kb_file)
            total_examples += len(df[df['type'] == 'query'])
        except:
            pass
    
    if total_examples < 10:
        return jsonify({
            'error': f'Not enough training data. Have {total_examples}, need at least 10 examples.'
        }), 400
    
    # Start fine-tuning in background
    import subprocess
    subprocess.Popen(['python', 'finetune_llm.py', '--model', model])
    
    return jsonify({
        'status': 'started',
        'message': f'Fine-tuning {model} with {total_examples} examples',
        'examples': total_examples
    })

@app.route('/api/finetune/status', methods=['GET'])
def finetuning_status():
    """Check if fine-tuned model exists"""
    finetuned_path = Path('personalized_rag/local_models/finetuned_qwen')
    
    return jsonify({
        'exists': finetuned_path.exists(),
        'path': str(finetuned_path) if finetuned_path.exists() else None
    })

@app.route('/api/code/execute', methods=['POST'])
def execute_code():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    try:
        if language == 'python':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(['python3', f.name], capture_output=True, text=True, timeout=5)
                os.unlink(f.name)
                return jsonify({'output': result.stdout, 'error': result.stderr})
        elif language == 'javascript':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=5)
                os.unlink(f.name)
                return jsonify({'output': result.stdout, 'error': result.stderr})
        else:
            return jsonify({'error': 'Unsupported language'}), 400
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Execution timeout (5s limit)'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/practice/generate', methods=['POST'])
def generate_practice():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'medium')
    count = data.get('count', 3)
    
    username = session['username']
    profile = session.get('profile', {})
    rag = get_rag_instance(username)
    
    age = profile.get('age', 'Not specified')
    grade = profile.get('grade', 'Not specified')
    
    prompt = f"""Generate {count} clear practice problems about {topic}.
Difficulty: {difficulty}
Student: Age {age}, Grade {grade}

Return ONLY a JSON array. Each problem must have:
- "question": the problem statement
- "answer": the complete solution
- "concept": "{topic}"

Format: [{{"question":"...","answer":"...","concept":"{topic}"}}]

Make questions clear and age-appropriate. No extra text outside JSON."""
    
    response = rag.call_ollama(prompt)
    
    try:
        import re
        response = response.strip()
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            problems = json.loads(json_match.group())
            # Normalize field names
            for p in problems:
                if 'question' in p and 'problem' not in p:
                    p['problem'] = p['question']
                if 'answer' in p and 'solution' not in p:
                    p['solution'] = p['answer']
                if 'problem' not in p:
                    p['problem'] = 'Problem not generated properly'
                if 'solution' not in p:
                    p['solution'] = 'Solution not available'
                if 'concept' not in p:
                    p['concept'] = topic
        else:
            problems = [{'problem': response, 'solution': 'See above', 'concept': topic}]
    except Exception as e:
        print(f"Parse error: {e}")
        problems = [{'problem': 'Error generating problem', 'solution': response[:500], 'concept': topic}]
    
    # Save to knowledge base
    practice_data = json.dumps({'problems': problems, 'topic': topic, 'difficulty': difficulty})
    rag.save_to_user_kb(
        query=f"Practice: {topic} ({difficulty})",
        topic=topic,
        response=practice_data,
        entry_type='practice'
    )
    
    return jsonify({'problems': problems, 'topic': topic, 'difficulty': difficulty})

@app.route('/api/practice/history')
def get_practice_history():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    kb_file = PROFILES_DIR / f"{username}_knowledge_base.csv"
    
    history = []
    if kb_file.exists():
        import pandas as pd
        df = pd.read_csv(kb_file)
        practice_entries = df[df['type'] == 'practice']
        
        for idx, row in practice_entries.iterrows():
            try:
                practice_data = json.loads(row['response'])
                history.append({
                    'id': int(idx),
                    'topic': str(row['topic']),
                    'difficulty': practice_data.get('difficulty', 'medium'),
                    'timestamp': str(row['timestamp']),
                    'problems': practice_data.get('problems', [])
                })
            except:
                pass
    
    return jsonify({'history': history})

def extract_search_keywords(query):
    """Intelligent keyword extraction using NLP"""
    import re
    
    query_lower = query.lower().strip()
    
    # Remove question words and common phrases
    stop_patterns = [
        r'^(what is|what are|who is|who are|when is|when was|where is|how does|how do|why is|why are)\s+',
        r'^(tell me about|explain|describe|define|find|search for|look up|show me)\s+',
        r'^(information about|details about|facts about)\s+',
        r'\s+(please|thanks|thank you)$'
    ]
    
    for pattern in stop_patterns:
        query_lower = re.sub(pattern, '', query_lower)
    
    # Remove filler words
    filler_words = ['the', 'a', 'an', 'of', 'for', 'in', 'on', 'at', 'to', 'from']
    words = query_lower.split()
    keywords = [w for w in words if w not in filler_words and len(w) > 2]
    
    # Spell correction
    corrections = {
        'pyhton': 'python', 'javascirpt': 'javascript', 'machien': 'machine',
        'artifical': 'artificial', 'algoritm': 'algorithm', 'programing': 'programming'
    }
    
    keywords = [corrections.get(w, w) for w in keywords]
    
    return ' '.join(keywords) if keywords else query_lower

@app.route('/api/analytics', methods=['GET'])
def get_user_analytics():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    analytics = get_analytics(username)
    return jsonify(analytics)

@app.route('/api/search/web', methods=['POST'])
def search_web():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['username']
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query required'}), 400
    
    # Intelligent keyword extraction
    search_keywords = extract_search_keywords(query)
    
    print(f"[WEB SEARCH] Original: {query}")
    print(f"[WEB SEARCH] Keywords: {search_keywords}")
    
    try:
        searcher = WebSearcher()
        results = searcher.search(search_keywords)
        images = searcher.search_images(search_keywords, max_images=3)
        
        print(f"[WEB SEARCH] Found {len(results)} results, {len(images)} images")
        
        # Format results with sources
        formatted_results = []
        sources_text = ''
        
        for i, r in enumerate(results, 1):
            formatted_results.append(r)
            sources_text += f"{i}. {r['title']} - {r['source']}: {r['url']}\n"
            print(f"  {i}. {r['title']}: {r['url']}")
        
        # Don't save here - let frontend save with formatted HTML
        
        return jsonify({
            'query': search_keywords,
            'original_query': query,
            'results': formatted_results,
            'images': images,
            'count': len(formatted_results),
            'sources': sources_text
        })
    except Exception as e:
        print(f"[WEB SEARCH] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting DeepSeek RAG Web Application...")
    print("📍 Open: http://localhost:5001")
    
    # Auto-learning disabled by default (requires 16GB+ RAM)
    # Uncomment to enable:
    # init_auto_learning(finetune_every=20)
    # print("🎓 Auto-learning enabled: Model will learn from every query")
    
    print("⚠️  Auto-learning disabled (requires 16GB+ RAM)")
    print("💾 Knowledge base learning still active (instant)")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
