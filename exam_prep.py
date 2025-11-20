import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class ExamPreparation:
    def __init__(self, user_kb_file, profile_file, llm_caller):
        self.user_kb_file = user_kb_file
        self.profile_file = profile_file
        self.llm_caller = llm_caller
        self.exams_file = Path(profile_file).parent / f"{Path(profile_file).stem}_exams.json"
        self._init_exams_file()
    
    def _init_exams_file(self):
        """Initialize exams file if it doesn't exist"""
        if not self.exams_file.exists():
            with open(self.exams_file, 'w') as f:
                json.dump({"exams": []}, f, indent=2)
    
    def add_exam(self, exam_name, exam_date, topics, difficulty='medium'):
        """Add a new exam to track"""
        with open(self.exams_file, 'r') as f:
            data = json.load(f)
        
        exam = {
            'id': len(data['exams']) + 1,
            'name': exam_name,
            'date': exam_date,
            'topics': topics,
            'difficulty': difficulty,
            'added_on': datetime.now().isoformat(),
            'study_plan': None,
            'progress': []
        }
        
        data['exams'].append(exam)
        
        with open(self.exams_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return exam['id']
    
    def generate_study_plan(self, exam_id, profile):
        """Generate detailed study plan using LLM"""
        with open(self.exams_file, 'r') as f:
            data = json.load(f)
        
        exam = next((e for e in data['exams'] if e['id'] == exam_id), None)
        if not exam:
            return None
        
        # Calculate days until exam
        exam_date = datetime.fromisoformat(exam['date'])
        days_until = (exam_date - datetime.now()).days
        
        if days_until < 0:
            return "Exam date has passed!"
        
        # Generate plan using LLM
        prompt = f"""Create a detailed {days_until}-day study plan for {exam['name']}.

Student Profile:
- Name: {profile.get('name', 'Student')}
- Grade: {profile.get('grade', 'N/A')}
- Learning Style: {profile.get('learning_style', 'general')}
- Weak Areas: {profile.get('weak_topics', 'None')}
- Difficulty Level: {exam['difficulty']}

Exam Details:
- Exam: {exam['name']}
- Topics: {', '.join(exam['topics'])}
- Days Until Exam: {days_until} days
- Exam Date: {exam['date'][:10]}

Create a structured study plan with:
1. Weekly breakdown (Week 1, Week 2, etc.)
2. Daily tasks and topics to cover
3. Practice quiz recommendations
4. Review sessions before exam
5. Time management tips
6. Focus on weak areas: {profile.get('weak_topics', 'general topics')}

Format as a clear, actionable plan. Keep it concise but comprehensive."""

        study_plan = self.llm_caller(prompt)
        
        # Save plan
        exam['study_plan'] = study_plan
        exam['plan_generated_on'] = datetime.now().isoformat()
        
        with open(self.exams_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return study_plan
    
    def get_all_exams(self):
        """Get all registered exams"""
        with open(self.exams_file, 'r') as f:
            data = json.load(f)
        return data['exams']
    
    def get_upcoming_exams(self):
        """Get exams that haven't passed yet"""
        exams = self.get_all_exams()
        upcoming = []
        
        for exam in exams:
            exam_date = datetime.fromisoformat(exam['date'])
            if exam_date >= datetime.now():
                days_until = (exam_date - datetime.now()).days
                exam['days_until'] = days_until
                upcoming.append(exam)
        
        return sorted(upcoming, key=lambda x: x['days_until'])
    
    def display_exam_plan(self, exam_id):
        """Display study plan for an exam"""
        with open(self.exams_file, 'r') as f:
            data = json.load(f)
        
        exam = next((e for e in data['exams'] if e['id'] == exam_id), None)
        if not exam:
            print("❌ Exam not found!")
            return
        
        exam_date = datetime.fromisoformat(exam['date'])
        days_until = (exam_date - datetime.now()).days
        
        print("\n" + "="*80)
        print(f"📚 EXAM PREPARATION PLAN".center(80))
        print("="*80)
        
        print(f"\n📝 Exam: {exam['name']}")
        print(f"📅 Date: {exam['date'][:10]}")
        print(f"⏰ Days Until Exam: {days_until} days")
        print(f"📖 Topics: {', '.join(exam['topics'])}")
        print(f"🎯 Difficulty: {exam['difficulty'].upper()}")
        
        if exam.get('study_plan'):
            print("\n" + "="*80)
            print("📋 YOUR PERSONALIZED STUDY PLAN")
            print("="*80 + "\n")
            print(exam['study_plan'])
        else:
            print("\n⚠️  Study plan not generated yet. Generate it first!")
        
        print("\n" + "="*80)
    
    def display_all_exams(self):
        """Display all upcoming exams"""
        upcoming = self.get_upcoming_exams()
        
        if not upcoming:
            print("\n📚 No upcoming exams registered.")
            return
        
        print("\n" + "="*80)
        print("📚 UPCOMING EXAMS".center(80))
        print("="*80 + "\n")
        
        for exam in upcoming:
            status = "✅ Plan Ready" if exam.get('study_plan') else "⏳ Plan Pending"
            print(f"{exam['id']}. {exam['name']} - {exam['days_until']} days away {status}")
            print(f"   📅 Date: {exam['date'][:10]}")
            print(f"   📖 Topics: {', '.join(exam['topics'])}")
            print()
    
    def mark_progress(self, exam_id, day_completed, notes=""):
        """Mark a day as completed in study plan"""
        with open(self.exams_file, 'r') as f:
            data = json.load(f)
        
        exam = next((e for e in data['exams'] if e['id'] == exam_id), None)
        if exam:
            exam['progress'].append({
                'day': day_completed,
                'completed_on': datetime.now().isoformat(),
                'notes': notes
            })
            
            with open(self.exams_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        return False
    
    def get_today_tasks(self, exam_id):
        """Get today's tasks from study plan (simplified)"""
        with open(self.exams_file, 'r') as f:
            data = json.load(f)
        
        exam = next((e for e in data['exams'] if e['id'] == exam_id), None)
        if not exam or not exam.get('study_plan'):
            return None
        
        # Calculate which day of the plan we're on
        added_date = datetime.fromisoformat(exam['added_on'])
        days_since_start = (datetime.now() - added_date).days + 1
        
        return f"Day {days_since_start} of your study plan"
