    def generate_quiz(self, topic, difficulty='medium', num_questions=5, profile=None):
        """Generate quiz questions using DeepSeek-R1 with specific difficulty"""
        
        if not self.llm:
            return [{'question': f'What is {topic}?', 'options': ['A) Option 1', 'B) Option 2', 'C) Option 3', 'D) Option 4'], 'correct': 'A'}]
        
        prompt = f"""<|im_start|>system
You are an expert quiz generator for educational purposes.
<|im_end|>

<|im_start|>user
Generate {num_questions} quiz questions about "{topic}" with {difficulty} difficulty.

Format:
Q1: [Question]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: A

Make questions {difficulty} level.
<|im_end|>

<|im_start|>assistant"""

        if self.llm == "ollama":
            response = self.call_ollama(prompt)
        else:
            response = "Q1: What is the main concept?\nA) Concept A\nB) Concept B\nC) Concept C\nD) Concept D\nCorrect: A"
        
        return self.parse_quiz_response(response)

    def parse_quiz_response(self, response):
        questions = []
        lines = response.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_question:
                    questions.append(current_question)
                current_question = {
                    'question': line.split(':', 1)[1].strip(),
                    'options': [],
                    'correct': None
                }
            elif line.startswith(('A)', 'B)', 'C)', 'D)')) and current_question:
                current_question['options'].append(line)
            elif line.startswith('Correct:') and current_question:
                current_question['correct'] = line.split(':', 1)[1].strip()
        
        if current_question:
            questions.append(current_question)
        
        return questions

    def generate_quiz_session(self):
        topics = ['mathematics', 'physics', 'chemistry', 'biology', 'computer science', 'vectors', 'quantum mechanics']
        
        print("\n📚 Quiz Topics:")
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic.title()}")
        
        try:
            topic_choice = int(input("\nSelect topic (1-7): ")) - 1
            if 0 <= topic_choice < len(topics):
                selected_topic = topics[topic_choice]
                
                print("\n🎯 Difficulty:")
                print("  1. Easy")
                print("  2. Medium")
                print("  3. Hard")
                
                diff_choice = int(input("\nSelect (1-3): "))
                difficulty = {1: 'easy', 2: 'medium', 3: 'hard'}.get(diff_choice, 'medium')
                
                print("\n📝 Questions:")
                print("  1. Quick (3)")
                print("  2. Standard (5)")
                print("  3. Long (10)")
                
                num_choice = int(input("\nSelect (1-3): "))
                num_questions = {1: 3, 2: 5, 3: 10}.get(num_choice, 5)
                
                self.run_advanced_quiz(selected_topic, difficulty, num_questions)
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a number!")

    def run_advanced_quiz(self, topic, difficulty, num_questions):
        print(f"\n🎯 Generating {difficulty.upper()} quiz on {topic.title()}...")
        questions = self.rag_system.generate_quiz(topic, difficulty, num_questions, self.current_user)
        
        if not questions:
            print("❌ Could not generate quiz questions. Please try again.")
            return
        
        score = 0
        total_questions = len(questions)
        
        print(f"\n📝 {topic.title()} Quiz - {difficulty.upper()} Level")
        print("=" * 50)
        
        for i, q in enumerate(questions, 1):
            print(f"\nQuestion {i}/{total_questions}:")
            print(f"❓ {q['question']}")
            
            if q.get('options'):
                for option in q['options']:
                    print(f"   {option}")
                
                user_answer = input("\n👉 Your answer (A/B/C/D): ").strip().upper()
                correct_answer = q.get('correct', '').upper()
                
                if user_answer == correct_answer:
                    print("✅ Correct! Well done!")
                    score += 1
                else:
                    print(f"❌ Incorrect. The correct answer is: {correct_answer}")
            else:
                user_answer = input("👉 Your answer: ").strip()
                if user_answer:
                    print("✅ Good effort!")
                    score += 0.5
                else:
                    print("❌ No answer provided.")
        
        percentage = (score / total_questions) * 100
        
        print("\n" + "=" * 50)
        print(f"📊 Quiz Results: {score}/{total_questions} ({percentage:.1f}%)")
        
        if percentage >= 90:
            print("🏆 Outstanding! You've mastered this topic!")
        elif percentage >= 70:
            print("🎉 Great job! You have a solid understanding!")
        elif percentage >= 50:
            print("👍 Good effort! Review the topics you missed.")
        else:
            print("📚 Keep studying! Focus on the fundamentals.")