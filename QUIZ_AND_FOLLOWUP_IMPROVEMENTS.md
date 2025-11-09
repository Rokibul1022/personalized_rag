# Quiz Generation & Follow-up Question Improvements

## Overview
Enhanced the RAG system to support dynamic quiz generation for ANY topic and intelligent follow-up question handling using LLM.

## Key Features

### 1. Dynamic Quiz Generation for Any Topic
**Before:** Hardcoded quiz topics (quantum numbers, chemical bonding, photosynthesis, etc.)
**After:** Generate quizzes for ANY topic the user is learning

#### Changes:
- **User can enter any topic** when requesting a quiz (arrays, GPU, pressure, scoping, etc.)
- **Customizable number of questions** (1-10 questions)
- **AI-powered question generation** (in rag_api.py with Gemini)
- **Template-based generation** (in main.py for basic RAG)

#### Usage:
```
User: "quiz"
System: "Enter any topic you want to practice (e.g., arrays, photosynthesis, calculus, etc.)"
User: "arrays"
System: "Number of questions (1-10):"
User: "5"
```

### 2. Intelligent Follow-up Question Handling
**Problem:** When users ask follow-up questions about the same topic, the RAG couldn't generate proper answers because it was searching the knowledge base again instead of using context.

**Solution:** Track conversation context and use LLM for up to 10 follow-up questions on the same topic.

#### How It Works:
1. **First Question:** User asks about a topic (e.g., "tell me about arrays")
   - System searches knowledge base
   - Stores topic context
   - Resets follow-up counter to 0

2. **Follow-up Questions (1-10):** User asks related questions
   - System detects follow-up indicators: "how", "why", "what about", "can you explain", "tell me more", etc.
   - Uses LLM with conversation context instead of RAG retrieval
   - Increments follow-up counter
   - Shows: "Follow-up 3/10 - Using AI context..."

3. **After 10 Follow-ups:** System automatically resets
   - Searches knowledge base again for fresh context
   - Resets counter to 0

4. **Manual Reset:** User can type "new topic" to reset context anytime

#### Example Conversation:
```
User: "tell me about arrays"
System: [Searches knowledge base, provides answer]
Context: topic="arrays", follow_up_count=0

User: "how do I access elements?"
System: "Follow-up 1/10 - Using AI context..."
[Uses LLM with array context]

User: "what about insertion?"
System: "Follow-up 2/10 - Using AI context..."
[Uses LLM with array context]

... (up to 10 follow-ups)

User: "new topic"
System: "Context reset. Ask me about a new topic!"
```

## Technical Implementation

### Conversation Context Structure:
```python
self.conversation_context = {
    'current_topic': None,           # Current topic being discussed
    'follow_up_count': 0,            # Number of follow-ups (0-10)
    'max_follow_ups': 10,            # Maximum allowed follow-ups
    'topic_content': None            # Previous content for context
}
```

### Follow-up Detection:
```python
follow_up_indicators = [
    'how', 'why', 'what about', 'can you explain', 
    'tell me more', 'elaborate', 'example', 
    'more details', 'clarify', 'also'
]
```

### LLM Follow-up Prompt (rag_api.py):
- Includes student profile (name, grade, learning style, difficulty)
- Includes current topic context
- Includes previous content
- Asks LLM to build on previous context
- Maintains personalization

## Files Modified

### 1. `/personalized_rag/rag_api.py` (AI-Powered RAG)
- Added `conversation_context` tracking
- Added `is_follow_up_question()` method
- Added `generate_llm_follow_up_response()` method
- Updated `generate_quiz()` to accept any topic + num_questions
- Updated `generate_quiz_session()` for dynamic topic input
- Updated `run_quiz()` to show generation progress
- Updated `chat_loop()` with follow-up logic

### 2. `/personalized_rag/main.py` (Basic RAG)
- Added `conversation_context` tracking
- Updated `generate_quiz()` with generic question templates
- Updated `generate_quiz_session()` for dynamic topic input
- Updated `run_quiz()` to show generation progress
- Updated `chat_loop()` with follow-up counter display

## Benefits

1. **Flexibility:** Quiz generation works for ANY topic, not just predefined ones
2. **Context Awareness:** System remembers conversation context for better follow-up answers
3. **Intelligent Routing:** Uses LLM for follow-ups (better quality) and RAG for new topics (grounded in knowledge base)
4. **User Control:** Users can reset context with "new topic" command
5. **Scalability:** No need to hardcode quiz questions for every possible topic
6. **Better Learning Experience:** Follow-up questions get contextual answers instead of generic retrieval

## Commands

- `quiz` - Generate quiz on any topic
- `new topic` - Reset conversation context
- `profile` - View user profile
- `quit` - Exit the system

## Example Use Cases

### Arrays Topic:
```
User: "I am learning arrays"
System: [Provides array explanation from knowledge base]

User: "make a quiz preparation on array"
System: "Enter any topic: arrays"
        "Number of questions: 5"
        [Generates 5 array questions]

User: "how do I insert elements?"
System: "Follow-up 1/10 - Using AI context..."
        [Provides contextual answer about array insertion]

User: "what about deletion?"
System: "Follow-up 2/10 - Using AI context..."
        [Provides contextual answer about array deletion]
```

### Any Other Topic:
- GPU and machine learning
- Pressure and force
- Scoping in programming
- Photosynthesis
- Quantum mechanics
- Calculus
- etc.

## Future Enhancements

1. Store conversation history for better context
2. Add topic similarity detection to maintain context even without explicit follow-up indicators
3. Implement conversation summarization after 10 follow-ups
4. Add difficulty progression in quiz questions
5. Track quiz performance per topic
