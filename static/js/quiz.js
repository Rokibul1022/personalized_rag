let currentQuiz = null;

const quizSetup = document.getElementById('quizSetup');
const quizQuestions = document.getElementById('quizQuestions');
const quizResults = document.getElementById('quizResults');
const generateQuizBtn = document.getElementById('generateQuizBtn');
const submitQuizBtn = document.getElementById('submitQuizBtn');
const newQuizBtn = document.getElementById('newQuizBtn');

// Load topics
let allTopics = {};

async function loadTopics() {
    try {
        const response = await fetch('/api/quiz/topics');
        const data = await response.json();
        allTopics = data.topics;
        
        const select = document.getElementById('quizTopic');
        Object.keys(allTopics).forEach(mainTopic => {
            const option = document.createElement('option');
            option.value = mainTopic;
            option.textContent = mainTopic.charAt(0).toUpperCase() + mainTopic.slice(1);
            select.appendChild(option);
        });
        
        // Add change listener to show subtopics
        select.addEventListener('change', showSubtopics);
    } catch (error) {
        console.error('Failed to load topics:', error);
    }
}

function showSubtopics() {
    const mainTopic = document.getElementById('quizTopic').value;
    const subtopicContainer = document.getElementById('subtopicContainer');
    const subtopicSelect = document.getElementById('subtopicSelect');
    
    if (!mainTopic || mainTopic === '') {
        subtopicContainer.style.display = 'none';
        return;
    }
    
    const subtopics = allTopics[mainTopic] || [];
    subtopicSelect.innerHTML = '<option value="">Select subtopic...</option>';
    
    subtopics.forEach(subtopic => {
        const option = document.createElement('option');
        option.value = subtopic;
        option.textContent = subtopic;
        subtopicSelect.appendChild(option);
    });
    
    // Add custom option
    const customOption = document.createElement('option');
    customOption.value = 'custom';
    customOption.textContent = 'Custom Topic (Enter below)';
    subtopicSelect.appendChild(customOption);
    
    subtopicContainer.style.display = 'block';
}

loadTopics();
loadQuizHistory();

// Generate quiz
generateQuizBtn.addEventListener('click', async () => {
    const mainTopic = document.getElementById('quizTopic').value.trim();
    const subtopic = document.getElementById('subtopicSelect').value.trim();
    const customTopic = document.getElementById('customTopic').value.trim();
    const difficulty = document.getElementById('quizDifficulty').value;
    const num_questions = parseInt(document.getElementById('quizQuestions').value);
    
    let topic = '';
    
    // Determine final topic
    if (subtopic === 'custom' && customTopic) {
        topic = customTopic;
    } else if (subtopic && subtopic !== '') {
        topic = `${mainTopic} - ${subtopic}`;
    } else if (mainTopic) {
        alert('Please select a subtopic or enter a custom topic');
        return;
    } else {
        alert('Please select a topic');
        return;
    }
    
    generateQuizBtn.textContent = 'Generating...';
    generateQuizBtn.disabled = true;
    
    try {
        const response = await fetch('/api/quiz/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({topic, difficulty, num_questions})
        });
        
        if (!response.ok) {
            const error = await response.json();
            alert('Error: ' + (error.error || 'Failed to generate quiz'));
            return;
        }
        
        const data = await response.json();
        console.log('Quiz data received:', data);
        console.log('Questions:', data.questions);
        
        if (!data.questions || data.questions.length === 0) {
            alert('No questions generated. The LLM might not be running. Please try again.');
            console.error('No questions in response');
            return;
        }
        
        currentQuiz = data;
        console.log('Displaying questions...');
        displayQuestions(data.questions);
        
        quizSetup.style.display = 'none';
        quizQuestions.style.display = 'block';
        quizQuestions.style.visibility = 'visible';
        quizQuestions.style.opacity = '1';
        document.getElementById('questionsContainer').style.display = 'block';
        console.log('Quiz questions displayed');
        
        // Reload history to show newly generated quiz
        loadQuizHistory();
        
        // Force reflow
        setTimeout(() => {
            const container = document.getElementById('questionsContainer');
            console.log('Container after timeout:', container);
            console.log('Container children:', container.children.length);
            Array.from(container.children).forEach((child, i) => {
                console.log(`Child ${i}:`, child.style.display, child.style.visibility);
            });
        }, 100);
        
    } catch (error) {
        alert('Failed to generate quiz. Please try again.');
        console.error('Quiz generation error:', error);
    } finally {
        generateQuizBtn.textContent = 'Generate Quiz';
        generateQuizBtn.disabled = false;
    }
});

// Display questions
function displayQuestions(questions) {
    const container = document.getElementById('questionsContainer');
    console.log('Container element:', container);
    console.log('Questions to display:', questions);
    
    if (!container) {
        console.error('questionsContainer not found!');
        return;
    }
    
    container.innerHTML = '';
    
    if (!questions || questions.length === 0) {
        container.innerHTML = '<p style="color: red;">No questions available</p>';
        console.error('No questions to display');
        return;
    }
    
    console.log(`Displaying ${questions.length} questions`);
    
    questions.forEach((q, index) => {
        console.log(`Question ${index + 1}:`, q);
        
        const questionDiv = document.createElement('div');
        questionDiv.className = 'quiz-question';
        questionDiv.style.display = 'block';
        questionDiv.style.visibility = 'visible';
        questionDiv.style.marginBottom = '2rem';
        questionDiv.style.padding = '1.5rem';
        questionDiv.style.background = '#f8f9fa';
        questionDiv.style.borderRadius = '8px';
        questionDiv.style.border = '1px solid #dee2e6';
        
        let html = `<h4 style="margin-bottom: 1rem; color: var(--primary);">📝 Question ${index + 1}</h4>`;
        html += `<p style="font-size: 1.1rem; margin-bottom: 1rem;">${q.question}</p>`;
        
        if (q.options && q.options.length > 0) {
            html += '<div style="display: flex; flex-direction: column; gap: 0.75rem;">';
            q.options.forEach(option => {
                const letter = option.charAt(0);
                html += `
                    <label class="quiz-option" style="display: flex !important; align-items: center; padding: 1rem; background: white; border: 2px solid #e0e0e0; border-radius: 6px; cursor: pointer; transition: all 0.2s;" onmouseover="this.style.borderColor='#007bff'" onmouseout="this.style.borderColor='#e0e0e0'">
                        <input type="radio" name="q${index}" value="${letter}" style="margin-right: 0.75rem; width: 20px; height: 20px;">
                        <span style="flex: 1;">${option}</span>
                    </label>
                `;
            });
            html += '</div>';
        } else {
            html += '<p style="color: orange;">No options available for this question</p>';
            console.warn(`Question ${index + 1} has no options`);
        }
        
        questionDiv.innerHTML = html;
        container.appendChild(questionDiv);
    });
    
    console.log('All questions rendered successfully');
}

// Submit quiz
submitQuizBtn.addEventListener('click', async () => {
    const answers = [];
    
    currentQuiz.questions.forEach((q, index) => {
        const selected = document.querySelector(`input[name="q${index}"]:checked`);
        answers.push(selected ? selected.value : '');
    });
    
    submitQuizBtn.textContent = 'Submitting...';
    submitQuizBtn.disabled = true;
    
    try {
        const response = await fetch('/api/quiz/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                topic: currentQuiz.topic,
                difficulty: currentQuiz.difficulty,
                questions: currentQuiz.questions,
                answers: answers
            })
        });
        
        const data = await response.json();
        displayResults(data);
        
        quizQuestions.style.display = 'none';
        quizResults.style.display = 'block';
        
        // Reload history to show updated results
        loadQuizHistory();
        
    } catch (error) {
        alert('Failed to submit quiz. Please try again.');
        console.error('Quiz submission error:', error);
    } finally {
        submitQuizBtn.textContent = 'Submit Quiz';
        submitQuizBtn.disabled = false;
    }
});

// Display results
function displayResults(data) {
    document.getElementById('scoreText').textContent = `${data.score}/${data.total}`;
    document.getElementById('percentageText').textContent = `${data.percentage.toFixed(1)}%`;
    
    const detailsDiv = document.getElementById('resultsDetails');
    detailsDiv.innerHTML = '';
    
    data.results.forEach((result, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'quiz-question';
        resultDiv.innerHTML = `
            <h4>Question ${index + 1}: ${result.question}</h4>
            <p><strong>Your Answer:</strong> ${result.user_answer || 'Not answered'}</p>
            <p><strong>Correct Answer:</strong> ${result.correct_answer}</p>
            <p style="color: ${result.is_correct ? 'var(--success)' : 'var(--danger)'}">
                ${result.is_correct ? '✓ Correct' : '✗ Incorrect'}
            </p>
        `;
        detailsDiv.appendChild(resultDiv);
    });
}

// New quiz
newQuizBtn.addEventListener('click', () => {
    quizResults.style.display = 'none';
    quizSetup.style.display = 'block';
    document.getElementById('quizTopic').value = '';
    document.getElementById('subtopicContainer').style.display = 'none';
    currentQuiz = null;
});

// Load quiz history
async function loadQuizHistory() {
    try {
        const response = await fetch('/api/quiz/history');
        const data = await response.json();
        
        const historyDiv = document.getElementById('quizHistory');
        if (!data.history || data.history.length === 0) {
            historyDiv.innerHTML = '<p>No quiz history yet</p>';
            return;
        }
        
        historyDiv.innerHTML = '<h3>📊 Quiz History (Click to view)</h3>';
        data.history.reverse().forEach(quiz => {
            const quizCard = document.createElement('div');
            quizCard.className = 'quiz-history-card';
            quizCard.style.cursor = 'pointer';
            // Check if quiz has data
            const hasQuizData = quiz.quiz_data && quiz.quiz_data.questions && quiz.quiz_data.questions.length > 0;
            
            console.log('Quiz entry:', quiz.topic, 'Has data:', hasQuizData, 'Quiz data:', quiz.quiz_data);
            
            quizCard.innerHTML = `
                <div class="quiz-history-header">
                    <strong>${quiz.topic}</strong>
                    <span class="quiz-score">${quiz.score}</span>
                </div>
                <div class="quiz-history-details">
                    <span>🎯 ${quiz.difficulty}</span>
                    <span>📅 ${new Date(quiz.timestamp).toLocaleDateString()}</span>
                    ${hasQuizData ? '<span style="color: #10b981;">✅ Click to view</span>' : '<span style="color: #ef4444;">❌ No data</span>'}
                </div>
            `;
            
            if (hasQuizData) {
                quizCard.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Quiz card clicked:', quiz);
                    viewQuizHistory(quiz);
                });
                quizCard.style.cursor = 'pointer';
                quizCard.style.pointerEvents = 'auto';
                
                // Add indicator for legacy entries
                if (quiz.quiz_data.legacy) {
                    const legacyBadge = document.createElement('span');
                    legacyBadge.textContent = ' (Historical)';
                    legacyBadge.style.fontSize = '0.8rem';
                    legacyBadge.style.color = '#999';
                    quizCard.querySelector('.quiz-history-header strong').appendChild(legacyBadge);
                }
            } else {
                quizCard.style.cursor = 'not-allowed';
                quizCard.style.opacity = '0.6';
                quizCard.style.pointerEvents = 'none';
                quizCard.title = 'Quiz details not available';
            }
            
            historyDiv.appendChild(quizCard);
        });
    } catch (error) {
        console.error('Failed to load quiz history:', error);
    }
}

function viewQuizHistory(quiz) {
    console.log('Viewing quiz history:', quiz);
    
    if (!quiz.quiz_data || !quiz.quiz_data.questions) {
        console.error('No quiz data:', quiz);
        alert('Quiz questions not available for this entry');
        return;
    }
    
    currentQuiz = {
        topic: quiz.topic,
        difficulty: quiz.difficulty,
        questions: quiz.quiz_data.questions
    };
    
    const container = document.getElementById('questionsContainer');
    container.innerHTML = '';
    
    // Show legacy warning if applicable
    if (quiz.quiz_data.legacy) {
        const warning = document.createElement('div');
        warning.style.padding = '1.5rem';
        warning.style.background = '#fff3cd';
        warning.style.border = '2px solid #ffc107';
        warning.style.borderRadius = '8px';
        warning.style.marginBottom = '1rem';
        warning.style.textAlign = 'center';
        warning.innerHTML = `
            <h3 style="margin-bottom: 1rem;">📜 Historical Quiz Entry</h3>
            <p style="margin-bottom: 0.5rem;"><strong>Topic:</strong> ${quiz.topic}</p>
            <p style="margin-bottom: 0.5rem;"><strong>Score:</strong> ${quiz.score}</p>
            <p style="margin-bottom: 0.5rem;"><strong>Date:</strong> ${new Date(quiz.timestamp).toLocaleString()}</p>
            <hr style="margin: 1rem 0; border: none; border-top: 1px solid #ffc107;">
            <p style="color: #856404;">⚠️ This quiz was taken before the system update. Only the score was saved.<br>Question details are not available for historical entries.</p>
        `;
        container.appendChild(warning);
        return;
    }
    
    displayQuestions(quiz.quiz_data.questions);
    
    // Show results if available
    setTimeout(() => {
        if (quiz.quiz_data.results) {
            const container = document.getElementById('questionsContainer');
            quiz.quiz_data.results.forEach((result, index) => {
                const questionDiv = container.children[index];
                if (questionDiv) {
                    const resultBadge = document.createElement('div');
                    resultBadge.style.display = 'block';
                    resultBadge.style.marginTop = '1rem';
                    resultBadge.style.padding = '1rem';
                    resultBadge.style.borderRadius = '6px';
                    resultBadge.style.background = result.is_correct ? '#d4edda' : '#f8d7da';
                    resultBadge.style.color = result.is_correct ? '#155724' : '#721c24';
                    resultBadge.style.border = result.is_correct ? '2px solid #28a745' : '2px solid #dc3545';
                    resultBadge.innerHTML = `
                        <strong>👉 Your answer:</strong> ${result.user_answer || 'Not answered'}<br>
                        <strong>✅ Correct answer:</strong> ${result.correct_answer}
                    `;
                    questionDiv.appendChild(resultBadge);
                }
            });
        }
    }, 100);
    
    quizSetup.style.display = 'none';
    quizQuestions.style.display = 'block';
    quizQuestions.style.visibility = 'visible';
    quizQuestions.style.opacity = '1';
    submitQuizBtn.style.display = 'none';
    
    // Scroll to top
    window.scrollTo(0, 0);
    
    console.log('Quiz history displayed');
}
