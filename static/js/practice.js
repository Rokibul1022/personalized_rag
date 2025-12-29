// Load practice history
async function loadHistory() {
    try {
        const response = await fetch('/api/practice/history');
        const data = await response.json();
        
        const container = document.getElementById('historyContainer');
        
        if (data.history && data.history.length > 0) {
            container.innerHTML = '';
            data.history.reverse().forEach((item, idx) => {
                const div = document.createElement('div');
                div.style.cssText = 'padding: 1rem; border-bottom: 1px solid #ddd; cursor: pointer;';
                div.innerHTML = `
                    <strong>${item.topic}</strong> - ${item.difficulty} 
                    <span style="color: #666; font-size: 0.9em;">(${item.problems.length} problems)</span>
                    <br><small>${new Date(item.timestamp).toLocaleString()}</small>
                `;
                div.onclick = () => showHistoryProblems(item);
                container.appendChild(div);
            });
        } else {
            container.innerHTML = '<p>No practice history yet.</p>';
        }
    } catch (error) {
        document.getElementById('historyContainer').innerHTML = '<p>Error loading history.</p>';
    }
}

function showHistoryProblems(item) {
    const container = document.getElementById('problemsContainer');
    container.innerHTML = '';
    
    item.problems.forEach((problem, index) => {
        const card = document.createElement('div');
        card.className = 'card';
        const questionText = problem.problem || problem.question || 'No question available';
        const answerText = problem.solution || problem.answer || 'No answer available';
        
        card.innerHTML = `
            <h3>Problem ${index + 1}</h3>
            <div style="margin-bottom: 1rem;">
                <strong>Concept:</strong> ${problem.concept || item.topic}
            </div>
            <div style="margin-bottom: 1rem;">
                <strong style="display: block; margin-bottom: 0.5rem; color: #2563eb;">Question:</strong>
                <div style="background: #f5f5f5; padding: 1rem; border-radius: 4px; white-space: pre-wrap;">${questionText}</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <strong style="display: block; margin-bottom: 0.5rem;">Your Answer:</strong>
                <textarea id="user-answer-${index}" placeholder="Type your answer here..." style="width: 100%; min-height: 80px; padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; font-family: inherit;"></textarea>
            </div>
            <button class="btn btn-primary btn-small" onclick="checkAnswer(${index})" style="margin-right: 0.5rem;">
                Check Answer
            </button>
            <button class="btn btn-secondary btn-small" onclick="toggleSolution(${index})">
                Show Answer
            </button>
            <div id="feedback-${index}" style="display: none; margin-top: 1rem; padding: 1rem; border-radius: 4px;"></div>
            <div id="solution-${index}" style="display: none; margin-top: 1rem;">
                <strong style="display: block; margin-bottom: 0.5rem; color: #16a34a;">Correct Answer:</strong>
                <div style="padding: 1rem; background: #e8f5e9; border-radius: 4px; white-space: pre-wrap;">${answerText}</div>
            </div>
        `;
        container.appendChild(card);
    });
    
    window.scrollTo({top: 0, behavior: 'smooth'});
}

loadHistory();

document.getElementById('practiceForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const topic = document.getElementById('topicInput').value;
    const difficulty = document.getElementById('difficultySelect').value;
    const count = parseInt(document.getElementById('countSelect').value);
    
    const container = document.getElementById('problemsContainer');
    container.innerHTML = '<div class="card"><p>Generating problems...</p></div>';
    
    try {
        const response = await fetch('/api/practice/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({topic, difficulty, count})
        });
        
        const data = await response.json();
        
        if (data.problems && data.problems.length > 0) {
            container.innerHTML = '';
            
            data.problems.forEach((problem, index) => {
                const card = document.createElement('div');
                card.className = 'card';
                const questionText = problem.problem || problem.question || 'No question available';
                const answerText = problem.solution || problem.answer || 'No answer available';
                
                card.innerHTML = `
                    <h3>Problem ${index + 1}</h3>
                    <div style="margin-bottom: 1rem;">
                        <strong>Concept:</strong> ${problem.concept || topic}
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <strong style="display: block; margin-bottom: 0.5rem; color: #2563eb;">Question:</strong>
                        <div style="background: #f5f5f5; padding: 1rem; border-radius: 4px; white-space: pre-wrap;">${questionText}</div>
                    </div>
                    <div style="margin-bottom: 1rem;">
                        <strong style="display: block; margin-bottom: 0.5rem;">Your Answer:</strong>
                        <textarea id="user-answer-${index}" placeholder="Type your answer here..." style="width: 100%; min-height: 80px; padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; font-family: inherit;"></textarea>
                    </div>
                    <button class="btn btn-primary btn-small" onclick="checkAnswer(${index})" style="margin-right: 0.5rem;">
                        Check Answer
                    </button>
                    <button class="btn btn-secondary btn-small" onclick="toggleSolution(${index})">
                        Show Answer
                    </button>
                    <div id="feedback-${index}" style="display: none; margin-top: 1rem; padding: 1rem; border-radius: 4px;"></div>
                    <div id="solution-${index}" style="display: none; margin-top: 1rem;">
                        <strong style="display: block; margin-bottom: 0.5rem; color: #16a34a;">Correct Answer:</strong>
                        <div style="padding: 1rem; background: #e8f5e9; border-radius: 4px; white-space: pre-wrap;">${answerText}</div>
                    </div>
                `;
                container.appendChild(card);
            });
            loadHistory();
        } else {
            container.innerHTML = '<div class="card"><p>No problems generated. Try again.</p></div>';
        }
    } catch (error) {
        container.innerHTML = `<div class="card"><p style="color: red;">Error: ${error.message}</p></div>`;
    }
});

function checkAnswer(index) {
    const userAnswer = document.getElementById(`user-answer-${index}`).value.trim();
    const feedback = document.getElementById(`feedback-${index}`);
    
    if (!userAnswer) {
        feedback.style.display = 'block';
        feedback.style.background = '#fef3c7';
        feedback.style.color = '#92400e';
        feedback.innerHTML = '⚠️ Please write your answer first!';
        return;
    }
    
    feedback.style.display = 'block';
    feedback.style.background = '#dbeafe';
    feedback.style.color = '#1e40af';
    feedback.innerHTML = `
        <strong>Your Answer:</strong><br>
        <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px;">${userAnswer}</div>
        <em>Click "Show Answer" to compare with the correct solution.</em>
    `;
}

function toggleSolution(index) {
    const solution = document.getElementById(`solution-${index}`);
    const button = event.target;
    
    if (solution.style.display === 'none') {
        solution.style.display = 'block';
        button.textContent = 'Hide Answer';
    } else {
        solution.style.display = 'none';
        button.textContent = 'Show Answer';
    }
}
