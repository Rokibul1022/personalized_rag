// Force show quiz questions
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        #quizQuestions { display: block !important; }
        #questionsContainer { display: block !important; visibility: visible !important; opacity: 1 !important; }
        .quiz-question { display: block !important; visibility: visible !important; }
        .quiz-option { display: flex !important; visibility: visible !important; }
    `;
    document.head.appendChild(style);
});
