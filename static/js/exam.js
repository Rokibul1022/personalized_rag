let currentExamId = null;

// Load exams on page load
document.addEventListener('DOMContentLoaded', () => {
    loadExams();
    
    document.getElementById('addExamBtn').addEventListener('click', addExam);
    document.getElementById('backToListBtn').addEventListener('click', () => {
        document.getElementById('studyPlanView').style.display = 'none';
        document.getElementById('examsList').style.display = 'block';
        document.getElementById('addExamForm').style.display = 'block';
    });
    document.getElementById('generatePlanBtn').addEventListener('click', generatePlan);
    document.getElementById('downloadRoadmapBtn').addEventListener('click', downloadRoadmap);
    document.getElementById('markProgressBtn').addEventListener('click', markProgress);
    document.getElementById('deleteExamBtn').addEventListener('click', deleteExam);
});

async function loadExams() {
    try {
        const response = await fetch('/api/exam/list');
        const data = await response.json();
        
        const container = document.getElementById('examsContainer');
        
        if (!data.exams || data.exams.length === 0) {
            container.innerHTML = '<p class="no-data">No exams registered yet. Add your first exam above!</p>';
            return;
        }
        
        container.innerHTML = data.exams.map(exam => `
            <div class="exam-card" onclick="viewExam(${exam.id})">
                <h4>${exam.name}</h4>
                <p>📅 ${new Date(exam.date).toLocaleDateString()}</p>
                <p>⏰ ${exam.days_until} days ${exam.is_past ? '(Past)' : ''}</p>
                <p>📖 ${exam.topics.join(', ')}</p>
                <p>🎯 ${exam.difficulty}</p>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading exams:', error);
        document.getElementById('examsContainer').innerHTML = '<p class="error">Failed to load exams</p>';
    }
}

async function addExam() {
    const name = document.getElementById('examName').value.trim();
    const date = document.getElementById('examDate').value;
    const topics = document.getElementById('examTopics').value.split(',').map(t => t.trim()).filter(t => t);
    const difficulty = document.getElementById('examDifficulty').value;
    
    if (!name || !date || topics.length === 0) {
        alert('Please fill all fields');
        return;
    }
    
    try {
        const response = await fetch('/api/exam/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ exam_name: name, exam_date: date, topics, difficulty })
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('examName').value = '';
            document.getElementById('examDate').value = '';
            document.getElementById('examTopics').value = '';
            loadExams();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error adding exam:', error);
        alert('Failed to add exam');
    }
}

async function viewExam(examId) {
    currentExamId = examId;
    
    try {
        const response = await fetch(`/api/exam/${examId}/plan`);
        const data = await response.json();
        
        if (!data.success) {
            alert('Error loading exam');
            return;
        }
        
        document.getElementById('planExamName').textContent = data.exam_name;
        document.getElementById('planExamDate').textContent = new Date(data.exam_date).toLocaleDateString();
        document.getElementById('planDaysUntil').textContent = data.exam.days_until || 0;
        document.getElementById('planTopics').textContent = data.topics.join(', ');
        document.getElementById('planDifficulty').textContent = data.difficulty;
        
        if (data.study_plan) {
            document.getElementById('studyPlanContent').style.display = 'block';
            document.getElementById('planText').innerHTML = data.study_plan.replace(/\n/g, '<br>');
        } else {
            document.getElementById('studyPlanContent').style.display = 'none';
        }
        
        document.getElementById('examsList').style.display = 'none';
        document.getElementById('addExamForm').style.display = 'none';
        document.getElementById('studyPlanView').style.display = 'block';
    } catch (error) {
        console.error('Error viewing exam:', error);
        alert('Failed to load exam details');
    }
}

async function generatePlan() {
    document.getElementById('planLoadingMsg').style.display = 'block';
    
    try {
        const response = await fetch(`/api/exam/${currentExamId}/plan`, { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('studyPlanContent').style.display = 'block';
            document.getElementById('planText').innerHTML = data.study_plan.replace(/\n/g, '<br>');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error generating plan:', error);
        alert('Failed to generate study plan');
    } finally {
        document.getElementById('planLoadingMsg').style.display = 'none';
    }
}

function downloadRoadmap() {
    const planText = document.getElementById('planText').innerText;
    const examName = document.getElementById('planExamName').textContent;
    
    const blob = new Blob([planText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${examName}_study_plan.txt`;
    a.click();
}

async function markProgress() {
    const day = prompt('Which day did you complete? (Enter day number)');
    if (!day) return;
    
    const notes = prompt('Any notes about your progress?') || 'Completed planned tasks';
    
    try {
        const response = await fetch(`/api/exam/${currentExamId}/progress`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ day: parseInt(day), notes })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert('Progress marked successfully!');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error marking progress:', error);
        alert('Failed to mark progress');
    }
}

async function deleteExam() {
    if (!confirm('Are you sure you want to delete this exam?')) return;
    
    try {
        const response = await fetch(`/api/exam/${currentExamId}/delete`, { method: 'DELETE' });
        const data = await response.json();
        
        if (data.success) {
            alert('Exam deleted successfully');
            document.getElementById('studyPlanView').style.display = 'none';
            document.getElementById('examsList').style.display = 'block';
            document.getElementById('addExamForm').style.display = 'block';
            loadExams();
        } else {
            alert('Error deleting exam');
        }
    } catch (error) {
        console.error('Error deleting exam:', error);
        alert('Failed to delete exam');
    }
}
