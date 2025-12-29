// Store analytics data globally
let analyticsData = {};

async function loadDashboard() {
    try {
        const response = await fetch('/api/stats/dashboard');
        const data = await response.json();
        
        document.getElementById('totalQueries').textContent = data.total_queries || 0;
        document.getElementById('topicsExplored').textContent = data.topics_explored || 0;
        document.getElementById('quizzesTaken').textContent = data.quizzes_taken || 0;
        document.getElementById('avgScore').textContent = `${data.avg_quiz_score || 0}%`;
        
        displayRecentActivity(data.recent_activity || []);
        displayProgressChart(data);
        
        if (data.analytics) {
            analyticsData = data.analytics;
            displayPatternsPreview(data.analytics.learning_patterns);
            displaySchedulePreview(data.analytics.study_schedule);
            displayGapsPreview(data.analytics.knowledge_gaps);
            displayRecommendations(data.analytics.knowledge_gaps);
        }
        
        return data;
    } catch (error) {
        console.error('Failed to load dashboard:', error);
        return null;
    }
}

function displayRecentActivity(activities) {
    const container = document.getElementById('recentActivity');
    if (activities.length === 0) {
        container.innerHTML = '<p style="text-align:center;color:#999;padding:20px;">No activity yet</p>';
        return;
    }
    container.innerHTML = activities.slice(0, 5).map(a => `
        <div class="activity-item">
            <span>${a.type === 'quiz' ? '📝' : '💬'}</span>
            <div><strong>${a.topic}</strong><br><small>${a.timestamp.split(' ')[0]}</small></div>
        </div>
    `).join('');
}

function displayProgressChart(data) {
    const container = document.getElementById('progressChart');
    if (!data.quizzes_taken) {
        container.innerHTML = '<p style="text-align:center;color:#999;padding:40px;">Take quizzes to see progress</p>';
        return;
    }
    const avgScore = data.avg_quiz_score || 0;
    const level = avgScore >= 90 ? 'Expert' : avgScore >= 70 ? 'Intermediate' : avgScore >= 50 ? 'Beginner' : 'Novice';
    const color = avgScore >= 90 ? '#10b981' : avgScore >= 70 ? '#667eea' : avgScore >= 50 ? '#f59e0b' : '#ef4444';
    
    // Create topic distribution chart
    const topTopics = data.top_topics || [];
    const maxCount = Math.max(...topTopics.map(t => t.count), 1);
    const topicsChart = topTopics.slice(0, 5).map(t => {
        const percentage = (t.count / maxCount) * 100;
        return `
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:0.85rem;">
                    <span>${t.topic}</span>
                    <span style="font-weight:bold;color:#667eea;">${t.count}</span>
                </div>
                <div style="height:8px;background:#f0f0f0;border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:${percentage}%;background:linear-gradient(90deg,#667eea,#764ba2);transition:width 0.5s;"></div>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = `
        <div style="margin-bottom:25px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="font-weight:500;">Overall Performance</span>
                <span style="color:${color};font-weight:bold;">${level}</span>
            </div>
            <div style="height:50px;background:#f0f0f0;border-radius:25px;overflow:hidden;position:relative;">
                <div style="height:100%;width:${avgScore}%;background:${color};display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:1.2rem;transition:width 0.5s;">${avgScore.toFixed(1)}%</div>
            </div>
        </div>
        
        <div style="margin-bottom:25px;">
            <h4 style="margin:0 0 15px 0;font-size:0.95rem;color:#495057;">📊 Topic Distribution</h4>
            ${topicsChart || '<p style="text-align:center;color:#999;">No topics yet</p>'}
        </div>
        
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
            <div style="text-align:center;padding:15px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:12px;color:white;">
                <div style="font-size:2rem;font-weight:bold;">${data.quizzes_taken}</div>
                <div style="font-size:0.85rem;opacity:0.9;">Quizzes</div>
            </div>
            <div style="text-align:center;padding:15px;background:linear-gradient(135deg,#f093fb,#f5576c);border-radius:12px;color:white;">
                <div style="font-size:2rem;font-weight:bold;">${data.topics_explored}</div>
                <div style="font-size:0.85rem;opacity:0.9;">Topics</div>
            </div>
            <div style="text-align:center;padding:15px;background:linear-gradient(135deg,#4facfe,#00f2fe);border-radius:12px;color:white;">
                <div style="font-size:2rem;font-weight:bold;">${data.total_queries}</div>
                <div style="font-size:0.85rem;opacity:0.9;">Queries</div>
            </div>
        </div>
    `;
}

function displayPatternsPreview(patterns) {
    const container = document.getElementById('patternsPreview');
    if (!patterns || patterns.status === 'insufficient_data') {
        container.innerHTML = '<p style="text-align:center;color:#999;">Not enough data</p>';
        return;
    }
    const consistency = patterns.consistency_score;
    const color = consistency >= 70 ? '#22c55e' : consistency >= 40 ? '#f59e0b' : '#ef4444';
    container.innerHTML = `
        <div class="preview-row"><span>⏰ Peak:</span><strong>${patterns.peak_hours[0] || 'N/A'}</strong></div>
        <div class="preview-row"><span>🎯 Consistency:</span><strong style="color:${color}">${consistency}%</strong></div>
        <div class="preview-row"><span>⚡ Velocity:</span><strong>${patterns.learning_velocity}/day</strong></div>
    `;
}

function displaySchedulePreview(schedule) {
    const container = document.getElementById('schedulePreview');
    if (!schedule || !schedule.schedule || schedule.schedule.length === 0) {
        container.innerHTML = '<p style="text-align:center;color:#999;">No schedule</p>';
        return;
    }
    container.innerHTML = schedule.schedule.slice(0, 3).map(s => `
        <div class="preview-row"><strong>${s.day}</strong> ${s.time} - ${s.topic}</div>
    `).join('');
}

function displayGapsPreview(gaps) {
    const container = document.getElementById('gapsPreview');
    if (!gaps || gaps.status === 'insufficient_data') {
        container.innerHTML = '<p style="text-align:center;color:#999;">Not enough data</p>';
        return;
    }
    const weakCount = gaps.weak_areas?.length || 0;
    const unexploredCount = gaps.unexplored_topics?.length || 0;
    container.innerHTML = `
        <div class="preview-row"><span>⚠️ Weak:</span><strong style="color:#ef4444;">${weakCount}</strong></div>
        <div class="preview-row"><span>🌟 Unexplored:</span><strong style="color:#f59e0b;">${unexploredCount}</strong></div>
    `;
}

function displayRecommendations(gaps) {
    const container = document.getElementById('recommendations');
    if (!gaps || !gaps.recommendations) {
        container.innerHTML = '<p style="text-align:center;color:#999;">No recommendations</p>';
        return;
    }
    container.innerHTML = gaps.recommendations.slice(0, 3).map(r => `
        <div class="rec-item">💡 ${r}</div>
    `).join('');
}

function showPatternsModal() {
    const patterns = analyticsData.learning_patterns;
    if (!patterns || patterns.status === 'insufficient_data') {
        alert('Not enough data to show learning patterns');
        return;
    }
    document.getElementById('peakHours').innerHTML = patterns.peak_hours.map(h => `<span class="badge">${h}</span>`).join(' ');
    document.getElementById('activeDays').innerHTML = patterns.active_days.map(d => `<span class="badge">${d}</span>`).join(' ');
    document.getElementById('velocity').innerHTML = `<strong>${patterns.learning_velocity}</strong> queries per day`;
    const consistency = patterns.consistency_score;
    const color = consistency >= 70 ? '#22c55e' : consistency >= 40 ? '#f59e0b' : '#ef4444';
    document.getElementById('consistency').innerHTML = `<strong style="color:${color};font-size:2rem;">${consistency}%</strong>`;
    document.getElementById('nextSession').innerHTML = `<strong>${patterns.next_predicted_session}</strong>`;
    document.getElementById('patternsModal').style.display = 'flex';
}

function showScheduleModal() {
    const schedule = analyticsData.study_schedule;
    if (!schedule || !schedule.schedule || schedule.schedule.length === 0) {
        alert('No study schedule available');
        return;
    }
    const container = document.getElementById('scheduleDetails');
    container.innerHTML = `
        <div style="background:#e8f4f8;padding:15px;border-radius:10px;margin-bottom:20px;">
            <p><strong>Session Length:</strong> ${schedule.recommended_session_length}</p>
            <p><strong>Break Frequency:</strong> ${schedule.break_frequency}</p>
            <p><strong>Weekly Goal:</strong> ${schedule.weekly_goal}</p>
        </div>
        <div style="display:grid;gap:15px;">
            ${schedule.schedule.map(s => `
                <div style="background:white;padding:15px;border-radius:10px;border-left:4px solid ${s.priority === 'high' ? '#ef4444' : '#f59e0b'};box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                    <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                        <strong>${s.day}</strong>
                        <span>${s.time}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>${s.topic}</div>
                        <div style="font-size:0.85rem;color:#6c757d;">${s.duration}</div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    document.getElementById('scheduleModal').style.display = 'flex';
}

function showGapsModal() {
    const gaps = analyticsData.knowledge_gaps;
    if (!gaps || gaps.status === 'insufficient_data') {
        alert('Not enough data to analyze knowledge gaps');
        return;
    }
    document.getElementById('weakAreas').innerHTML = gaps.weak_areas.length > 0
        ? gaps.weak_areas.map(w => `<span class="gap-badge weak">${w}</span>`).join('')
        : '<p>No weak areas identified</p>';
    document.getElementById('repeatedQuestions').innerHTML = gaps.repeated_questions.length > 0
        ? gaps.repeated_questions.map(q => `<span class="gap-badge">${q}</span>`).join('')
        : '<p>No repeated questions</p>';
    document.getElementById('unexploredTopics').innerHTML = gaps.unexplored_topics.length > 0
        ? gaps.unexplored_topics.map(t => `<span class="gap-badge unexplored">${t}</span>`).join('')
        : '<p>All topics explored!</p>';
    document.getElementById('gapsModal').style.display = 'flex';
}

function showTopicsModal() {
    document.getElementById('topicsModal').style.display = 'flex';
    fetch('/api/stats/dashboard').then(r => r.json()).then(data => {
        const topics = data.top_topics || [];
        document.getElementById('allTopics').innerHTML = topics.map(t => `
            <div style="padding:20px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:12px;text-align:center;cursor:pointer;transition:all 0.3s;" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
                <h4 style="margin:0 0 10px 0;">${t.topic}</h4>
                <p style="margin:0;opacity:0.9;">${t.count} interactions</p>
            </div>
        `).join('');
    });
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

window.onclick = function(event) {
    if (event.target.classList.contains('modal')) {
        event.target.style.display = 'none';
    }
}

document.getElementById('refreshBtn')?.addEventListener('click', loadDashboard);
loadDashboard();
