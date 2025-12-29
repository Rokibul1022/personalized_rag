// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
        // Update buttons
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`${tab}Tab`).classList.add('active');
        
        // Load data
        if (tab === 'users') loadUsers();
        if (tab === 'groups') loadGroups();
    });
});

// Load users
async function loadUsers() {
    try {
        const response = await fetch('/api/collab/users');
        const data = await response.json();
        
        const container = document.getElementById('usersList');
        container.innerHTML = '';
        
        if (data.users.length === 0) {
            container.innerHTML = '<p class="loading">No other users found</p>';
            return;
        }
        
        data.users.forEach(user => {
            const div = document.createElement('div');
            div.className = 'user-card';
            div.innerHTML = `
                <h3>${user.profile.name || user.username}</h3>
                <p><strong>Grade:</strong> ${user.profile.grade || 'N/A'}</p>
                <p><strong>Interests:</strong> ${user.profile.favorite_topics || 'N/A'}</p>
                <p><strong>Stats:</strong> ${user.stats.topics_explored} topics, ${user.stats.avg_quiz_score}% avg score</p>
            `;
            container.appendChild(div);
        });
        
    } catch (error) {
        console.error('Failed to load users:', error);
    }
}

// Load groups
async function loadGroups() {
    try {
        const response = await fetch('/api/collab/groups');
        const data = await response.json();
        
        const container = document.getElementById('groupsList');
        container.innerHTML = '';
        
        if (data.groups.length === 0) {
            container.innerHTML = '<p class="loading">No study groups yet</p>';
            return;
        }
        
        data.groups.forEach(group => {
            const div = document.createElement('div');
            div.className = 'group-card';
            div.innerHTML = `
                <h3>${group.topic}</h3>
                <p>${group.description || 'No description'}</p>
                <p><strong>Members:</strong> ${group.members.join(', ')}</p>
                <p><strong>Created:</strong> ${new Date(group.created_at).toLocaleDateString()}</p>
                <button class="btn btn-primary btn-sm open-chat-btn" data-group-id="${group.id}" data-group-name="${group.topic}">💬 Open Chat</button>
            `;
            container.appendChild(div);
            
            const chatBtn = div.querySelector('.open-chat-btn');
            chatBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                openGroupChat(group.id, group.topic);
            });
        });
        
    } catch (error) {
        console.error('Failed to load groups:', error);
    }
}

// Create group modal
const modal = document.getElementById('createGroupModal');
const createGroupBtn = document.getElementById('createGroupBtn');
const closeBtn = document.querySelector('.close');

createGroupBtn.addEventListener('click', async () => {
    await loadMembersForSelection();
    modal.style.display = 'flex';
});

closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
});

window.addEventListener('click', (e) => {
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

async function loadMembersForSelection() {
    try {
        const response = await fetch('/api/collab/users');
        const data = await response.json();
        
        const container = document.getElementById('membersList');
        const currentUser = document.getElementById('username-display').textContent;
        
        container.innerHTML = data.users.map(user => `
            <label class="member-checkbox">
                <input type="checkbox" name="member" value="${user.username}" ${user.username === currentUser ? 'checked disabled' : ''}>
                <span>${user.profile.name || user.username} ${user.username === currentUser ? '(You)' : ''}</span>
            </label>
        `).join('');
    } catch (error) {
        console.error('Failed to load members:', error);
    }
}

// Create group
document.getElementById('createGroupForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const topic = document.getElementById('groupTopic').value;
    const description = document.getElementById('groupDescription').value;
    
    const checkboxes = document.querySelectorAll('input[name="member"]:checked');
    const members = Array.from(checkboxes).map(cb => cb.value);
    
    if (members.length === 0) {
        alert('Please select at least one member');
        return;
    }
    
    try {
        const response = await fetch('/api/collab/groups', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({topic, description, members})
        });
        
        const data = await response.json();
        
        if (data.success) {
            modal.style.display = 'none';
            document.getElementById('createGroupForm').reset();
            loadGroups();
        }
        
    } catch (error) {
        alert('Failed to create group. Please try again.');
        console.error('Group creation error:', error);
    }
});

// Group chat functionality
let currentGroupId = null;
let messageInterval = null;

function openGroupChat(groupId, groupName) {
    currentGroupId = groupId;
    document.getElementById('chatGroupName').textContent = groupName;
    document.getElementById('groupsTab').style.display = 'none';
    document.getElementById('groupChatView').style.display = 'block';
    
    loadMessages();
    messageInterval = setInterval(loadMessages, 3000);
}

document.getElementById('backToGroupsBtn').addEventListener('click', () => {
    clearInterval(messageInterval);
    document.getElementById('groupChatView').style.display = 'none';
    document.getElementById('groupsTab').style.display = 'block';
});

async function loadMessages() {
    if (!currentGroupId) return;
    
    try {
        const response = await fetch(`/api/collab/groups/${currentGroupId}/messages`);
        const data = await response.json();
        
        const container = document.getElementById('chatMessages');
        const currentUser = document.getElementById('username-display').textContent;
        
        container.innerHTML = data.messages.map(msg => `
            <div class="message ${msg.username === currentUser ? 'message-own' : 'message-other'}">
                <div class="message-header">
                    <strong>${msg.username}</strong>
                    <span class="message-time">${new Date(msg.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="message-text">${msg.message}</div>
            </div>
        `).join('');
        
        container.scrollTop = container.scrollHeight;
    } catch (error) {
        console.error('Failed to load messages:', error);
    }
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message || !currentGroupId) return;
    
    try {
        const response = await fetch(`/api/collab/groups/${currentGroupId}/messages`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message})
        });
        
        const data = await response.json();
        
        if (data.success) {
            input.value = '';
            loadMessages();
        }
    } catch (error) {
        console.error('Failed to send message:', error);
    }
}

document.getElementById('sendMessageBtn').addEventListener('click', sendMessage);
document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Load users on page load
loadUsers();
