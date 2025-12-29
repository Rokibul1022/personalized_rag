// Load profile data
async function loadProfile() {
    try {
        const response = await fetch('/api/profile');
        const profile = await response.json();
        
        // Populate form
        document.getElementById('name').value = profile.name || '';
        document.getElementById('age').value = profile.age || '';
        document.getElementById('grade').value = profile.grade || '';
        document.getElementById('favorite_topics').value = profile.favorite_topics || '';
        document.getElementById('weak_topics').value = profile.weak_topics || '';
        document.getElementById('learning_style').value = profile.learning_style || 'general';
        document.getElementById('difficulty').value = profile.difficulty || 'medium';
        document.getElementById('goals').value = profile.goals || '';
        
    } catch (error) {
        console.error('Failed to load profile:', error);
    }
}

// Update profile
document.getElementById('profileForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = {
        name: document.getElementById('name').value,
        age: document.getElementById('age').value,
        grade: document.getElementById('grade').value,
        favorite_topics: document.getElementById('favorite_topics').value,
        weak_topics: document.getElementById('weak_topics').value,
        learning_style: document.getElementById('learning_style').value,
        difficulty: document.getElementById('difficulty').value,
        goals: document.getElementById('goals').value
    };
    
    try {
        const response = await fetch('/api/profile', {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            const successMsg = document.getElementById('successMessage');
            successMsg.style.display = 'block';
            setTimeout(() => {
                successMsg.style.display = 'none';
            }, 3000);
        }
        
    } catch (error) {
        alert('Failed to update profile. Please try again.');
        console.error('Profile update error:', error);
    }
});

// Load on page load
loadProfile();
