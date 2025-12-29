// Login
if (document.getElementById('loginForm')) {
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('username').value.trim();
        const errorDiv = document.getElementById('error-message');
        
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username})
            });
            
            const data = await response.json();
            
            if (data.success) {
                window.location.href = '/chat';
            } else {
                errorDiv.textContent = data.error;
                errorDiv.style.display = 'block';
            }
        } catch (error) {
            errorDiv.textContent = 'Login failed. Please try again.';
            errorDiv.style.display = 'block';
        }
    });
}

// Register
if (document.getElementById('registerForm')) {
    document.getElementById('registerForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = {
            username: document.getElementById('username').value.trim(),
            name: document.getElementById('name').value.trim(),
            age: document.getElementById('age').value,
            grade: document.getElementById('grade').value,
            favorite_topics: document.getElementById('favorite_topics').value,
            weak_topics: document.getElementById('weak_topics').value,
            learning_style: document.getElementById('learning_style').value,
            difficulty: document.getElementById('difficulty').value,
            goals: document.getElementById('goals').value
        };
        
        const errorDiv = document.getElementById('error-message');
        
        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Auto-login after registration
                const loginResponse = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({username: data.username})
                });
                
                if (loginResponse.ok) {
                    window.location.href = '/chat';
                }
            } else {
                errorDiv.textContent = data.error;
                errorDiv.style.display = 'block';
            }
        } catch (error) {
            errorDiv.textContent = 'Registration failed. Please try again.';
            errorDiv.style.display = 'block';
        }
    });
}
