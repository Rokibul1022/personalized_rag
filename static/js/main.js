// Check session on page load
async function checkSession() {
    try {
        const response = await fetch('/api/auth/session');
        const data = await response.json();
        
        if (!data.logged_in) {
            window.location.href = '/login';
            return null;
        }
        
        // Update username display
        const usernameDisplay = document.getElementById('username-display');
        if (usernameDisplay) {
            usernameDisplay.textContent = data.profile.name || data.username;
        }
        
        return data;
    } catch (error) {
        console.error('Session check failed:', error);
        window.location.href = '/login';
        return null;
    }
}

// Logout
if (document.getElementById('logoutBtn')) {
    document.getElementById('logoutBtn').addEventListener('click', async () => {
        try {
            await fetch('/api/auth/logout', {method: 'POST'});
            window.location.href = '/';
        } catch (error) {
            console.error('Logout failed:', error);
        }
    });
}

// Dark mode toggle
async function loadTheme() {
    try {
        const response = await fetch('/api/theme');
        const data = await response.json();
        if (data.dark_mode) {
            document.body.classList.add('dark-mode');
            const icon = document.getElementById('themeIcon');
            if (icon) icon.textContent = '☀️';
        }
    } catch (error) {
        console.error('Theme load failed:', error);
    }
}

if (document.getElementById('darkModeToggle')) {
    document.getElementById('darkModeToggle').addEventListener('click', async () => {
        try {
            const response = await fetch('/api/theme/toggle', {method: 'POST'});
            const data = await response.json();
            
            if (data.dark_mode) {
                document.body.classList.add('dark-mode');
                document.getElementById('themeIcon').textContent = '☀️';
            } else {
                document.body.classList.remove('dark-mode');
                document.getElementById('themeIcon').textContent = '🌙';
            }
        } catch (error) {
            console.error('Theme toggle failed:', error);
        }
    });
}

// Initialize session check on protected pages
if (window.location.pathname !== '/' && 
    window.location.pathname !== '/login' && 
    window.location.pathname !== '/register') {
    checkSession();
    loadTheme();
}
