function toggleLogin() {
    const loginForm = document.getElementById('loginForm');
    const homeContent = document.getElementById('homeContent');
    const loginButton = document.querySelector('nav button');
    
    if (loginForm.classList.contains('hidden')) {
        loginForm.classList.remove('hidden');
        homeContent.classList.add('hidden');
        loginButton.textContent = 'Close';
    } else {
        loginForm.classList.add('hidden');
        homeContent.classList.remove('hidden');
        loginButton.textContent = 'Login';
    }
}

document.getElementById('loginForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const errorMessage = document.getElementById('errorMessage');

    if (username === 'farmer' && password === 'password') {
        errorMessage.textContent = '';
        alert('Login successful! Welcome to FarmWater Advisor.');
        // Here you would typically redirect to a dashboard
        // Redirect to the next page
        window.location.href = "water-calculator.html";
    } else {
        errorMessage.textContent = 'Invalid username or password';
    }
});