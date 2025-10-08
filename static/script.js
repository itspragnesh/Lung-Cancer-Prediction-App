// Form validation
function validateForm() {
    const age = document.getElementById('age').value;
    if (age <= 0 || age > 120) {
        alert('Age must be between 1 and 120');
        return false;
    }
    // Check if all selects are filled
    const selects = document.querySelectorAll('select[required]');
    for (let select of selects) {
        if (!select.value) {
            alert('Please fill in all fields');
            select.focus();
            return false;
        }
    }
    // Add a loading state to the button
    const button = document.querySelector('button[type="submit"]');
    button.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Predicting...';
    button.disabled = true;
    return true;
}

// Auto-focus on first field on page load
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('gender').focus();
    
    // Add smooth scrolling for any potential anchors (future-proof)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Reset form on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            document.querySelector('form').reset();
            document.getElementById('gender').focus();
            alert('Form reset!');
        }
    });
    
    // Re-enable button if page is reloaded with form data (e.g., validation error)
    const button = document.querySelector('button[type="submit"]');
    button.innerHTML = 'Predict';
    button.disabled = false;
});

// Optional: Add real-time age validation
document.getElementById('age').addEventListener('input', function() {
    const age = parseInt(this.value);
    if (age < 1 || age > 120 || isNaN(age)) {
        this.classList.add('is-invalid');
        this.classList.remove('is-valid');
    } else {
        this.classList.add('is-valid');
        this.classList.remove('is-invalid');
    }
});