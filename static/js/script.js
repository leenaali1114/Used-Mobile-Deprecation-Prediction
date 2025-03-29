document.addEventListener('DOMContentLoaded', function() {
    // Set default values for form fields
    const today = new Date();
    const currentYear = today.getFullYear();
    
    // Set default release year to current year
    const releaseYearSelect = document.getElementById('release_year');
    if (releaseYearSelect) {
        for (let i = 0; i < releaseYearSelect.options.length; i++) {
            if (releaseYearSelect.options[i].value == currentYear) {
                releaseYearSelect.selectedIndex = i;
                break;
            }
        }
    }
    
    // Set default days used to 30
    const daysUsedInput = document.getElementById('days_used');
    if (daysUsedInput) {
        daysUsedInput.value = 30;
    }
    
    // Set default screen size
    const screenSizeInput = document.getElementById('screen_size');
    if (screenSizeInput) {
        screenSizeInput.value = 15.0;
    }
    
    // Set default camera values
    const rearCameraInput = document.getElementById('rear_camera_mp');
    const frontCameraInput = document.getElementById('front_camera_mp');
    if (rearCameraInput) rearCameraInput.value = 12.0;
    if (frontCameraInput) frontCameraInput.value = 8.0;
    
    // Set default battery and weight
    const batteryInput = document.getElementById('battery');
    const weightInput = document.getElementById('weight');
    if (batteryInput) batteryInput.value = 4000;
    if (weightInput) weightInput.value = 180;
    
    // Add animation to result boxes
    const resultBoxes = document.querySelectorAll('.result-box');
    resultBoxes.forEach(box => {
        box.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#e9ecef';
        });
        
        box.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '#f8f9fa';
        });
    });
    
    // Form validation
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            const inputs = form.querySelectorAll('input[type="number"]');
            let isValid = true;
            
            inputs.forEach(input => {
                const min = parseFloat(input.getAttribute('min'));
                const max = parseFloat(input.getAttribute('max'));
                const value = parseFloat(input.value);
                
                if (isNaN(value) || value < min || value > max) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!isValid) {
                event.preventDefault();
                alert('Please check the form for errors. All values must be within the specified ranges.');
            }
        });
    }
}); 