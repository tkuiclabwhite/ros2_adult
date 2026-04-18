// Get all sliders
var sliders = document.querySelectorAll('input[type="range"]');

// Add input event listener to each slider
sliders.forEach(function(slider) {
    slider.addEventListener('input', function() {
        // Get the corresponding value span
        var valueSpan;
        if (slider.id.endsWith('2')) {
            // If the slider id ends with '2', use the new id relationship
            valueSpan = document.querySelector('#camera-' + slider.id.slice(0, -1) + '-value2');
        } else {
            // Otherwise, use the original id relationship
            valueSpan = document.querySelector('#camera-' + slider.id + '-value');
        }
        
        // Update the value span's text
        valueSpan.textContent = slider.value;
    });
});