document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const fileInput = document.getElementById('audioFile');
        const file = fileInput.files[0];
        
        if (!file) {
            showMessage('Please select an audio file', 'error');
            return;
        }

        // Show loading
        loadingDiv.style.display = 'block';
        resultDiv.style.display = 'none';

        // Simulate processing (replace this with actual API call)
        setTimeout(() => {
            loadingDiv.style.display = 'none';
            
            // Demo result (replace this with actual API response)
            const demoResult = {
                success: true,
                prediction: {
                    gender: 'Male',
                    confidence: 95.5
                }
            };
            
            displayResult(demoResult);
        }, 2000);
    });

    function displayResult(data) {
        resultDiv.style.display = 'block';
        if (data.success) {
            resultDiv.innerHTML = `
                <h2>Prediction Result</h2>
                <div class="prediction-details">
                    <p>Predicted Gender: ${data.prediction.gender}</p>
                    <p>Confidence: ${data.prediction.confidence}%</p>
                </div>
            `;
        } else {
            showMessage(data.error || 'Unknown error occurred', 'error');
        }
    }

    function showMessage(message, type = 'info') {
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `
            <div class="message ${type}">
                <p>${message}</p>
            </div>
        `;
    }
});