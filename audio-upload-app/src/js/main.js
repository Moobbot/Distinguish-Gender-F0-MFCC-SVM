document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const audioPlayer = document.getElementById('audioPlayer');
    const fileInput = document.getElementById('audioFile');

    // Handle file selection for audio preview
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            audioPlayer.src = url;
            audioPlayer.classList.remove('d-none');
        } else {
            audioPlayer.src = '';
            audioPlayer.classList.add('d-none');
        }
    });

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const file = fileInput.files[0];
        
        if (!file) {
            showMessage('Please select an audio file', 'error');
            return;
        }

        // Show loading
        loadingDiv.classList.remove('d-none');
        resultDiv.classList.add('d-none');

        // Simulate processing (replace this with actual API call)
        // Create FormData to send the file
        const formData = new FormData();
        formData.append('audio', file);

        // Make API call to your backend
        fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingDiv.classList.add('d-none');
            displayResult(data);
        })
        .catch(error => {
            loadingDiv.classList.add('d-none');
            showMessage('Error processing the audio file: ' + error.message, 'error');
        });
    });

    function displayResult(data) {
        resultDiv.classList.remove('d-none');
        if (data.success) {
            document.getElementById('svm-prediction').textContent = `SVM Prediction: ${data.svm_prediction}`;
            document.getElementById('rf-prediction').textContent = `Random Forest Prediction: ${data.rf_prediction}`;
            document.getElementById('final-prediction').textContent = `Final Prediction: ${data.final_prediction}`;
            document.getElementById('confidence').textContent = `Confidence: ${data.confidence}%`;
        } else {
            showMessage(data.error || 'Unknown error occurred', 'error');
        }
    }

    function showMessage(message, type = 'info') {
        resultDiv.classList.remove('d-none');
        const alertClass = type === 'error' ? 'alert-danger' : 'alert-info';
        resultDiv.innerHTML = `
            <div class="alert ${alertClass}" role="alert">
                ${message}
            </div>
        `;
    }
});