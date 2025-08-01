document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const audioPlayer = document.getElementById('audioPlayer');
    const fileInput = document.getElementById('audioFile');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const submitButton = document.getElementById('submitButton');
    const recordingTimer = document.getElementById('recordingTimer');
    const timerDisplay = document.getElementById('timer');

    let mediaRecorder;
    let audioChunks = [];
    let recordingStartTime;
    let timerInterval;
    let recordedBlob;

    // Handle file selection for audio preview
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            audioPlayer.src = url;
            audioPlayer.classList.remove('d-none');
            submitButton.disabled = false;
            recordButton.disabled = false;
            stopButton.disabled = true;
        } else {
            audioPlayer.src = '';
            audioPlayer.classList.add('d-none');
            submitButton.disabled = !recordedBlob;
        }
    });

    // Recording functionality
    recordButton.addEventListener('click', startRecording);
    stopButton.addEventListener('click', stopRecording);

    function clearFileInput() {
        fileInput.value = '';
        if (fileInput.value) {
            fileInput.type = 'text';
            fileInput.type = 'file';
        }
    }

    function startRecording() {
        // Clear file input and recorded blob when starting to record
        clearFileInput();
        recordedBlob = null;
        
        navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1,
                sampleRate: 44100
            }
        })
        .then(stream => {
            // Create AudioContext
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            
            // Create Recorder
            const recorder = new Recorder(source, {
                numChannels: 1,
                sampleRate: 44100
            });
            
            // Start recording
            recorder.record();
            mediaRecorder = recorder;

            // Update UI
            recordButton.disabled = true;
            stopButton.disabled = false;
            fileInput.disabled = true;
            recordingTimer.style.display = 'block';
            
            // Start timer
            recordingStartTime = Date.now();
            updateTimer();
            timerInterval = setInterval(updateTimer, 1000);

            // Store stream for later cleanup
            mediaRecorder.stream = stream;
        })
        .catch(error => {
            showMessage('Error accessing microphone: ' + error.message, 'error');
        });
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.recording) {
            mediaRecorder.stop();
            
            // Export the WAV file
            mediaRecorder.exportWAV(blob => {
                recordedBlob = blob;
                const audioUrl = URL.createObjectURL(blob);
                audioPlayer.src = audioUrl;
                audioPlayer.classList.remove('d-none');
                submitButton.disabled = false;
            });
            
            // Stop all tracks in the stream
            if (mediaRecorder.stream) {
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            
            // Update UI
            recordButton.disabled = false;
            stopButton.disabled = true;
            fileInput.disabled = false;
            recordingTimer.style.display = 'none';
            clearInterval(timerInterval);
        }
    }

    function updateTimer() {
        const elapsedTime = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsedTime / 60).toString().padStart(2, '0');
        const seconds = (elapsedTime % 60).toString().padStart(2, '0');
        timerDisplay.textContent = `${minutes}:${seconds}`;
    }

    // Add form submit handler
    uploadForm.addEventListener('submit', async function(event) {
        event.preventDefault(); // Ngăn form submit mặc định
        
        let file = fileInput.files[0];
        
        // Ưu tiên sử dụng recordedBlob nếu có
        if (recordedBlob) {
            file = null; // Đảm bảo không sử dụng file từ input
        } else if (!file) {
            showMessage('Please select an audio file or record audio', 'error');
            return;
        }

        try {
            // Show loading
            loadingDiv.classList.remove('d-none');
            resultDiv.classList.add('d-none');

            // Create FormData to send the file
            const formData = new FormData();
            
            if (file) {
                // If it's an uploaded file
                formData.append('file', file, file.name);
            } else {
                // If it's a recorded blob, create a File object with proper name and type
                const now = new Date();
                const fileName = `recording_${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate()}_${now.getHours()}${now.getMinutes()}${now.getSeconds()}.wav`;
                file = new File([recordedBlob], fileName, { type: 'audio/wav' });
                formData.append('file', file, fileName);
            }

            // Make API call to your backend
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                if (response.status === 422) {
                    throw new Error('Invalid file format or data. Please make sure you are sending a valid audio file.');
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log('Server response:', data); // Debug log
            loadingDiv.classList.add('d-none');
            displayResult(data);

        } catch (error) {
            loadingDiv.classList.add('d-none');
            if (error.message.includes('Failed to fetch') || error.message.includes('ERR_FAILED')) {
                showMessage('Cannot connect to the server. Please make sure the server is running at http://localhost:8000', 'error');
            } else {
                showMessage(error.message, 'error');
            }
            console.error('Error details:', error);
        }
    });

    function displayResult(data) {
        resultDiv.classList.remove('d-none');
        console.log('Displaying result:', data);
        
        if (data && data.success) {
            try {
                const predictions = data.prediction.predictions;
                const svmResult = predictions.SVM;
                const rfResult = predictions.RandomForest;
                const finalPrediction = data.prediction.final_prediction;
                const finalConfidence = data.prediction.final_confidence;

                // Update the result elements
                document.getElementById('svm-prediction').textContent = 
                    `SVM Prediction: ${svmResult.prediction} (${(svmResult.confidence * 100).toFixed(2)}% confidence)`;
                
                document.getElementById('rf-prediction').textContent = 
                    `Random Forest Prediction: ${rfResult.prediction} (${(rfResult.confidence * 100).toFixed(2)}% confidence)`;
                
                document.getElementById('final-prediction').textContent = 
                    `Final Prediction: ${finalPrediction}`;
                
                document.getElementById('confidence').textContent = 
                    `Confidence: ${(finalConfidence * 100).toFixed(2)}%`;

                // Thêm classes cho styling
                resultDiv.querySelectorAll('.prediction-details p').forEach(p => {
                    p.classList.add('alert', 'alert-info', 'py-2', 'mb-2');
                });
                
                document.getElementById('final-prediction').classList.add('alert', 'alert-success', 'py-2', 'fw-bold');
                
            } catch (error) {
                console.error('Error displaying result:', error);
                showMessage('Error displaying result: ' + error.message, 'error');
            }
        } else {
            showMessage(data?.error || 'Unknown error occurred', 'error');
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