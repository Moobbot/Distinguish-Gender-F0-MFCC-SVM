#ccc;
            border-radius: 5px;
            max-width: 600px;
        }
        h1 {
            text-align: center;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        button {
            padding: 10px 15px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #218838;
        }
        #result {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            display: none;
        }
    </style>
</head>
<body>

    <h1>Gender Classification from Audio</h1>
    <form id="uploadForm">
        <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.m4a,.ogg" required>
        <button type="submit">Upload and Classify</button>
    </form>

    <div id="result">
        <h2>Prediction Result</h2>
        <p id="prediction"></p>
        <p id="confidence"></p>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(event) {
            event.preventDefault(); // Prevent the form from submitting the default way

            const fileInput = document.getElementById('audioFile');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('http://localhost:8000/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();
                document.getElementById('prediction').innerText = `Gender: ${data.predictions.final_prediction}`;
                document.getElementById('confidence').innerText = `Confidence: ${data.predictions.final_confidence}`;
                document.getElementById('result').style.display = 'block';

            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while processing your request.');
            }
        });
    </script>

</body>
</html>
```

### Instructions to Use:
1. **Save the HTML code**: Copy the above code and save it as `index.html`.
2. **Run your API**: Make sure your FastAPI server is running at `http://localhost:8000`.
3. **Open the HTML file**: Open the `index.html` file in a web browser.
4. **Upload an audio file**: Use the file input to select an audio file (WAV, MP3, FLAC, M4A, or OGG) and click the "Upload and Classify" button.
5. **View results**: After the file is processed, the predicted gender and confidence level will be displayed below the form.

This simple web page provides a user-friendly interface for interacting with your gender classification API.