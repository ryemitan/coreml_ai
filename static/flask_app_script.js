document.addEventListener('DOMContentLoaded', function () {
    // Function to update status message and show/hide notification
    function updateStatusMessage(message, showNotification = true) {
        const notificationDiv = document.getElementById("notification");
        notificationDiv.textContent = message;

        if (showNotification) {
            notificationDiv.style.display = "block";
        } else {
            notificationDiv.style.display = "none";
        }
    }

    // Event listener for training data upload
    document.getElementById("trainForm").addEventListener("submit", function (e) {
        e.preventDefault();

        const trainDataInput = document.getElementById("trainData");
        const targetColumnSelect = document.getElementById("targetColumn");
        const trainerClassDisplay = document.getElementById("trainerClassDisplay");

        const trainDataFile = trainDataInput.files;
        const target_column = targetColumnSelect.value;
        const trainer_class = trainerClassDisplay.innerText;

        if (!trainDataFile) {
            updateStatusMessage("Please select a training CSV file.");
            return;
        }

        if (!target_column) {
            updateStatusMessage("Please select a target column.");
            return;
        }

        const requestData = new FormData();
        requestData.append("file", trainDataFile[0]);
        requestData.append("target_column", target_column);
        requestData.append("trainer_class", trainer_class);

        const requestOptions = {
            method: "POST",
            body: requestData,
        };

        updateStatusMessage("Your File has been Uploaded and Model Training in Process ...");

        fetch("/train_models", requestOptions)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log("Response Data:", data);

                if (!data.best_model_info || !data.best_model_info.headers) {
                    throw new Error("Invalid or missing data in the response.");
                }

                const headers = data.best_model_info.headers;
                populateTargetColumnsDropdown(headers);

                updateStatusMessage("Training data uploaded. Target column: " + target_column);

                // Check if a redirect URL is provided
                if (data.redirect_url) {
                    // Redirect to the provided URL
                    window.location.href = data.redirect_url;
                } else {
                    // Handle the rest of your logic (if needed)
                    updateStatusMessage("Training completed successfully.");
                }
            })
            .catch(error => {
                console.error("Error:", error);
                updateStatusMessage("Error uploading training data.");

                // Check if a redirect URL is provided
                if (data.redirect_url) {
                    // Redirect to the provided URL
                    window.location.href = data.redirect_url;
                } else {
                    // Handle the rest of your error handling logic (if needed)
                }
            })
            .finally(function () {
                // Hide moving dotted line after training completion
                document.getElementById("progressContainer").style.display = "none";
            });
    });

    // Event listener for test data upload
    document.getElementById("testForm").addEventListener("submit", function (e) {
        e.preventDefault();

        const testDataInput = document.getElementById("testData");
        const testDataFile = testDataInput.files[0];

        if (!testDataFile) {
            updateStatusMessage("Please select a test CSV file.");
            return;
        }

        const selectedModelRadioButtons = document.querySelectorAll('input[name="selectedModel"]:checked');

        if (selectedModelRadioButtons.length === 0) {
            updateStatusMessage("Please select a model.");
            return;
        }

        // Get the value of the selected radio button (model_filename)
        const selectedModel = selectedModelRadioButtons[0].value;

        // Set the model_name in the hidden input field
        document.getElementById("model_name").value = selectedModel;

        const testFormData = new FormData();
        testFormData.append("file", testDataFile);
        testFormData.append("model_name", selectedModel);

        updateStatusMessage("Uploading test data...");

        fetch("/predict", {
            method: "POST",
            body: testFormData,
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log("Response Data:", data);
                updateStatusMessage("Test data uploaded.");
                console.log("Prediction API Response:", data);
                console.log("Predictions Array:", data.predictions);
                displayPredictionsTable(data.predictions);
                displayJSONTable({...data.predictions, model_name: data.model_name});
            })
            .catch(error => {
                console.error("Error:", error);
                updateStatusMessage("Error uploading test data.");
            });
    });

    // Event listener to fetch and list available models
    document.getElementById("fetchModelsButton").addEventListener("click", function () {
        updateStatusMessage("Fetching available models...");

        fetch("/get_models", {
            method: "GET",
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(modelsInfo => {
                populateModelsTable(modelsInfo);
            })
            .catch(error => {
                console.error("Error:", error);
                updateStatusMessage("Error fetching models.");
            });
    });

    // Function to update the selected model's filename
    function updateSelectedModelFilename(filename) {
        const selectedModelFilename = document.getElementById("selectedModelFilename");
        selectedModelFilename.textContent = filename;
    }

    // Event listener for selecting a model in the models table
    document.getElementById("modelsTableBody").addEventListener("click", function (e) {
        const clickedElement = e.target;
        const selectedModelRadio = clickedElement.closest('input[type="radio"][name="selectedModel"]');
        if (selectedModelRadio) {
            const modelFilename = selectedModelRadio.value;
            updateSelectedModelFilename(modelFilename);
        }
    });

    // Function to populate the models table with the provided modelsInfo
    function populateModelsTable(modelsInfo) {
        const modelsTableBody = document.getElementById("modelsTableBody");
        modelsTableBody.innerHTML = ""; // Clear previous data

        if (modelsInfo.models && modelsInfo.models.length > 0) {
            modelsInfo.models.forEach(model => {
                const row = modelsTableBody.insertRow();
                const cell1 = row.insertCell(0);
                const cell2 = row.insertCell(1);
                const cell3 = row.insertCell(2);

                const radioButton = document.createElement("input");
                radioButton.type = "radio";
                radioButton.name = "selectedModel";
                radioButton.value = model.model_filename;

                cell1.appendChild(radioButton);
                cell2.textContent = model.model_filename;
                cell3.textContent = model.accuracy.toFixed(4);
            });

            // Enable the button to fetch predictions
            document.getElementById("fetchPredictionsButton").removeAttribute("disabled");
            updateStatusMessage("Available models fetched.");
        } else {
            updateStatusMessage("No models available.");
            // Disable the button to fetch predictions if no models are available
            document.getElementById("fetchPredictionsButton").setAttribute("disabled", "true");
        }
    }

    // Event listener for fetching predictions
    document.getElementById("fetchPredictionsButton").addEventListener("click", function () {
        updateStatusMessage("Fetching predictions...");

        fetch("/predict", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log("Response Data:", data);
                updateStatusMessage("Predictions fetched.");
                displayPredictionsTable(data.predictions);
                displayJSONTable({...data.predictions, model_name: data.model_name});
            })
            .catch(error => {
                console.error("Error:", error);
                updateStatusMessage("Error fetching predictions.");
            });
    });

    // Function to display predictions in a table
    function displayPredictionsTable(predictions) {
        const predictionsTableBody = document.getElementById("predictionsTableBody");
        predictionsTableBody.innerHTML = ""; // Clear previous data

        if (predictions && predictions.length > 0) {
            predictions.forEach(prediction => {
                const row = predictionsTableBody.insertRow();
                for (const key in prediction) {
                    if (Object.hasOwnProperty.call(prediction, key)) {
                        const cell = row.insertCell();
                        cell.textContent = prediction[key];
                    }
                }
            });
        }
    }

    // Function to display JSON data in a table
    function displayJSONTable(data) {
        const jsonTableBody = document.getElementById("jsonTableBody");
        jsonTableBody.innerHTML = ""; // Clear previous data

        const row = jsonTableBody.insertRow();
        const cell = row.insertCell();

        // Convert the JSON data to a string and add it to the cell
        cell.textContent = JSON.stringify(data, null, 2);
    }

    // Function to populate the target column dropdown
    function populateTargetColumnsDropdown(headers) {
        const targetColumnSelect = document.getElementById("targetColumn");
        targetColumnSelect.innerHTML = ""; // Clear previous options

        headers.forEach(header => {
            const option = document.createElement("option");
            option.value = header;
            option.text = header;
            targetColumnSelect.add(option);
        });
    }

    // Event listener for submitting the predict form
    document.getElementById("predictForm").addEventListener("submit", function (e) {
        e.preventDefault();

        // ... (Your existing code)

        const model_name = document.getElementsByName("model_name")[0].value;

        // Fetch column analysis data based on the selected model
        fetch(`/get_column_analysis?model_name=${model_name}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(columnAnalysis => {
                // Display column analysis in the table
                populateColumnAnalysisTable(columnAnalysis);
            })
            .catch(error => {
                console.error("Error:", error);
            });
    });

    // Function to populate the column analysis table
    function populateColumnAnalysisTable(columnAnalysis) {
        const columnAnalysisBody = document.getElementById("columnAnalysisBody");
        columnAnalysisBody.innerHTML = ""; // Clear existing rows

        if (columnAnalysis && Array.isArray(columnAnalysis)) {
            columnAnalysis.forEach(column => {
                const row = document.createElement("tr");

                const columnNameCell = document.createElement("td");
                columnNameCell.textContent = column.column;

                const analysisCell = document.createElement("td");
                analysisCell.textContent = column.analysis;

                row.appendChild(columnNameCell);
                row.appendChild(analysisCell);

                columnAnalysisBody.appendChild(row);
            });
        }
    }
    if (window.location.pathname === '/signin-page') {
        document.getElementById('signInForm').style.display = 'block';
    } else {
        document.getElementById('signInForm').style.display = 'none';
    }
    
});

