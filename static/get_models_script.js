function navigateToPredict() {
    // calling 'predict' route for the predict.html page
    window.location.href = "{{ url_for('pre_dict') }}";
}


document.addEventListener('DOMContentLoaded', function () {
    
    function populateModelsTable(modelsInfo) {
        // console.log(modelsInfo.Model);
        const modelsTableBody = document.getElementById("modelsTableBody");
        modelsTableBody.innerHTML = ""; // Clear existing rows

        if (modelsInfo && Array.isArray(modelsInfo.models)) {
            modelsInfo.models.forEach(modelInfo => {
                const row = document.createElement("tr");
// console.log(modelsInfo.models)
// console.log('Accuracy:', modelInfo.models.Accuracy)
// console.log('RMSE:', modelInfo)


                const modelNameCell = document.createElement("td");
                modelNameCell.textContent = modelInfo.Model;

                const modelFileNameCell = document.createElement("td");
                modelFileNameCell.textContent = modelInfo.Model_Filename;

                // const accuracyCell = document.createElement("td");
                // accuracyCell.textContent = (modelInfo.Accuracy * 100).toFixed(2);
                

                // Create cell for accuracy or RMSE based on model type
                const accuracyCell = document.createElement("td");
                const accuracyKey = Object.keys(modelInfo).find(key => key.toLowerCase().includes('accuracy'));
                const rmseKey = Object.keys(modelInfo).find(key => key.toLowerCase().includes('rmse'));

                if (rmseKey) {
                    // For regression models, display RMSE
                    accuracyCell.textContent = modelInfo[rmseKey].toFixed(4);
                } else if (accuracyKey) {
                    // For classification models, display Accuracy
                    const accuracyValue = (modelInfo[accuracyKey] * 100).toFixed(2);
                    accuracyCell.textContent = isNaN(accuracyValue) ? 'NaN' : accuracyValue;
                } else {
                    // Handle other model types as needed
                    accuracyCell.textContent = 'N/A';
                }

                // Log accuracy values for debugging
                console.log(`Model: ${modelInfo.Model}, Accuracy: ${accuracyKey ? modelInfo[accuracyKey] : 'N/A'}, RMSE: ${rmseKey ? modelInfo[rmseKey] : 'N/A'}`);
                // ...

                const selectCell = document.createElement("td");

                const radioButton = document.createElement("input");
                radioButton.type = "radio";
                radioButton.name = "selectedModel";
                // console.log(modelInfo.Model_Filename);
                radioButton.value = modelInfo.Model_Filename;

                // Set the first radio button as checked by default
                if (!document.querySelector('input[type="radio"][name="selectedModel"]:checked')) {
                    radioButton.checked = true;
                }


                // Add an event listener to update the selected model's filename when a radio button is selected
                radioButton.addEventListener("change", function () {
                    const selectedModelFilename = document.getElementById("selectedModel");
                    // console.log("selectedModel:", selectedModelFilename);

                    // Check if the element exists before setting its textContent
                    if (selectedModelFilename) {
                        selectedModelFilename.textContent = modelInfo.Model_Filename;
                    }

                    // Set the model_name in the hidden input field for submission to the /predict API
                    const model_name = document.querySelector('input[type="radio"][name="selectedModel"]:checked');
                    // model_name.value = modelInfo.Model_Filename;
                });

                selectCell.appendChild(radioButton);

                row.appendChild(modelNameCell);
                row.appendChild(accuracyCell);
                row.appendChild(modelFileNameCell);
                row.appendChild(selectCell);

                modelsTableBody.appendChild(row);
            });
        } else {
            console.error("Invalid or missing models data:", modelsInfo);
            updateStatusMessage("Error: Invalid or missing models data.");
        }
    }

    // Fetch models data and populate the table
    fetch("/get_models")
        .then(response => {
            // console.log(response);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if ('models' in data && Array.isArray(data.models)) {
                console.log("Received data:", data);
                populateModelsTable(data);

                // Find the selected radio button dynamically
                var selectedModelElement = document.querySelector('input[type="radio"][name="selectedModel"]:checked');
                console.log("selectedModelElement:", selectedModelElement);

                if (selectedModelElement) {
                    // Now you can use 'selectedModelElement.value' as needed
                    console.log("Default model name:", selectedModelElement.value);

                    // Set the default model name based on the selected model element
                    var model_name = document.querySelector('input[type="radio"][name="selectedModel"]:checked').value; //document.getElementById("model_name");
                    console.log("model_name:", model_name);
                    if (model_name) {
                        model_name.value = selectedModelElement.value;
                    } else {
                        console.error('Error: Element with ID "model_name" not found.');
                    }
                } else {
                    console.error('Error: No model selected.');
                }
            } else {
                console.error("Invalid or missing models data:", data);
                updateStatusMessage("Error: Invalid or missing models data.");
            }
        })
        .catch(error => console.error('Error:', error));

    function handleModelSelection(modelName) {
        console.log(modelName)
        // Assuming you have an input field with the id 'model_name' in the Predict page
        document.getElementById('model_name').value = modelName;
        
    }


    // function handleModelSelection(modelName) {
    //     // Assuming you have an input field with the id 'model_name' in the Predict page
    //     var selectedModelElement = document.querySelector('input[type="radio"][name="selectedModel"]:checked');
    //     if (selectedModelElement) {
    //         selectedModelElement.value = modelName;
    //     } else {
    //         console.error('Error: No model selected.');
    //     }
    // }

    document.addEventListener('DOMContentLoaded', function () {
        var modelsTable = document.getElementById('modelsTableBody');

        modelsTable.addEventListener('click', function (event) {
            var target = event.target;

            // Check if the clicked element is a table row
            if (target.tagName.toLowerCase() === 'tr') {
                // Find the radio button within the clicked table row
                var radioButton = target.querySelector('input[type="radio"][name="selectedModel"]');
                
                // Check if the radio button is found
                if (radioButton) {
                    // Set the value of the hidden input field
                    var selectedModelElement = document.querySelector('input[type="radio"][name="selectedModel"]:checked');
                    console.log(selectedModelElement)
                    if (selectedModelElement) {
                        selectedModelElement.value = radioButton.value;
                        // console.log("Default model name:", radioButton.value);
                    } else {
                        console.error('Error: Element with ID "selectedModel" not found.');
                    }
                }
            }
        });
    });
    

    
    // function navigateToPredict() {
    //     // calling 'predict' route for the pre_dict.html page
    //     window.location.href = predictUrl;
    // }

    

    
    function navigateToPredict() {
        // Get the value of the hidden input field
        var selectedModelElement = document.querySelector('input[type="radio"][name="selectedModel"]:checked');
        console.log(selectedModelElement); // Log the element to the console
        var selectedModel = selectedModelElement ? selectedModelElement.value : null;
    
        // Redirect to the predict page with the selected model name
        
        // window.location.href = "/pre_dict?model_name=" + encodeURIComponent(selectedModel);
        window.location.href = "/cre_dict?model_name=" + encodeURIComponent(selectedModel);
        // console.log("Current URL:", window.location.href);
    }
    

    // function navigateToPredict() {
    //     // Get the value of the hidden input field
    //     var selectedModelElement = document.querySelector('input[type="radio"][name="selectedModel"]:checked');
    //     console.log(selectedModelElement); // Log the element to the console
    //     var selectedModel = selectedModelElement ? selectedModelElement.value : null;
    //     // var selectedModel = document.getElementById('selectedModel').value;

    //     // Redirect to the predict page with the selected model name
    //     window.location.href = "/pre_dict"//?model_name=" + selectedModel;
    // }
    if (window.location.pathname === '/signin-page') {
        document.getElementById('signInForm').style.display = 'block';
    } else {
        document.getElementById('signInForm').style.display = 'none';
    }
    
});
