// train_script.js

// Initialize lines variable in the global scope
let lines;
let headers;

document.getElementById("trainData").addEventListener("change", function () {
    const trainDataFile = this.files[0];

    if (trainDataFile) {
        const reader = new FileReader();
        reader.onload = function (event) {
            const csvData = event.target.result;
            lines = csvData.split("\n"); // Update the global lines variable
            headers = lines[0].split(",");
    
            // Clean up column names
            headers = headers.map(column => column.trim().replace(/\r\n/g, ''));
    
            // Populate the target column dropdown
            populateTargetColumnsDropdown(headers);
    
            // Populate column features table
            populateColumnFeaturesTable(headers, lines.slice(1)); // Exclude header row
    
            // Ensure that the Trainer Class is updated after the target column is populated
            updateTrainerClassAfterPopulatingTargetColumn();
        };
        reader.readAsText(trainDataFile);
    }
    
});

// Event listener for the change in the target column
document.getElementById("targetColumn").addEventListener("change", function () {
    const selectedTargetColumn = this.value;

    // Update Trainer Class
    updateTrainerClass(selectedTargetColumn);
});

// Event listener for the change in the specified data type
document.getElementById("dataTypeInput").addEventListener("change", function () {
    const selectedTargetColumn = document.getElementById("targetColumn").value;

    // Update Trainer Class based on the selected target column
    updateTrainerClass(selectedTargetColumn);
});


function populateTargetColumnsDropdown(headers) {
    const targetColumnSelect = document.getElementById("targetColumn");

    // Clear existing options
    targetColumnSelect.innerHTML = "";

    headers.forEach(header => {
        const option = document.createElement("option");
        option.value = header;
        option.text = header;
        targetColumnSelect.add(option);
    });
}

function populateColumnFeaturesTable(headers, data) {
    const columnFeaturesTableBody = document.getElementById("columnFeaturesTableBody");
    columnFeaturesTableBody.innerHTML = "";

    headers.forEach((header, index) => {
        const row = document.createElement("tr");

        const columnCell = document.createElement("td");
        columnCell.textContent = header;

        const dataTypeCell = document.createElement("td");
        const dataType = getDataType(data, index);
        dataTypeCell.textContent = dataType;

        const missingDataCountCell = document.createElement("td");
        const missingDataCount = countMissingData(data, index);
        missingDataCountCell.textContent = missingDataCount;

        row.appendChild(columnCell);
        row.appendChild(dataTypeCell);
        row.appendChild(missingDataCountCell);

        columnFeaturesTableBody.appendChild(row);
    });
}

function updateTrainerClass(selectedTargetColumn) {
    const trainerClassDisplay = document.getElementById("trainerClassDisplay");
    const dataTypeInput = document.getElementById("dataTypeInput");

    if (selectedTargetColumn) {
        // Check if user-specified data type is available
        const userSpecifiedDataType = dataTypeInput.value.trim();

        // Use user-specified data type if provided, otherwise, use automatic detection
        const dataType = userSpecifiedDataType || getDataType(lines, headers.indexOf(selectedTargetColumn));

        trainerClassDisplay.textContent = dataType === "Numeric" ? "Regression" : "Classification";
    } else {
        trainerClassDisplay.textContent = ""; // Clear Trainer Class if no target column selected
    }
}

function getDataType(data, columnIndex) {
    const columnValues = data.map(row => {
        const columns = row.split(",");
        return columns.length > columnIndex ? columns[columnIndex].trim() : '';
    });

    // Remove empty and undefined values
    const filteredValues = columnValues.filter(value => value !== undefined && value !== "");

    if (filteredValues.length === 0) {
        return "Unknown"; // No valid values to determine data type
    }

    const uniqueValues = new Set(filteredValues);

    // Check if all unique values are numeric
    if (Array.from(uniqueValues).every(value => !isNaN(value))) {
        return "Numeric";
    } else {
        return "Categorical";
    }
}


// Replace with your specific implementation to count missing data in a column
function countMissingData(data, columnIndex) {
    const columnValues = data.map(row => {
        const columns = row.split(",");
        return columns.length > columnIndex ? columns[columnIndex].trim() : '';
    });

    // Count the occurrences of empty or undefined values in the column
    return columnValues.filter(value => value === "").length;
}

// Ensure that the document is ready before executing the code
document.addEventListener('DOMContentLoaded', function () {
    // Your additional code specific to train_script.js (if any)
    // ...
});