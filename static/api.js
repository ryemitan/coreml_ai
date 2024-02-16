document.addEventListener('DOMContentLoaded', function () {
    const modal = document.getElementById('deploy-model-modal');
    const modelDropdown = document.getElementById('model-dropdown');
    const deployButton = document.getElementById('deploy-button');
    const apiSummary = document.getElementById('api-summary');
    const apiTableHeaders = document.getElementById('api-table-headers');
    const apiTableBody = document.getElementById('api-table').getElementsByTagName('tbody')[0];

    function openDeployModelModal() {
        modal.style.display = 'block';
        fetchTrainedModels();
    }

    

    function closeDeployModelModal() {
        modal.style.display = 'none';
    }

    function fetchTrainedModels() {
        fetch('/get_trained_models')
            .then(response => response.json())
            .then(data => {
                populateModelDropdown(data.trainedModels);
            })
            .catch(error => console.error('Error fetching trained models:', error.message));
    }

    function populateModelDropdown(models) {
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.filename;
            option.text = model.name;
            modelDropdown.add(option);
        });
    }

    function deployExistingModel() {
        const selectedModel = modelDropdown.value;
        console.log('Deploying existing model:', selectedModel);
        closeDeployModelModal();
    }

    window.onclick = function (event) {
        if (event.target === modal) {
            closeDeployModelModal();
        }
    };

    function setApiSummary(apiInfo) {
        apiSummary.innerHTML = '';
        createApiSummaryCard('APIs', apiInfo.api_count);
        createApiSummaryCard('Model Files', apiInfo.model_files_count);
        createApiSummaryCard('Folders', apiInfo.folders_count);
    }

    function createApiSummaryCard(title, count) {
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('api-summary-card');
        cardDiv.innerHTML = `<h3>${title}</h3><p>${count}</p>`;
        apiSummary.appendChild(cardDiv);
    }

     // Fetch API information dynamically from Flask
     fetch('/get_api_info')
        .then(response => response.json())
        .then(apiInfo => {
            setApiSummary(apiInfo.apiInfo);  // Adjust the property name based on your Flask route
            // Update other parts of your UI using the fetched data as needed
        })
        .catch(error => {
            console.error('Error fetching API information:', error.message);
            // Update loading message to indicate an error
            document.getElementById('api-count').innerText = 'Error loading API count';
        });
   
    fetchApiColumnsAndPopulateTable();

    function fetchApiColumnsAndPopulateTable() {
        fetch('/api.json')
            .then(response => response.json())
            .then(apiColumns => {
                populateTableHeaders(apiColumns);
                fetchApiDetailsAndPopulateTable(apiColumns);
            })
            .catch(error => console.error('Error fetching API columns:', error.message));
    }

    function populateTableHeaders(columns) {
        columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = column;
            apiTableHeaders.appendChild(th);
        });
    }

    // function fetchApiDetailsAndPopulateTable(columns) {
    //     fetch('/get_api_details')
    //         .then(response => response.json())
    //         .then(apiDetails => {
    //             populateTableBody(apiDetails, columns);
    //         })
    //         .catch(error => console.error('Error fetching API details:', error.message));
    // }

    function populateTableBody(details, columns) {
        details.forEach(detail => {
            const tr = document.createElement('tr');
            columns.forEach(column => {
                const td = document.createElement('td');
                td.textContent = detail[column];
                tr.appendChild(td);
            });
            apiTableBody.appendChild(tr);
        });
    }

    deployButton.addEventListener('click', deployExistingModel);
});
