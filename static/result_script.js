// // result_script.js

// // Function to navigate to deployment setup
// function navigateToDeployment() {
//     window.location.href = "/production_deploy_setup";
// }

// // Function to download result as CSV
// function downloadResultAsCSV() {
//     // Logic to download the result as CSV
// }

// // Function to download result as Excel
// function downloadResultAsExcel() {
//     // Logic to download the result as Excel
// }

// // Function to download result as PDF
// function downloadResultAsPDF() {
//     // Logic to download the result as PDF
// }

// // Function to navigate to evaluation deployment setup
// function navigateToEvaluationDeployment() {
//     window.location.href = "/evaluation_production_deploy_setup";
// }

// // Function to download evaluation result as CSV
// function downloadEvaluationAsCSV() {
//     // Logic to download the evaluation result as CSV
// }

// // Function to download evaluation result as Excel
// function downloadEvaluationAsExcel() {
//     // Logic to download the evaluation result as Excel
// }

// // Function to download evaluation result as PDF
// function downloadEvaluationAsPDF() {
//     // Logic to download the evaluation result as PDF
// }

// // You can add more functions as needed




// // Retrieve data from the URL query parameter
// const urlParams = new URLSearchParams(window.location.search);
// const data = JSON.parse(urlParams.get('data'));

// // Populate Result Table
// const resultTableHeaders = data.resultTableHeaders;
// const resultTableData = data.resultTableData;
// populateTable('resultTable', resultTableHeaders, resultTableData);

// // Populate Model Evaluation Results Table
// const evaluationTableData = data.evaluationTableData;
// populateTable('evaluationTable', ['Metric', 'Value'], evaluationTableData);

// // Function to populate a table with headers and data
// function populateTable(tableId, headers, data) {
//     const table = document.getElementById(tableId);
//     const headerRow = table.querySelector('thead');
//     const body = table.querySelector('tbody');

//     // Populate headers
//     let headerHtml = '';
//     headers.forEach(header => {
//         headerHtml += `<th>${header}</th>`;
//     });
//     headerRow.innerHTML = `<tr>${headerHtml}</tr>`;

//     // Populate data rows
//     let bodyHtml = '';
//     data.forEach(row => {
//         let rowHtml = '';
//         row.forEach(cell => {
//             rowHtml += `<td>${cell}</td>`;
//         });
//         bodyHtml += `<tr>${rowHtml}</tr>`;
//     });
//     body.innerHTML = bodyHtml;
// }

// Logic for Result Table Navigation and Download
function navigateToDeployment() {
    window.location.href = "/production_deploy_setup";
}

function downloadResultAsCSV() {
    const resultTable = document.getElementById('resultTable');
    tableToCSV(resultTable, 'result');
}

function downloadResultAsExcel() {
    const resultTable = document.getElementById('resultTable');
    tableToExcel(resultTable, 'result');
}

function downloadResultAsPDF() {
    const resultTable = document.getElementById('resultTable');
    tableToPDF(resultTable, 'result');
}

// Logic for Evaluation Table Navigation and Download
function navigateToEvaluationDeployment() {
    window.location.href = "/evaluation_production_deploy_setup";
}

function downloadEvaluationAsCSV() {
    const evaluationTable = document.getElementById('evaluationTable');
    tableToCSV(evaluationTable, 'evaluation');
}

function downloadEvaluationAsExcel() {
    const evaluationTable = document.getElementById('evaluationTable');
    tableToExcel(evaluationTable, 'evaluation');
}

function downloadEvaluationAsPDF() {
    const evaluationTable = document.getElementById('evaluationTable');
    tableToPDF(evaluationTable, 'evaluation');
}

// Function to convert a table to CSV format
function tableToCSV(table, filename) {
    // Placeholder logic for converting table to CSV
    console.log(`Converting ${filename} table to CSV`);
}

// Function to convert a table to Excel format
function tableToExcel(table, filename) {
    // Placeholder logic for converting table to Excel
    console.log(`Converting ${filename} table to Excel`);
}

// Function to convert a table to PDF format
function tableToPDF(table, filename) {
    // Placeholder logic for converting table to PDF
    console.log(`Converting ${filename} table to PDF`);
}

if (window.location.pathname === '/signin-page') {
    document.getElementById('signInForm').style.display = 'block';
} else {
    document.getElementById('signInForm').style.display = 'none';
}
