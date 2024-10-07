document.getElementById('uploadForm').onsubmit = async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('csv_file').files[0];
    const formData = new FormData();
    formData.append('file', fileInput);

    const resultDiv = document.getElementById('result');
    resultDiv.textContent = '';  // Reset result message
    resultDiv.innerHTML = '';  // Clear previous results

    try {
        // Send the file to the backend
        const response = await fetch('http://127.0.0.1:8000/api/predict_batch/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.predictions && result.probabilities) {
            createTable(result.predictions, result.probabilities);
        } else {
            resultDiv.textContent = 'No predictions returned.';
        }

    } catch (error) {
        resultDiv.textContent = 'Error: Could not reach the server.';
        resultDiv.classList.add('error');
    }
};

// Function to create a table and display results with fraud probabilities
function createTable(predictions, probabilities) {
    const resultDiv = document.getElementById('result');
    
    const table = document.createElement('table');
    table.classList.add('styled-table');
    
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Transaction</th>
            <th>Prediction</th>
            <th>Fraud Probability (%)</th>
        </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    predictions.forEach((prediction, index) => {
        const row = document.createElement('tr');
        
        const transactionCell = document.createElement('td');
        transactionCell.textContent = `Transaction ${index + 1}`;
        row.appendChild(transactionCell);

        const predictionCell = document.createElement('td');
        predictionCell.innerHTML = prediction === 0 
            ? '<span class="valid">✅ Valid (Real)</span>' 
            : '<span class="fraudulent">⚠️ Fraudulent (Fake)</span>';
        row.appendChild(predictionCell);

        const probabilityCell = document.createElement('td');
        probabilityCell.textContent = `${(probabilities[index] * 100).toFixed(2)}%`;
        row.appendChild(probabilityCell);

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    resultDiv.appendChild(table);
}
