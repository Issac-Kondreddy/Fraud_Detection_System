document.getElementById('fraudForm').onsubmit = async function (e) {
    e.preventDefault();

    const amount = document.getElementById('transaction_amount').value;
    const time = document.getElementById('transaction_time').value;
    const resultDiv = document.getElementById('result');

    // Reset result message
    resultDiv.textContent = '';
    resultDiv.classList.remove('success', 'error');

    if (!amount || !time) {
        resultDiv.textContent = 'Please fill in all fields.';
        resultDiv.classList.add('error');
        return;
    }

    // Construct the input data
    const data = {
        "data": [parseFloat(time), parseFloat(amount)] // Add more fields here as needed
    };

    try {
        // Send the data to the backend API
        const response = await fetch('http://127.0.0.1:8000/api/predict_fraud/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Display the result
        if (result.prediction === 1) {
            resultDiv.textContent = '⚠️ Fraudulent Transaction Detected!';
            resultDiv.classList.add('error');
        } else {
            resultDiv.textContent = '✅ Transaction is Valid!';
            resultDiv.classList.add('success');
        }

    } catch (error) {
        resultDiv.textContent = 'Error: Could not reach the server.';
        resultDiv.classList.add('error');
    }
};
