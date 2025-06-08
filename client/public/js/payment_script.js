document.addEventListener('DOMContentLoaded', function() {
    const paymentForm = document.getElementById('payment-form');
    const paymentMethods = document.getElementsByName('paymentMethod');
    const cardDetails = document.getElementById('card-details');
    const upiDetails = document.getElementById('upi-details');
    const toast = document.getElementById('toast');

    // Handle payment method selection
    paymentMethods.forEach(method => {
        method.addEventListener('change', function() {
            if (this.value === 'upi') {
                cardDetails.style.display = 'none';
                upiDetails.style.display = 'block';
                // Make card fields not required
                document.getElementById('cardNumber').required = false;
                document.getElementById('expiryDate').required = false;
                document.getElementById('cvv').required = false;
                // Make UPI field required
                document.getElementById('upiId').required = true;
            } else {
                cardDetails.style.display = 'block';
                upiDetails.style.display = 'none';
                // Make card fields required
                document.getElementById('cardNumber').required = true;
                document.getElementById('expiryDate').required = true;
                document.getElementById('cvv').required = true;
                // Make UPI field not required
                document.getElementById('upiId').required = false;
            }
        });
    });

    // Format card number input
    const cardNumberInput = document.getElementById('cardNumber');
    cardNumberInput.addEventListener('input', function(e) {
        let value = e.target.value.replace(/\D/g, '');
        let formattedValue = '';
        for (let i = 0; i < value.length; i++) {
            if (i > 0 && i % 4 === 0) {
                formattedValue += ' ';
            }
            formattedValue += value[i];
        }
        e.target.value = formattedValue;
    });

    // Format expiry date input
    const expiryDateInput = document.getElementById('expiryDate');
    expiryDateInput.addEventListener('input', function(e) {
        let value = e.target.value.replace(/\D/g, '');
        if (value.length > 2) {
            value = value.slice(0, 2) + '/' + value.slice(2, 4);
        }
        e.target.value = value;
    });

    // Format CVV input
    const cvvInput = document.getElementById('cvv');
    cvvInput.addEventListener('input', function(e) {
        e.target.value = e.target.value.replace(/\D/g, '').slice(0, 3);
    });

    // Handle form submission
    paymentForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Show loading state
        const payButton = document.querySelector('.pay-btn');
        const originalButtonText = payButton.textContent;
        payButton.textContent = 'Processing...';
        payButton.disabled = true;

        try {
            // Get form data
            const formData = new FormData(paymentForm);
            const paymentData = {
                personalInfo: {
                    fullName: formData.get('fullName'),
                    email: formData.get('email'),
                    phone: formData.get('phone')
                },
                shippingAddress: {
                    address: formData.get('address'),
                    city: formData.get('city'),
                    state: formData.get('state'),
                    pincode: formData.get('pincode')
                },
                paymentMethod: formData.get('paymentMethod'),
                paymentDetails: formData.get('paymentMethod') === 'upi' 
                    ? { upiId: formData.get('upiId') }
                    : {
                        cardNumber: formData.get('cardNumber'),
                        expiryDate: formData.get('expiryDate'),
                        cvv: formData.get('cvv')
                    }
            };

            // Send payment data to server
            const response = await fetch('/process-payment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(paymentData)
            });

            const result = await response.json();

            if (result.success) {
                showToast('Payment successful! Redirecting to order confirmation...', 'success');
                // Redirect to order confirmation page after 2 seconds
                setTimeout(() => {
                    window.location.href = '/order-confirmation';
                }, 2000);
            } else {
                throw new Error(result.message || 'Payment failed');
            }
        } catch (error) {
            showToast(error.message || 'An error occurred during payment', 'error');
            // Reset button state
            payButton.textContent = originalButtonText;
            payButton.disabled = false;
        }
    });

    // Function to show toast messages
    function showToast(message, type = 'info') {
        toast.textContent = message;
        toast.style.backgroundColor = type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#333';
        toast.style.display = 'block';
        
        setTimeout(() => {
            toast.style.display = 'none';
        }, 3000);
    }
}); 