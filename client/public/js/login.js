document.addEventListener("DOMContentLoaded", () => {
	const loginForm = document.getElementById("loginForm");
	const errorMessage = document.getElementById("error-message");
	const loginButton = document.getElementById("loginButton");
	const buttonText = loginButton.querySelector(".button-text");
	const emailError = document.getElementById("email-error");
	const passwordError = document.getElementById("password-error");

	// Function to show error message
	function showError(message, type = "general") {
		if (type === "general") {
			errorMessage.textContent = message;
			errorMessage.classList.add("show");
			setTimeout(() => {
				errorMessage.classList.remove("show");
			}, 5000);
		} else if (type === "email") {
			emailError.textContent = message;
			emailError.style.display = "block";
		} else if (type === "password") {
			passwordError.textContent = message;
			passwordError.style.display = "block";
		}
	}

	// Function to clear errors
	function clearErrors() {
		errorMessage.classList.remove("show");
		emailError.style.display = "none";
		passwordError.style.display = "none";
	}

	// Function to set loading state
	function setLoading(isLoading) {
		if (isLoading) {
			loginButton.classList.add("loading");
			buttonText.textContent = "Logging in...";
			loginButton.disabled = true;
		} else {
			loginButton.classList.remove("loading");
			buttonText.textContent = "Login";
			loginButton.disabled = false;
		}
	}

	// Function to validate email
	function isValidEmail(email) {
		const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
		return emailRegex.test(email);
	}

	// Function to validate form
	function validateForm(email, password) {
		let isValid = true;
		clearErrors();

		if (!email.trim()) {
			showError("Please enter your email or username", "email");
			isValid = false;
		} else if (email.includes("@") && !isValidEmail(email)) {
			showError("Please enter a valid email address", "email");
			isValid = false;
		}

		if (!password.trim()) {
			showError("Please enter your password", "password");
			isValid = false;
		} else if (password.length < 6) {
			showError(
				"Password must be at least 6 characters long",
				"password"
			);
			isValid = false;
		}

		return isValid;
	}

	// Handle form submission
	if (loginForm) {
		loginForm.addEventListener("submit", async (event) => {
			event.preventDefault();

			const email = document.getElementById("email").value;
			const password = document.getElementById("password").value;

			// Validate form
			if (!validateForm(email, password)) {
				return;
			}

			// Set loading state
			setLoading(true);

			try {
				const response = await fetch("/login", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({ email, password }),
					credentials: "include", // Important for session cookies
				});

				const data = await response.json();

				if (response.ok) {
					// Store user data in localStorage
					if (data.user) {
						localStorage.setItem(
							"user",
							JSON.stringify(data.user)
						);
					}

					// Show success message
					showError(
						"Login successful! Redirecting...",
						"general"
					);

					// Redirect after a short delay
					setTimeout(() => {
						window.location.href = data.redirectUrl || "/";
					}, 1000);
				} else {
					showError(
						data.message || "Login failed. Please try again.",
						"general"
					);
				}
			} catch (error) {
				console.error("Error during login:", error);
				showError(
					"An error occurred during login. Please try again.",
					"general"
				);
			} finally {
				setLoading(false);
			}
		});
	}

	// Add input event listeners for real-time validation
	const emailInput = document.getElementById("email");
	const passwordInput = document.getElementById("password");

	emailInput.addEventListener("input", () => {
		clearErrors();
		if (
			emailInput.value.includes("@") &&
			!isValidEmail(emailInput.value)
		) {
			showError("Please enter a valid email address", "email");
		}
	});

	passwordInput.addEventListener("input", () => {
		clearErrors();
		if (passwordInput.value.length < 6) {
			showError(
				"Password must be at least 6 characters long",
				"password"
			);
		}
	});

	// Add focus event listeners to clear errors
	emailInput.addEventListener("focus", () => {
		emailError.style.display = "none";
	});

	passwordInput.addEventListener("focus", () => {
		passwordError.style.display = "none";
	});
});
