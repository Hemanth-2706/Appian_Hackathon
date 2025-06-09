console.log("Loaded chatbot.js");
const chatbot = document.getElementById("chatbot-float");
const openBtn = document.getElementById("chatbot-open");
const closeBtn = document.getElementById("chatbot-close");
const sendBtn = document.getElementById("chatbot-send");
const input = document.getElementById("chatbot-input");
const messages = document.getElementById("chatbot-messages");

// Initial bot responses
const initialResponses = [
	"👋 Hey there! I'm your personal shopping assistant 🛍️",
	"📸 Upload a photo or ✍️ describe what you're looking for, and I'll find the best options for you in a snap! ⚡✨",
];

// Common responses for different scenarios
const botResponses = {
	greeting: [
		"Hello! How can I help you today?",
		"Hi there! What are you looking for?",
		"Hey! Ready to find your perfect style?",
	],
	imageUpload: [
		"Great image! Let me analyze it for you...",
		"Perfect! I'll find similar styles based on this image.",
		"Thanks for sharing! I'll search for matching items.",
	],
	noInput: [
		"Could you please provide more details?",
		"I'd love to help, but I need more information.",
		"Feel free to describe what you're looking for or upload an image!",
	],
	processing: [
		"Let me search for the perfect match...",
		"Analyzing your request...",
		"Finding the best options for you...",
	],
};

// Initialize chatbot and check session history
document.addEventListener("DOMContentLoaded", async () => {
	try {
		// Initialize empty chat history in session if it doesn't exist
		await fetch("/chatbot/init-session", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
		});

		// Get current question
		const questionResponse = await fetch("/chatbot/current-question");
		const questionData = await questionResponse.json();

		// Clear messages container
		messages.innerHTML = "";

		if (questionData.success) {
			// Display the current question
			appendMessage(questionData.question, "bot");
		} else {
			// Show initial messages if no questions available
			initialResponses.forEach((response) => {
				appendMessage(response, "bot");
			});
		}

		// Show chatbot with animation after a short delay
		setTimeout(() => {
			chatbot.style.display = "flex";
			setTimeout(() => {
				chatbot.classList.add("show");
			}, 50);
		}, 500); // Show after 1 second
	} catch (error) {
		console.error("Error initializing chat:", error);
		// Show initial messages even if there's an error
		messages.innerHTML = "";
		initialResponses.forEach((response) => {
			appendMessage(response, "bot");
		});
	}
});

openBtn.onclick = () => {
	chatbot.style.display = "flex";
	setTimeout(() => {
		chatbot.classList.add("show");
	}, 50);
};

sendBtn.onclick = sendMessage;

input.addEventListener("keydown", (e) => {
	if (e.key === "Enter" && !e.shiftKey) {
		e.preventDefault();
		sendMessage();
	}
});

function getRandomResponse(type) {
	const responses = botResponses[type];
	return responses[Math.floor(Math.random() * responses.length)];
}

async function sendMessage() {
	const text = input.value.trim();
	if (!text) return;

	// Display user's answer
	appendMessage(text, "user");
	input.value = "";

	// Show bot thinking with random processing message
	const thinkingMsg = appendMessage(getRandomResponse("processing"), "bot");

	try {
		// Add a delay to show the thinking message
		await new Promise((resolve) => setTimeout(resolve, 15000));

		const response = await fetch("/chatbot/answer", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ answer: text }),
		});

		const data = await response.json();

		// Remove thinking message
		messages.removeChild(thinkingMsg);

		if (data.success) {
			// If there's a bot response, display it
			if (data.botResponse) {
				appendMessage(data.botResponse, "bot");
			}

			// If the Q&A is complete and we have recommendations
			if (data.isComplete && data.hasRecommendations) {
				appendMessage(
					"Great! Let me show you some recommendations based on your preferences.",
					"bot"
				);
				setTimeout(() => {
					window.location.href = "/recommend";
				}, 2000);
			} else if (!data.isComplete) {
				// Get and display next question
				const questionResponse = await fetch(
					"/chatbot/current-question"
				);
				const questionData = await questionResponse.json();

				if (questionData.success) {
					appendMessage(questionData.question, "bot");
				}
			}
		}
	} catch (error) {
		// Remove thinking message on error
		messages.removeChild(thinkingMsg);
		console.error("Error sending message:", error);
		appendMessage(
			"Sorry, there was an error processing your answer. Please try again.",
			"bot"
		);
	}
}

function appendMessage(text, sender) {
	const msg = document.createElement("div");
	msg.className = sender + "-message";
	msg.textContent = text;
	messages.appendChild(msg);
	messages.scrollTop = messages.scrollHeight;
	return msg;
}

// Chatbot Image Upload Feature$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

const imageBtn = document.getElementById("chatbot-image-btn");
const imageInput = document.getElementById("chatbot-image-upload");

imageBtn.addEventListener("click", () => {
	imageInput.click();
});

imageInput.addEventListener("change", async (e) => {
	const file = e.target.files[0];
	if (file && file.type.startsWith("image/")) {
		const reader = new FileReader();
		reader.onload = async function (event) {
			appendImage(event.target.result, "user");

			// Show bot thinking with random image upload message
			const thinkingMsg = appendMessage(
				getRandomResponse("imageUpload"),
				"bot"
			);

			// Send image to server to store in session
			try {
				await fetch("/chatbot/image", {
					method: "POST",
					body: JSON.stringify({ image: event.target.result }),
					headers: {
						"Content-Type": "application/json",
					},
				});

				// Remove thinking message after a delay
				setTimeout(() => {
					messages.removeChild(thinkingMsg);
					// Get and display next question
					fetch("/chatbot/current-question")
						.then((response) => response.json())
						.then((questionData) => {
							if (questionData.success) {
								appendMessage(
									questionData.question,
									"bot"
								);
							}
						})
						.catch((error) => {
							console.error(
								"Error getting next question:",
								error
							);
						});
				}, 15000);
			} catch (err) {
				messages.removeChild(thinkingMsg);
				console.error("Upload error", err);
				appendMessage(
					"Sorry, there was an error processing your image. Please try again.",
					"bot"
				);
			}
		};
		reader.readAsDataURL(file);
	}
});

function appendImage(imageUrl, sender) {
	const msg = document.createElement("div");
	msg.className = sender + "-message";
	const img = document.createElement("img");
	img.src = imageUrl;
	img.alt = "Uploaded";
	img.style.maxWidth = "200px";
	img.style.borderRadius = "8px";
	msg.appendChild(img);
	messages.appendChild(msg);
	messages.scrollTop = messages.scrollHeight;
	return msg;
}

// Recommendations Button
const recommendBtn = document.getElementById("chatbot-recommend-btn");

recommendBtn.addEventListener("click", async () => {
	// Check if we have either text or image in the session
	if (!document.querySelector(".user-message")) {
		appendMessage(
			"Please describe what you're looking for or upload an image first!",
			"bot"
		);
		return;
	}

	// Show processing message
	appendMessage("Processing your request...", "bot");

	try {
		// Call the process endpoint
		const response = await fetch("/process-recommendations", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
		});

		if (!response.ok) {
			throw new Error("Failed to process recommendations");
		}

		// Redirect to recommendations page
		window.location.href = "/recommend";
	} catch (error) {
		console.error("Error:", error);
		appendMessage(
			"Sorry, there was an error processing your request. Please try again.",
			"bot"
		);
	}
});

// Clear History Button
const clearBtn = document.getElementById("chatbot-clear-btn");

clearBtn.addEventListener("click", async () => {
	try {
		// Call the clear-history endpoint
		const response = await fetch("/chatbot/clear-history", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
		});

		if (!response.ok) {
			throw new Error("Failed to clear history");
		}

		// Clear the messages container
		messages.innerHTML = "";

		// Add initial bot messages back
		initialResponses.forEach((response) => {
			appendMessage(response, "bot");
		});

		appendMessage(
			"History cleared successfully! How can I help you today?",
			"bot"
		);
	} catch (error) {
		console.error("Error:", error);
		appendMessage(
			"Sorry, there was an error clearing the history. Please try again.",
			"bot"
		);
	}
});

// Resizing
const resizeHandle = document.getElementById("chatbot-resize");
let isResizing = false;

resizeHandle.addEventListener("mousedown", (e) => {
	isResizing = true;
	e.preventDefault();
});

window.addEventListener("mousemove", (e) => {
	if (isResizing) {
		chatbot.style.width = e.clientX - chatbot.offsetLeft + "px";
		chatbot.style.height = e.clientY - chatbot.offsetTop + "px";
	}
});

window.addEventListener("mouseup", () => (isResizing = false));

// Dragging
const header = document.getElementById("chatbot-header");
let isDragging = false;
let offsetX = 0;
let offsetY = 0;

header.addEventListener("mousedown", (e) => {
	isDragging = true;
	offsetX = e.clientX - chatbot.offsetLeft;
	offsetY = e.clientY - chatbot.offsetTop;
	e.preventDefault();
});

window.addEventListener("mousemove", (e) => {
	if (isDragging) {
		chatbot.style.left = `${e.clientX - offsetX}px`;
		chatbot.style.top = `${e.clientY - offsetY}px`;
	}
});

window.addEventListener("mouseup", () => {
	isDragging = false;
});

closeBtn.onclick = () => {
	chatbot.classList.remove("show"); // trigger hide animation
	setTimeout(() => {
		chatbot.style.display = "none"; // hide after animation
	}, 400); // match the CSS transition duration
};
