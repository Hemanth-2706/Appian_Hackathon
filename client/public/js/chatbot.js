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

// Speech synthesis variables
let speechQueue = [];
let isSpeaking = false;
let userHasInteracted = false;
let voicesLoaded = false;
let preferredVoice = null;

// Debug speech synthesis capabilities
console.log("Speech synthesis supported:", "speechSynthesis" in window);
console.log(
	"Speech synthesis enabled:",
	window.speechSynthesis && !window.speechSynthesis.speaking
);

// Load voices immediately and set up voice loading
function loadVoices() {
	const voices = window.speechSynthesis.getVoices();
	console.log("Available voices:", voices.length);

	if (voices.length > 0) {
		preferredVoice =
			voices.find(
				(voice) =>
					voice.name.includes("female") ||
					voice.name.includes("Female") ||
					voice.name.includes("Samantha") ||
					voice.name.includes("Google UK English Female")
			) || voices[0];
		voicesLoaded = true;
		console.log("Voices loaded, preferred voice:", preferredVoice?.name);
	} else {
		console.log("No voices available yet");
	}
}

// Initialize speech synthesis with better voice loading
if ("speechSynthesis" in window) {
	// Try to load voices immediately
	loadVoices();

	// Set up the voices changed event
	window.speechSynthesis.onvoiceschanged = () => {
		console.log("Voices changed event fired");
		loadVoices();
	};

	// Fallback: try to load voices after a short delay
	setTimeout(() => {
		if (!voicesLoaded) {
			console.log("Fallback voice loading");
			loadVoices();
		}
	}, 100);
}

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

		// Get session data
		const sessionResponse = await fetch("/session-debug");
		const sessionData = await sessionResponse.json();

		// Clear messages container
		messages.innerHTML = "";

		// Check if we have chat history
		if (
			sessionData.chatHistory.image ||
			sessionData.chatHistory.userText
		) {
			console.log(
				"sessionData.chatHistory =",
				sessionData.chatHistory
			);
			// Display image if exists
			if (sessionData.chatHistory.image) {
				appendImage(sessionData.chatHistory.image.content, "user");
			}
			// Display text if exists
			if (sessionData.chatHistory.userText) {
				appendMessage(
					sessionData.chatHistory.userText.content,
					"user"
				);
			}

			// If we have a meaningful caption from recommendations, display it
			if (sessionData.recommendationResults?.meaningfulCaption) {
				appendMessage(
					sessionData.recommendationResults.meaningfulCaption,
					"bot"
				);
			}
		} else {
			console.log("Chatbot is empty");
			// Show initial messages if no history
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
		}, 500);
	} catch (error) {
		console.error("Error initializing chat:", error);
		// Show initial messages even if there's an error
		messages.innerHTML = "";
		initialResponses.forEach((response) => {
			appendMessage(response, "bot");
		});
	}
});

// Track user interaction for speech synthesis
document.addEventListener("click", () => {
	userHasInteracted = true;
	console.log("User interaction detected - speech synthesis enabled");
});

document.addEventListener("keydown", () => {
	userHasInteracted = true;
	console.log("User interaction detected - speech synthesis enabled");
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

	// Store text in session
	try {
		await fetch("/chatbot/text", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ text }),
		});
	} catch (error) {
		console.error("Error storing text:", error);
	}
}

// Function to ensure voices are loaded before speaking
function ensureVoicesLoaded() {
	return new Promise((resolve) => {
		if (voicesLoaded && preferredVoice) {
			resolve();
			return;
		}

		const voices = window.speechSynthesis.getVoices();
		if (voices.length > 0) {
			loadVoices();
			resolve();
			return;
		}

		// Wait for voices to load
		const checkVoices = () => {
			const voices = window.speechSynthesis.getVoices();
			if (voices.length > 0) {
				loadVoices();
				resolve();
			} else {
				setTimeout(checkVoices, 50);
			}
		};

		// Trigger voice loading
		window.speechSynthesis.onvoiceschanged = () => {
			loadVoices();
			resolve();
		};

		checkVoices();
	});
}

// Function to speak text - IMPROVED VERSION
async function speakText(text, speakerBtn) {
	if (!text || typeof text !== "string") {
		console.log("Invalid text for speech");
		return;
	}

	// Clean text for speech
	const cleanText = text.replace(/[*#]/g, "").replace(/\s+/g, " ").trim();
	console.log("Speaking text:", cleanText);

	try {
		// Ensure voices are loaded before speaking
		await ensureVoicesLoaded();

		// Cancel any existing speech
		window.speechSynthesis.cancel();

		// Small delay to ensure cancel has taken effect
		await new Promise((resolve) => setTimeout(resolve, 100));

		const utterance = new SpeechSynthesisUtterance(cleanText);

		// Set voice properties
		utterance.rate = 1.0;
		utterance.pitch = 1.1;
		utterance.volume = 1.0;

		// Set preferred voice if available
		if (preferredVoice) {
			utterance.voice = preferredVoice;
			console.log("Using voice:", preferredVoice.name);
		}

		// Set up event handlers
		utterance.onstart = () => {
			console.log("Speech started");
			speakerBtn.innerHTML = "🔇";
			speakerBtn.classList.add("speaking");
			speakerBtn.isSpeaking = true;
		};

		utterance.onend = () => {
			console.log("Speech ended");
			speakerBtn.innerHTML = "🔊";
			speakerBtn.classList.remove("speaking");
			speakerBtn.isSpeaking = false;
		};

		utterance.onerror = (event) => {
			console.log("Speech error:", event.error);
			speakerBtn.innerHTML = "🔊";
			speakerBtn.classList.remove("speaking");
			speakerBtn.isSpeaking = false;
		};

		// Speak the text
		window.speechSynthesis.speak(utterance);
	} catch (error) {
		console.error("Error in speakText:", error);
		speakerBtn.innerHTML = "🔊";
		speakerBtn.classList.remove("speaking");
		speakerBtn.isSpeaking = false;
	}
}

// Function to convert markdown bullet points to HTML
function convertMarkdownToHtml(text) {
	const lines = text.split("\n");
	let htmlContent = "";
	let inList = false;

	lines.forEach((line) => {
		const trimmedLine = line.trim();
		if (trimmedLine.startsWith("*")) {
			// If not currently in a list, start a new one
			if (!inList) {
				htmlContent += "<ul>";
				inList = true;
			}
			// Add the list item
			htmlContent += `<li>${trimmedLine.substring(1).trim()}</li>`;
		} else {
			// If currently in a list, close it before adding non-list content
			if (inList) {
				htmlContent += "</ul>";
				inList = false;
			}
			// Add non-list content as a paragraph, if it's not an empty line
			if (trimmedLine) {
				htmlContent += `<p>${trimmedLine}</p>`;
			}
		}
	});

	// If the text ended with a list, close the list tag
	if (inList) {
		htmlContent += "</ul>";
	}

	return htmlContent;
}

// Function to append message to chat - IMPROVED VERSION
function appendMessage(message, sender) {
	userHasInteracted = true;
	const messageDiv = document.createElement("div");
	messageDiv.className = `message ${sender}-message`;

	// Convert markdown bullet points to HTML and set as innerHTML
	messageDiv.innerHTML = convertMarkdownToHtml(message);

	// Add speaker button for bot messages
	if (sender === "bot") {
		const speakerBtn = document.createElement("button");
		speakerBtn.className = "speaker-btn";
		speakerBtn.innerHTML = "🔊";
		speakerBtn.title = "Read aloud";
		speakerBtn.isSpeaking = false;

		speakerBtn.addEventListener("click", async () => {
			if (speakerBtn.isSpeaking) {
				// Stop speaking
				console.log("Stopping speech");
				window.speechSynthesis.cancel();
				speakerBtn.innerHTML = "🔊";
				speakerBtn.classList.remove("speaking");
				speakerBtn.isSpeaking = false;
				return;
			}

			// Start speaking
			console.log("Starting speech");
			speakerBtn.innerHTML = "🔇";
			speakerBtn.classList.add("speaking");
			speakerBtn.isSpeaking = true;

			// Cancel any existing speech first
			window.speechSynthesis.cancel();

			// Small delay to ensure cancel has taken effect, then speak
			setTimeout(() => {
				speakText(message, speakerBtn);
			}, 100);
		});

		messageDiv.appendChild(speakerBtn);
	}

	messages.appendChild(messageDiv);
	messages.scrollTop = messages.scrollHeight;

	return messageDiv;
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
			appendMessage(getRandomResponse("imageUpload"), "bot");

			// Send image to server to store in session
			try {
				await fetch("/chatbot/image", {
					method: "POST",
					body: JSON.stringify({ image: event.target.result }),
					headers: {
						"Content-Type": "application/json",
					},
				});
			} catch (err) {
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
	const sessionResponse = await fetch("/session-debug");
	const sessionData = await sessionResponse.json();

	if (
		!sessionData.chatHistory?.image &&
		!sessionData.chatHistory?.userText
	) {
		appendMessage(
			"Please describe what you're looking for or upload an image first!",
			"bot"
		);
		return;
	}

	// Show processing message with animation
	const processingMsg = document.createElement("div");
	processingMsg.className = "bot-message processing-message";
	processingMsg.textContent = "Searching for related products...";
	messages.appendChild(processingMsg);
	messages.scrollTop = messages.scrollHeight;

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

		// Get updated session data to get the meaningful caption
		const updatedSessionResponse = await fetch("/session-debug");
		const updatedSessionData = await updatedSessionResponse.json();

		// Display the meaningful caption if available
		if (updatedSessionData.recommendationResults?.meaningfulCaption) {
			appendMessage(
				updatedSessionData.recommendationResults.meaningfulCaption,
				"bot"
			);
		}

		// Redirect to recommendations page after a short delay
		setTimeout(() => {
			window.location.href = "/recommend";
		}, 1000);
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
