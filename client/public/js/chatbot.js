console.log("Loaded chatbot.js");
const chatbot = document.getElementById("chatbot-float");
const openBtn = document.getElementById("chatbot-open");
const closeBtn = document.getElementById("chatbot-close");
const sendBtn = document.getElementById("chatbot-send");
const input = document.getElementById("chatbot-input");
const messages = document.getElementById("chatbot-messages");

// Initial bot responses
const initialResponses = [
	"Hey there! 👋 I'm your personal shopping assistant at HSM ShopSmarter 🛍️",
	"I can help you find exactly what you're looking for! Just 📸 upload a photo or ✍️ describe what you want, and I'll find the best options for you ⚡✨",
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

// IMPROVED TEXT-TO-SPEECH FUNCTIONS
// Track if we're currently speaking to avoid interruptions
let currentlySpeaking = false;

// Function to speak text with better reliability
function speakText(text) {
	// If already speaking, queue this instead of interrupting
	if (currentlySpeaking) {
		setTimeout(() => speakText(text), 1000);
		return;
	}

	// Stop any ongoing speech
	window.speechSynthesis.cancel();

	// Remove markdown formatting and clean text
	const cleanText = text
		.replace(/\*/g, "") // Remove asterisks
		.replace(/<[^>]*>/g, "") // Remove HTML tags
		.replace(/[🔥⚡✨👋🛍️📸✍️]/g, "") // Remove emojis that might cause issues
		.trim();

	// Don't speak if text is empty after cleaning
	if (!cleanText) return;

	console.log("Speaking text:", cleanText); // Debug log

	const utterance = new SpeechSynthesisUtterance(cleanText);

	// Set voice properties
	utterance.rate = 0.9; // Slightly slower for better clarity
	utterance.pitch = 1.1;
	utterance.volume = 1.0;

	// Add error handling
	utterance.onerror = (event) => {
		currentlySpeaking = false;
		// Only log non-interruption errors to reduce console noise
		if (event.error !== "interrupted") {
			console.error("Speech synthesis error:", event.error);
			// Retry once if it fails (but not for interruptions)
			setTimeout(() => {
				const retryUtterance = new SpeechSynthesisUtterance(
					cleanText
				);
				retryUtterance.rate = 0.9;
				retryUtterance.pitch = 1.1;
				retryUtterance.volume = 1.0;
				window.speechSynthesis.speak(retryUtterance);
			}, 500);
		}
	};

	utterance.onstart = () => {
		currentlySpeaking = true;
		console.log("Speech started");
	};

	utterance.onend = () => {
		currentlySpeaking = false;
		console.log("Speech ended");
	};

	// Get voices and set appropriate voice
	const voices = window.speechSynthesis.getVoices();
	setVoiceAndSpeak(utterance, voices);
}

// Improved helper function to set voice and speak
function setVoiceAndSpeak(utterance, voices) {
	// If no voices available, wait and try again
	if (voices.length === 0) {
		console.log("No voices available, waiting...");
		setTimeout(() => {
			const newVoices = window.speechSynthesis.getVoices();
			if (newVoices.length > 0) {
				setVoiceAndSpeak(utterance, newVoices);
			} else {
				// Fallback: speak without specific voice
				window.speechSynthesis.speak(utterance);
			}
		}, 100);
		return;
	}

	// Try to get a good female voice (prioritize quality voices)
	const preferredVoices = [
		"Google UK English Female",
		"Samantha",
		"Microsoft Zira - English (United States)",
		"Karen",
		"Moira",
		"Tessa",
	];

	let selectedVoice = null;

	// First try to find preferred voices
	for (const preferredName of preferredVoices) {
		selectedVoice = voices.find((voice) =>
			voice.name.includes(preferredName)
		);
		if (selectedVoice) break;
	}

	// If no preferred voice found, try to find any female voice
	if (!selectedVoice) {
		selectedVoice = voices.find(
			(voice) =>
				voice.name.toLowerCase().includes("female") ||
				voice.name.toLowerCase().includes("woman") ||
				voice.gender === "female"
		);
	}

	// If still no voice found, use the default voice
	if (!selectedVoice && voices.length > 0) {
		selectedVoice = voices[0];
	}

	if (selectedVoice) {
		utterance.voice = selectedVoice;
		console.log("Using voice:", selectedVoice.name);
	}

	// Ensure speech synthesis is working with additional checks
	try {
		// Check if speech synthesis is supported
		if ("speechSynthesis" in window) {
			// Small delay to ensure everything is ready
			setTimeout(() => {
				window.speechSynthesis.speak(utterance);
			}, 50);
		} else {
			console.error("Speech synthesis not supported");
		}
	} catch (error) {
		console.error("Speech synthesis error:", error);
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

// Enhanced function to append message to chat
function appendMessage(message, sender) {
	const messageDiv = document.createElement("div");
	messageDiv.className = `message ${sender}-message`;

	// Convert markdown bullet points to HTML and set as innerHTML
	messageDiv.innerHTML = convertMarkdownToHtml(message);

	messages.appendChild(messageDiv);
	messages.scrollTop = messages.scrollHeight;

	// Enhanced TTS for bot messages - removed previous setTimeout and simplified
	if (sender === "bot") {
		// Double-check that speech synthesis is available
		if ("speechSynthesis" in window && window.speechSynthesis) {
			speakText(message);
		} else {
			console.warn("Speech synthesis not available");
		}
	}

	return messageDiv;
}

// Enhanced initialization for speech synthesis
function initializeSpeechSynthesis() {
	if ("speechSynthesis" in window) {
		// Load voices immediately
		const voices = window.speechSynthesis.getVoices();
		console.log("Available voices:", voices.length);

		// Set up voice loading event
		window.speechSynthesis.onvoiceschanged = () => {
			const newVoices = window.speechSynthesis.getVoices();
			console.log("Voices loaded:", newVoices.length);
		};

		// Force voice loading (some browsers need this)
		if (voices.length === 0) {
			// Trigger voice loading
			const testUtterance = new SpeechSynthesisUtterance("");
			window.speechSynthesis.speak(testUtterance);
			window.speechSynthesis.cancel();
		}
	} else {
		console.error("Speech synthesis not supported in this browser");
	}
}

// Initialize speech synthesis when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
	initializeSpeechSynthesis();
});

// Export for debugging (you can call this from browser console)
window.debugTTS = {
	speak: speakText,
	test: () => speakText("This is a test of the text to speech system."),
	voices: () => window.speechSynthesis.getVoices(),
	cancel: () => window.speechSynthesis.cancel(),
};

// Chatbot Image Upload Feature
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

		// Remove thinking message
		messages.removeChild(thinkingMsg);

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
		window.location.href = "/recommend";
	} catch (error) {
		// Remove thinking message on error
		messages.removeChild(thinkingMsg);
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
