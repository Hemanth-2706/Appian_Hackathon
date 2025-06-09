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

		// Get session data
		const sessionResponse = await fetch("/session-debug");
		const sessionData = await sessionResponse.json();

		// Clear messages container
		messages.innerHTML = "";

		// Check if we have chat history
		if (sessionData.chatHistory) {
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
	appendMessage(getRandomResponse("processing"), "bot");

	try {
		// Add a minimal delay to show the thinking message
		await new Promise((resolve) => setTimeout(resolve, 500));

		const response = await fetch("/chatbot/answer", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ answer: text }),
		});

		const data = await response.json();

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
				window.location.href = "/recommend";
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
		console.error("Error sending message:", error);
		appendMessage(
			"Sorry, there was an error processing your answer. Please try again.",
			"bot"
		);
	}
}

// Function to speak text
function speakText(text) {
	console.log('speakText: Input text for speech:', text); // Debug log
	// Stop any ongoing speech
	window.speechSynthesis.cancel();
	
	// Remove markdown bullet points for speech
	const cleanText = text.replace(/\*/g, '');
	const utterance = new SpeechSynthesisUtterance(cleanText);
	
	// Set voice properties
	utterance.rate = 1.0;
	utterance.pitch = 1.1; // Slightly higher pitch for female voice
	utterance.volume = 1.0;
	
	// Get available voices
	let voices = window.speechSynthesis.getVoices();
	
	// If voices aren't loaded yet, wait for them
	if (voices.length === 0) {
		console.log('speakText: No voices available yet, waiting...'); // Debug log
		window.speechSynthesis.onvoiceschanged = () => {
			voices = window.speechSynthesis.getVoices();
			console.log('speakText: Voices loaded:', voices); // Debug log
			setVoiceAndSpeak(utterance, voices);
		};
	} else {
		setVoiceAndSpeak(utterance, voices);
	}
}

// Helper function to set voice and speak
function setVoiceAndSpeak(utterance, voices) {
	console.log('setVoiceAndSpeak: Setting voice and initiating speech.'); // Debug log
	// Try to get a female voice
	const femaleVoice = voices.find(voice => 
		voice.name.includes('female') || 
		voice.name.includes('Female') || 
		voice.name.includes('Samantha') || 
		voice.name.includes('Google UK English Female')
	);
	
	if (femaleVoice) {
		console.log('setVoiceAndSpeak: Using female voice:', femaleVoice.name); // Debug log
		utterance.voice = femaleVoice;
	} else {
		console.log('setVoiceAndSpeak: No specific female voice found, using default.'); // Debug log
	}
	
	// Add some emotion by varying the pitch and rate
	utterance.onboundary = (event) => {
		if (event.name === 'sentence') {
			// Vary pitch slightly for each sentence
			utterance.pitch = 1.0 + Math.random() * 0.2;
		}
	};
	
	// Ensure speech synthesis is working
	try {
		window.speechSynthesis.speak(utterance);
		console.log('setVoiceAndSpeak: Speech initiated.'); // Debug log
	} catch (error) {
		console.error('setVoiceAndSpeak: Speech synthesis error:', error); // Debug log
	}
}

// Function to convert markdown bullet points to HTML
function convertMarkdownToHtml(text) {
	console.log('convertMarkdownToHtml: Input text:', text); // Debug log
	const lines = text.split('\n');
	let htmlContent = '';
	let inList = false;

	lines.forEach(line => {
		const trimmedLine = line.trim();
		if (trimmedLine.startsWith('*')) {
			// If not currently in a list, start a new one
			if (!inList) {
				htmlContent += '<ul>';
				inList = true;
			}
			// Add the list item
			htmlContent += `<li>${trimmedLine.substring(1).trim()}</li>`;
		} else {
			// If currently in a list, close it before adding non-list content
			if (inList) {
				htmlContent += '</ul>';
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
		htmlContent += '</ul>';
	}
	console.log('convertMarkdownToHtml: Output HTML:', htmlContent); // Debug log
	return htmlContent;
}

// Function to append message to chat
function appendMessage(message, sender) {
	console.log('appendMessage: Message content:', message); // Debug log
	console.log('appendMessage: Sender:', sender); // Debug log

	const messageDiv = document.createElement("div");
	messageDiv.className = `message ${sender}-message`;
	
	// Convert markdown bullet points to HTML and set as innerHTML
	messageDiv.innerHTML = convertMarkdownToHtml(message);
	
	messages.appendChild(messageDiv);
	messages.scrollTop = messages.scrollHeight;
	
	// Speak the message if it's from the bot
	if (sender === 'bot') {
		// Add a small delay to ensure the message is displayed before speaking
		setTimeout(() => {
			speakText(message);
		}, 500); // Increased delay to ensure everything is ready
	}
	
	return messageDiv;
}

// Initialize speech synthesis (only to trigger voices to load if not already)
document.addEventListener('DOMContentLoaded', () => {
	console.log('DOMContentLoaded: Initializing speech synthesis.'); // Debug log
	window.speechSynthesis.getVoices(); // This call often prompts the browser to load voices
	console.log('DOMContentLoaded: getVoices() called.'); // Debug log
});

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
				// Get and display next question
				fetch("/chatbot/current-question")
					.then((response) => response.json())
					.then((questionData) => {
						if (questionData.success) {
							appendMessage(questionData.question, "bot");
						}
					})
					.catch((error) => {
						console.error(
							"Error getting next question:",
							error
						);
					});
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

		// Redirect to recommendations page immediately
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
