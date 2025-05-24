console.log("Loaded chatbot.js");
const chatbot = document.getElementById("chatbot-float");
const openBtn = document.getElementById("chatbot-open");
const closeBtn = document.getElementById("chatbot-close");
const sendBtn = document.getElementById("chatbot-send");
const input = document.getElementById("chatbot-input");
const messages = document.getElementById("chatbot-messages");

openBtn.onclick = () => (chatbot.style.display = "flex");

document.addEventListener("DOMContentLoaded", () => {
	chatbot.style.display = "flex"; // make it visible first
	setTimeout(() => {
		chatbot.classList.add("show"); // trigger animation
	}, 50); // slight delay ensures animation applies
});

sendBtn.onclick = sendMessage;

input.addEventListener("keydown", (e) => {
	if (e.key === "Enter" && !e.shiftKey) {
		e.preventDefault();
		sendMessage();
	}
});

function sendMessage() {
	const text = input.value.trim();
	if (!text) return;
	appendMessage(text, "user");
	input.value = "";
	setTimeout(() => {
		appendMessage("That's interesting! Tell me more.", "bot");
	}, 800);
}

function appendMessage(text, sender) {
	const msg = document.createElement("div");
	msg.className = sender + "-message";
	msg.textContent = text;
	messages.appendChild(msg);
	messages.scrollTop = messages.scrollHeight;
}

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

// Chatbot Image Upload Feature$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

const imageBtn = document.getElementById("chatbot-image-btn");
const imageInput = document.getElementById("chatbot-image-upload");

imageBtn.addEventListener("click", () => {
	imageInput.click();
});

imageInput.addEventListener("change", (e) => {
	const file = e.target.files[0];
	if (file && file.type.startsWith("image/")) {
		const reader = new FileReader();
		reader.onload = function (event) {
			appendImage(event.target.result, "user");
			// Send image to server to store in session (optional)
			fetch("/chatbot/image", {
				method: "POST",
				body: JSON.stringify({ image: event.target.result }),
				headers: {
					"Content-Type": "application/json",
				},
			}).catch((err) => console.error("Upload error", err));
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
}
