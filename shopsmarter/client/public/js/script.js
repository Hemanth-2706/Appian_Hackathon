console.log("Loaded script.js");
function search() {
	const query = document.getElementById("searchInput").value;
	if (query) {
		alert("Search triggered for: " + query);
		// You can redirect to `/search?q=${query}` if implemented
	}
}

function uploadImage() {
	const input = document.getElementById("imageInput");
	input.click();
	input.onchange = () => {
		const file = input.files[0];
		if (file) {
			alert("Image uploaded: " + file.name);
			// Logic to send to backend for visual search
		}
	};
}

const track = document.getElementById("carouselTrack");
const slides = document.querySelectorAll(".carousel-slide");

if (slides.length > 0 && track) {
	let currentSlide = 0;

	function updateSlidePosition() {
		track.style.transform = `translateX(-${currentSlide * 100}%)`;
	}

	function nextSlide() {
		currentSlide = (currentSlide + 1) % slides.length;
		updateSlidePosition();
	}

	function prevSlide() {
		currentSlide = (currentSlide - 1 + slides.length) % slides.length;
		updateSlidePosition();
	}

	// Auto-slide
	setInterval(() => {
		nextSlide();
	}, 5000);
}

const buttons = document.querySelectorAll(".add-to-cart-btn");
const toast = document.getElementById("toast");
const cartIcon = document.getElementById("cart-icon"); // Ensure your cart icon has this ID

buttons.forEach((btn) => {
	btn.addEventListener("click", async () => {
		const productId = btn.dataset.productId;
		// Send add to cart request with quantity = 1
		try {
			await fetch("/cart/add", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({ productId, quantity: 1 }), // <-- added quantity here
			});
			console.log("Product added to cart:", productId);
			// Show toast
			toast.classList.add("show");
			setTimeout(() => toast.classList.remove("show"), 2000);

			// Jingle cart
			if (cartIcon) {
				cartIcon.classList.add("jingle");
				setTimeout(() => cartIcon.classList.remove("jingle"), 500);
			}
		} catch (err) {
			console.error("Add to cart failed:", err);
		}
	});
});
