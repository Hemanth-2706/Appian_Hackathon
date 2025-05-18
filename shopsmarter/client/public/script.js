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

let currentSlide = 0;
const track = document.getElementById("carouselTrack");
const slides = document.querySelectorAll(".carousel-slide");

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
