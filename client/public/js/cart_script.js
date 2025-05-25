console.log("Loaded cart_script.js");

function showToast(message = "Notification") {
	const toast = document.getElementById("toast");
	if (!toast) return;

	toast.textContent = message;
	toast.classList.add("show");
	setTimeout(() => toast.classList.remove("show"), 2000);
}

document.querySelectorAll(".remove-btn").forEach((button) => {
	button.addEventListener("click", (event) => {
		event.preventDefault();

		const form = button.closest("form");
		const formData = new FormData(form);

		fetch(form.action, {
			method: "POST",
			body: formData,
		})
			.then((res) => res.json())
			.then((data) => {
				if (data.success) {
					showToast(data.message || "Item removed from cart.");
					button.closest(".cart-item").remove();
				} else {
					showToast("Failed to remove item from cart.");
				}
			})
			.catch((err) => {
				console.error("Error:", err);
				showToast("Server error. Try again.");
			});
	});
});
