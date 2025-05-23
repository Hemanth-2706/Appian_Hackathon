console.log("Loaded cart_script.js");
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
					alert(data.message);
					button.closest(".cart-item").remove();
				} else {
					alert("Failed to remove item from cart.");
				}
			})
			.catch((err) => {
				console.error("Error:", err);
				alert("Server error");
			});
	});
});
