console.log("Loaded recommend.js");
document.addEventListener("DOMContentLoaded", function () {
	// Handle individual product add to cart
	document.querySelectorAll(".add-to-cart").forEach((button) => {
		button.addEventListener("click", async function () {
			const productId = this.getAttribute("data-product-id");
			const quantitySelect = document.getElementById(
				`quantity-${productId}`
			);
			const quantity = parseInt(quantitySelect.value);

			// Disable button during request
			this.disabled = true;
			this.textContent = "Adding...";

			try {
				const response = await fetch("/cart/add", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({
						productId: productId,
						quantity: quantity,
					}),
				});

				const data = await response.json();

				if (response.ok && data.success) {
					showNotification(
						`Added ${quantity} item(s) to cart!`,
						"success"
					);
					// Reset quantity to 1 after successful addition
					quantitySelect.value = "1";
				} else {
					throw new Error(
						data.message || "Failed to add product to cart"
					);
				}
			} catch (error) {
				showNotification(
					error.message || "Error adding product to cart",
					"error"
				);
				console.error("Error:", error);
			} finally {
				// Re-enable button
				this.disabled = false;
				this.textContent = "Add to Cart";
			}
		});
	});

	// Handle add all to cart
	document
		.getElementById("add-all-to-cart")
		?.addEventListener("click", async function () {
			const products = Array.from(
				document.querySelectorAll(".add-to-cart")
			).map((button) => {
				const productId = button.getAttribute("data-product-id");
				const quantity = parseInt(
					document.getElementById(`quantity-${productId}`).value
				);
				return { productId, quantity };
			});

			// Disable button during request
			this.disabled = true;
			this.textContent = "Adding...";

			try {
				const response = await fetch("/cart/add-multiple", {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
					},
					body: JSON.stringify({ products }),
				});

				const data = await response.json();

				if (response.ok && data.success) {
					showNotification(
						"All products added to cart successfully!",
						"success"
					);
					// Reset all quantities to 1
					document
						.querySelectorAll(".quantity-dropdown")
						.forEach((select) => {
							select.value = "1";
						});
				} else {
					throw new Error(
						data.message ||
							"Failed to add all products to cart"
					);
				}
			} catch (error) {
				showNotification(
					error.message || "Error adding products to cart",
					"error"
				);
				console.error("Error:", error);
			} finally {
				// Re-enable button
				this.disabled = false;
				this.textContent = "Add All to Cart";
			}
		});

	function showNotification(message, type) {
		const notification = document.createElement("div");
		notification.className = `alert alert-${
			type === "success" ? "success" : "danger"
		} notification`;
		notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px;
            border-radius: 5px;
            z-index: 1000;
            background-color: ${type === "success" ? "#28a745" : "#dc3545"};
            color: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.5s ease-out;
        `;
		notification.textContent = message;

		document.body.appendChild(notification);

		setTimeout(() => {
			notification.style.animation = "slideOut 0.5s ease-in";
			setTimeout(() => {
				notification.remove();
			}, 500);
		}, 3000);
	}

	// Add CSS for animations if not already present
	if (!document.getElementById("toast-animations")) {
		const style = document.createElement("style");
		style.id = "toast-animations";
		style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); }
                to { transform: translateX(0); }
            }
            @keyframes slideOut {
                from { transform: translateX(0); }
                to { transform: translateX(100%); }
            }
        `;
		document.head.appendChild(style);
	}
});
