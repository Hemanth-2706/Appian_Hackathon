console.log("Loaded product.js");

document.addEventListener("DOMContentLoaded", function () {
	const addToCartButtons = document.querySelectorAll(".add-to-cart");

	console.log("Add to Cart Buttons found:", addToCartButtons.length);
	console.log("Add to Cart Buttons:", addToCartButtons);

	if (addToCartButtons.length === 0) {
		console.error("No Add to Cart buttons found!");
		return;
	}

	// Add event listener to each button
	addToCartButtons.forEach(function (button) {
		button.addEventListener("click", async function () {
			const productId = this.getAttribute("data-product-id");

			console.log("Button clicked for product:", productId);

			// Find the corresponding quantity select for this specific product
			let quantitySelect;

			// Check if this is the main product (has id="quantity")
			if (
				document.getElementById("quantity") &&
				this.closest(".product-detail-card")
			) {
				quantitySelect = document.getElementById("quantity");
				console.log("Using main product quantity selector");
			} else {
				// For similar/recommended products, find the specific quantity select
				quantitySelect = document.getElementById(
					`quantity-${productId}`
				);
				console.log(
					"Using product-specific quantity selector:",
					`quantity-${productId}`
				);
			}

			if (!quantitySelect) {
				console.error(
					`Quantity select not found for product ${productId}!`
				);
				showNotification(
					"Error: Could not find quantity selector",
					"error"
				);
				return;
			}

			const quantity = parseInt(quantitySelect.value);

			console.log("Adding to cart:", { productId, quantity });

			// Disable button during request to prevent double-clicks
			this.disabled = true;
			const originalText = this.textContent;
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

				console.log("Server response:", response);

				if (response.ok) {
					const data = await response.json();
					console.log("Response data:", data);
					showNotification(
						`Added ${quantity} item(s) to cart!`,
						"success"
					);
					// Reset quantity to 1 after successful addition
					quantitySelect.value = "1";
				} else {
					const errorData = await response.text();
					console.error("Server error:", errorData);
					throw new Error("Failed to add product to cart");
				}
			} catch (error) {
				console.error("Error details:", error);
				showNotification("Error adding product to cart", "error");
			} finally {
				// Re-enable button
				this.disabled = false;
				this.textContent = originalText;
			}
		});
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
