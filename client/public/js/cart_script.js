console.log("Loaded cart_script.js");

function showToast(message = "Notification", type = "success") {
	const toast = document.getElementById("toast");
	if (!toast) return;

	toast.textContent = message;
	toast.className = `toast ${type}`;
	toast.classList.add("show");
	setTimeout(() => toast.classList.remove("show"), 3000);
}

// Handle quantity updates
document.querySelectorAll(".quantity-dropdown").forEach((select) => {
	// Store the initial value
	select.setAttribute("data-previous-value", select.value);

	select.addEventListener("change", async function () {
		const productId = this.getAttribute("data-product-id");
		const quantity = parseInt(this.value);

		try {
			const response = await fetch("/cart/set-quantity", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					productId,
					quantity,
				}),
			});

			const data = await response.json();

			if (data.success) {
				showToast(`Quantity updated to ${quantity}`);

				// Update the item total
				const cartItem = this.closest(".cart-item");
				const priceElement = cartItem.querySelector(".item-price");
				const totalElement =
					cartItem.querySelector(".item-total p");

				const price = parseFloat(
					priceElement.textContent.replace("Price: ₹", "")
				);

				totalElement.textContent = `Total: ₹${(
					price * quantity
				).toFixed(2)}`;

				// Update the cart total
				updateCartTotal();

				// Save current value as the new previous
				this.setAttribute("data-previous-value", this.value);
			} else {
				throw new Error(
					data.message || "Failed to update quantity"
				);
			}
		} catch (error) {
			showToast(error.message || "Error updating quantity", "error");
			console.error("Error:", error);

			// Reset to previous value
			this.value = this.getAttribute("data-previous-value") || "1";
		}
	});
});

// Handle remove buttons
document.querySelectorAll(".remove-btn").forEach((button) => {
	button.addEventListener("click", async function () {
		const productId = this.getAttribute("data-product-id");
		const cartItem = this.closest(".cart-item");

		try {
			const response = await fetch("/cart/remove", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({ productId }),
			});

			const data = await response.json();

			if (data.success) {
				showToast("Item removed from cart");
				cartItem.remove();

				// Update the cart total
				updateCartTotal();

				// Check if cart is empty
				const remainingItems =
					document.querySelectorAll(".cart-item");
				if (remainingItems.length === 0) {
					location.reload(); // Reload to show empty cart state
				}
			} else {
				throw new Error(data.message || "Failed to remove item");
			}
		} catch (error) {
			showToast(error.message || "Error removing item", "error");
			console.error("Error:", error);
		}
	});
});

// Function to update cart total
function updateCartTotal() {
	const cartItems = document.querySelectorAll(".cart-item");
	let total = 0;

	cartItems.forEach((item) => {
		const priceElement = item.querySelector(".item-price");
		const quantityElement = item.querySelector(".quantity-dropdown");
		const price = parseFloat(
			priceElement.textContent.replace("Price: ₹", "")
		);
		const quantity = parseInt(quantityElement.value);
		total += price * quantity;
	});

	const totalElement = document.querySelector(
		".total-price span:last-child"
	);
	if (totalElement) {
		totalElement.textContent = `₹${total.toFixed(2)}`;
	}
}

// Handle checkout button
document.querySelector(".checkout-btn")?.addEventListener("click", function () {
	// Add your checkout logic here
	showToast("Proceeding to checkout...");
});
