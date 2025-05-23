const express = require("express");
const path = require("path");
const fs = require("fs");
const router = express.Router();

router.use(express.json()); // Needed to parse JSON body
router.use(express.urlencoded({ extended: true }));
const products = require("../data/product"); // Adjust path as needed

// Home Route - loads product data and banner images
router.get("/", (req, res) => {
	const bannerDir = path.join(
		__dirname,
		"../../client/public/images/banner"
	);
	let bannerImages = [];

	try {
		bannerImages = fs.readdirSync(bannerDir);
	} catch (err) {
		console.error("Error reading banner images folder:", err);
	}

	res.render("home", { products, bannerImages });
});

router.get("/recommend", (req, res) => {
	res.render("recommendations", { recommended: products });
});

router.get("/product/:id", (req, res) => {
	const product = products.find((p) => p.productId === req.params.id);
	if (!product) {
		return res.status(404).send("Product not found");
	}
	res.render("product", { product });
});

router.get("/cart", (req, res) => {
	console.log("get to /cart triggered");

	const sessionCart = req.session.cart || [];

	const cartItems = sessionCart.map((cartItem) => {
		const product = products.find(
			(p) => p.productId === cartItem.productId
		);
		return {
			...product,
			quantity: cartItem.quantity,
		};
	});

	res.render("cart", { cartItems });
});

router.get("/checkout/:id", (req, res) => {
	const product = products.find((p) => p.productId === req.params.id);
	res.render("payment", { product });
});

router.post("/cart/add", (req, res) => {
	console.log("post to /cart/add triggered");
	const { productId, quantity } = req.body;

	// Initialize cart if not present
	if (!req.session.cart) {
		req.session.cart = [];
	}

	// Add or update product
	const index = req.session.cart.findIndex(
		(item) => item.productId === productId
	);
	if (index !== -1) {
		req.session.cart[index].quantity += parseInt(quantity);
	} else {
		req.session.cart.push({ productId, quantity: parseInt(quantity) });
	}
	console.log("Updated cart:", req.session.cart);
	res.redirect("/cart");
});

router.post("/cart/remove", (req, res) => {
	console.log("post to /cart/remove triggered");
	const { productId } = req.body;

	if (!req.session.cart) {
		req.session.cart = [];
	}

	req.session.cart = req.session.cart.filter(
		(item) => item.productId !== productId
	);

	console.log("Updated cart after removal:", req.session.cart);
	res.json({ success: true, message: "Item removed from cart" });
});

module.exports = router;
