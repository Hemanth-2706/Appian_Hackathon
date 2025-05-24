const express = require("express");
const path = require("path");
const fs = require("fs");

const { products, similarProducts, recommendedProducts } = require(path.join(
	__dirname,
	"../data/products"
));

const router = express.Router();

router.use(express.json()); // Needed to parse JSON body
router.use(express.urlencoded({ extended: true }));

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
	// Pass both similar and recommended products to the template
	res.render("recommendations", {
		similarProducts,
		recommendedProducts,
	});
});

// Add an alias route for /recommendations
router.get("/recommendations", (req, res) => {
	res.render("recommendations", {
		similarProducts,
		recommendedProducts,
	});
});

router.get("/product/:id", (req, res) => {
	console.log("Product route accessed with id:", req.params.id);

	// Search for the product across all product arrays
	let foundInArray = "";
	const product =
		products.find((p) => {
			if (p.productId === req.params.id) {
				foundInArray = "products";
				return true;
			}
			return false;
		}) ||
		similarProducts.find((p) => {
			if (p.productId === req.params.id) {
				foundInArray = "similarProducts";
				return true;
			}
			return false;
		}) ||
		recommendedProducts.find((p) => {
			if (p.productId === req.params.id) {
				foundInArray = "recommendedProducts";
				return true;
			}
			return false;
		});

	if (!product) {
		console.log("Product not found. Available IDs:", {
			products: products.map((p) => p.productId),
			similarProducts: similarProducts.map((p) => p.productId),
			recommendedProducts: recommendedProducts.map((p) => p.productId),
		});
		return res.status(404).send("Product not found");
	}

	console.log("Found product in array:", foundInArray);
	console.log("Found product:", product);
	console.log("Product category:", product.category);

	// Get similar products based on category (keep this category-based)
	const filteredSimilarProducts = similarProducts.filter((p) => {
		const matches =
			p.productId !== product.productId &&
			p.category === product.category;
		console.log(
			`Similar product ${p.productId} category: ${p.category}, matches: ${matches}`
		);
		return matches;
	});

	// Show all recommended products except the current one
	const filteredRecommendedProducts = recommendedProducts.filter(
		(p) => p.productId !== product.productId
	);

	console.log(
		"Filtered similar products:",
		filteredSimilarProducts.map((p) => p.productId)
	);
	console.log(
		"Filtered recommended products:",
		filteredRecommendedProducts.map((p) => p.productId)
	);

	res.render("product", {
		product,
		similarProducts: filteredSimilarProducts,
		recommendedProducts: filteredRecommendedProducts,
	});
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
	console.log("Request body:", req.body);

	const { productId, quantity } = req.body;

	if (!productId || !quantity) {
		console.error("Missing required fields:", { productId, quantity });
		return res.status(400).json({
			success: false,
			message: "Missing required fields",
		});
	}

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
	res.json({ success: true, message: "Product added to cart successfully" });
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

router.post("/chatbot/image", (req, res) => {
	console.log("post to /chatbot/image triggered");
	const { image } = req.body;
	if (image) {
		req.session.uploadedImage = image; // Store in session
		res.json({ success: true });
	} else {
		res.json({ success: false });
	}
});

router.get("/session-debug", (req, res) => {
	res.json(req.session);
});

module.exports = router;
