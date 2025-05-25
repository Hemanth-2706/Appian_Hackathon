const express = require("express");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const { products, similarProducts, recommendProducts } = require(path.join(
	__dirname,
	"../data/products"
));
// console.log("finding model");
// const { FashionRecommender } = require(path.join(__dirname, "../data/model"));
// console.log("model found");

const router = express.Router();

// Configure logging
const logsDir = path.join(__dirname, "../logs");
if (!fs.existsSync(logsDir)) {
	fs.mkdirSync(logsDir, { recursive: true });
}

const logFile = fs.createWriteStream(path.join(logsDir, "mockRoutes.log"), {
	flags: "a",
});

function log(message, type = "INFO") {
	const timestamp = new Date().toISOString();
	const logMessage = `${timestamp} - ${type} - ${message}\n`;
	logFile.write(logMessage);
	console.log(logMessage);
}

router.use(express.json()); // Needed to parse JSON body
router.use(express.urlencoded({ extended: true }));

// Function to get images from directory
function getImagesFromDirectory(dirPath) {
	log(`Reading images from directory: ${dirPath}`);
	try {
		const files = fs.readdirSync(dirPath);
		const images = files.filter((file) => {
			const ext = path.extname(file).toLowerCase();
			return [".jpg", ".jpeg", ".png", ".gif"].includes(ext);
		});
		log(`Found ${images.length} images in directory`);
		return images;
	} catch (err) {
		log(`Error reading directory ${dirPath}: ${err.message}`, "ERROR");
		return [];
	}
}

// Home Route - loads product data and banner images
router.get("/", (req, res) => {
	const bannerDir = path.join(
		__dirname,
		"../../client/public/images/banner"
	);
	const dataImagesDir = path.join(__dirname, "../data/images");
	let bannerImages = [];

	try {
		bannerImages = fs.readdirSync(bannerDir);
	} catch (err) {
		console.error("Error reading banner images folder:", err);
	}

	// Get only the images from the root of data/images directory
	const rootImages = getImagesFromDirectory(dataImagesDir);

	// Create featured products using product details from products.js and root images
	const featuredProducts = products
		.map((product, index) => {
			// Use the image from root of data/images
			const imageName = rootImages[index % rootImages.length]; // Cycle through available root images
			return {
				...product,
				image: `/images/${imageName}`, // This will point to images in the root directory
			};
		})
		.slice(0, 8); // Show first 8 products as featured

	res.render("home", {
		products: featuredProducts,
		bannerImages,
	});
});

// Process recommendations endpoint
router.post("/process-recommendations", async (req, res) => {
	log("Received recommendation request");
	try {
		// Get user input from session
		const userText = req.session.userText || null;
		const userImage = req.session.uploadedImage || null;
		log(
			`Session data - Text: ${userText ? "Present" : "None"}, Image: ${
				userImage ? "Present" : "None"
			}`
		);

		if (!userText && !userImage) {
			log("No input provided in session", "WARN");
			return res.status(400).json({ error: "No input provided" });
		}

		// Call FastAPI endpoint to use the Python model
		log("Calling FastAPI endpoint for recommendations");
		const response = await axios.post(
			"http://localhost:5001/process-recommendations",
			{
				text: userText,
				image: userImage,
			},
			{
				headers: {
					"Content-Type": "application/json",
					Accept: "application/json",
				},
			}
		);

		if (!response.data.success) {
			throw new Error(
				response.data.detail || "Failed to process recommendations"
			);
		}

		// Store the results in session
		log("Storing recommendation results in session");
		req.session.recommendationResults = response.data;
		log(
			`Processed ${
				response.data.similarProducts?.length || 0
			} similar products and ${
				response.data.recommendProducts?.length || 0
			} recommended products`
		);

		res.json({ success: true });
	} catch (error) {
		log(
			`Error processing recommendations: ${
				error.response?.data || error.message
			}`,
			"ERROR"
		);
		res.status(500).json({
			error: "Failed to process recommendations",
			details: error.response?.data?.detail || error.message,
		});
	}
});

// Get product details endpoint
router.post("/get-product-details", async (req, res) => {
	log(`Received product details request for IDs: ${req.body.product_ids}`);
	try {
		const { product_ids } = req.body;

		if (!product_ids || !Array.isArray(product_ids)) {
			log("Invalid product IDs provided", "WARN");
			return res.status(400).json({ error: "Invalid product IDs" });
		}

		// Call FastAPI endpoint to get product details
		log("Calling FastAPI endpoint for product details");
		const response = await axios.post(
			"http://localhost:5001/get-product-details",
			{ product_ids },
			{
				headers: {
					"Content-Type": "application/json",
					Accept: "application/json",
				},
			}
		);

		if (!response.data.success) {
			throw new Error(
				response.data.detail || "Failed to get product details"
			);
		}

		log(
			`Retrieved details for ${
				response.data.products?.length || 0
			} products`
		);
		res.json(response.data);
	} catch (error) {
		log(
			`Error getting product details: ${
				error.response?.data || error.message
			}`,
			"ERROR"
		);
		res.status(500).json({
			error: "Failed to get product details",
			details: error.response?.data?.detail || error.message,
		});
	}
});

// Health check endpoint
router.get("/model-health", async (req, res) => {
	log("Health check requested");
	try {
		const response = await axios.get("http://localhost:5001/health", {
			headers: {
				Accept: "application/json",
			},
		});
		log("Health check successful");
		res.json(response.data);
	} catch (error) {
		log(
			`Error checking model health: ${
				error.response?.data || error.message
			}`,
			"ERROR"
		);
		res.status(500).json({
			error: "Model server is not responding",
			details: error.response?.data?.detail || error.message,
		});
	}
});

// Recommend page route
router.get("/recommend", (req, res) => {
	log("Recommend page requested");
	// Get the processed results from session
	const results = req.session.recommendationResults;

	if (!results) {
		log("No recommendation results found in session", "WARN");
		return res.redirect("/"); // Redirect to home if no results
	}

	log(
		`Rendering recommend page with ${
			results.similarProducts?.length || 0
		} similar products and ${
			results.recommendProducts?.length || 0
		} recommended products`
	);
	res.render("recommend", {
		products: products, // Base products
		similarProducts: results?.similarProducts || [], // Similar products if available
		recommendProducts: results?.recommendProducts || [], // Recommended products if available
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
		recommendProducts.find((p) => {
			if (p.productId === req.params.id) {
				foundInArray = "recommendProducts";
				return true;
			}
			return false;
		});

	if (!product) {
		console.log("Product not found. Available IDs:", {
			products: products.map((p) => p.productId),
			similarProducts: similarProducts.map((p) => p.productId),
			recommendProducts: recommendProducts.map((p) => p.productId),
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
	const filteredRecommendProducts = recommendProducts.filter(
		(p) => p.productId !== product.productId
	);

	console.log(
		"Filtered similar products:",
		filteredSimilarProducts.map((p) => p.productId)
	);
	console.log(
		"Filtered recommend products:",
		filteredRecommendProducts.map((p) => p.productId)
	);

	res.render("product", {
		product,
		similarProducts: filteredSimilarProducts,
		recommendProducts: filteredRecommendProducts,
	});
});

router.get("/cart", (req, res) => {
	console.log("get to /cart triggered");

	const sessionCart = req.session.cart || [];

	const cartItems = sessionCart
		.map((cartItem) => {
			// Search for the product across all product arrays
			let product = products.find(
				(p) => p.productId === cartItem.productId
			);
			let imagePath = product?.image;

			if (!product) {
				product = similarProducts.find(
					(p) => p.productId === cartItem.productId
				);
				if (product) {
					imagePath = `/images/similarProducts/${product.productId}.jpg`;
				}
			}

			if (!product) {
				product = recommendProducts.find(
					(p) => p.productId === cartItem.productId
				);
				if (product) {
					imagePath = `/images/recommendProducts/${product.productId}.jpg`;
				}
			}

			if (product) {
				return {
					productId: product.productId,
					productName: product.productName,
					articleType: product.articleType,
					subCategory: product.subCategory,
					season: product.season,
					usage: product.usage,
					image: imagePath,
					price: product.price,
					quantity: cartItem.quantity,
				};
			}
			return null;
		})
		.filter((item) => item !== null); // Remove any null items (products not found)

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

router.post("/chatbot/text", (req, res) => {
	console.log("post to /chatbot/text triggered");
	const { text } = req.body;
	if (text) {
		req.session.userText = text; // Store in session
		res.json({ success: true });
	} else {
		res.json({ success: false });
	}
});

router.get("/session-debug", (req, res) => {
	res.json(req.session);
});

module.exports = router;
