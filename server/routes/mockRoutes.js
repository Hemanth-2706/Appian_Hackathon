const express = require("express");
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const { products, similarProducts, recommendProducts } = require(path.join(
	__dirname,
	"../data/products"
));
// log("finding model");
// const { FashionRecommender } = require(path.join(__dirname, "../data/model"));
// log("model found");

const router = express.Router();

// Configure logging
const logsDir = path.join(__dirname, "../logs");
if (!fs.existsSync(logsDir)) {
	fs.mkdirSync(logsDir, { recursive: true });
}

const logFile = fs.createWriteStream(path.join(logsDir, "mockRoutes.log"), {
	flags: "a",
});

const allLogFile = fs.createWriteStream(path.join(logsDir, "all_logs.log"), {
	flags: "a",
});

function log(message, type = "INFO", data = null) {
	const timestamp = new Date().toISOString();
	let logMessage = `${timestamp} - (mockRoutes.js) - ${type} - ${message}`;

	if (data) {
		if (typeof data === "object") {
			logMessage += `\nData: ${JSON.stringify(data, null, 2)}`;
		} else {
			logMessage += `\nData: ${data}`;
		}
	}

	logMessage += "\n";
	logFile.write(logMessage);
	allLogFile.write(logMessage);
	console.log(logMessage);
}

router.use(express.json()); // Needed to parse JSON body
router.use(express.urlencoded({ extended: true }));

// Function to get images from directory
function getImagesFromDirectory(dirPath) {
	log(`=== Reading Images from Directory ===`);
	log(`Directory path: ${dirPath}`);
	try {
		const files = fs.readdirSync(dirPath);
		const images = files.filter((file) => {
			const ext = path.extname(file).toLowerCase();
			return [".jpg", ".jpeg", ".png", ".gif"].includes(ext);
		});
		log(`Found ${images.length} images in directory`, "INFO", images);
		log(`=== Directory Reading Complete ===`);
		return images;
	} catch (err) {
		log(`Error reading directory ${dirPath}: ${err.message}`, "ERROR");
		return [];
	}
}

// Home Route - loads product data and banner images
router.get("/", (req, res) => {
	log(`=== Processing Home Route Request ===`);
	const bannerDir = path.join(
		__dirname,
		"../../client/public/images/banner"
	);
	const dataImagesDir = path.join(__dirname, "../data/images");
	let bannerImages = [];

	try {
		log(`Reading banner images from: ${bannerDir}`);
		bannerImages = fs.readdirSync(bannerDir);
		log(
			`Found ${bannerImages.length} banner images`,
			"INFO",
			bannerImages
		);
	} catch (err) {
		log(`Error reading banner images folder: ${err.message}`, "ERROR");
	}

	// Get only the images from the root of data/images directory
	log(`Reading root images from: ${dataImagesDir}`);
	const rootImages = getImagesFromDirectory(dataImagesDir);

	// Create featured products using product details from products.js and root images
	log(
		`Creating featured products from ${products.length} available products`
	);
	const featuredProducts = products
		.map((product, index) => {
			const imageName = rootImages[index % rootImages.length];
			return {
				...product,
				image: `/images/${imageName}`,
			};
		})
		.slice(0, 8);

	log(
		`Created ${featuredProducts.length} featured products`,
		"INFO",
		featuredProducts
	);
	log(`=== Home Route Processing Complete ===`);

	res.render("home", {
		products: featuredProducts,
		bannerImages,
	});
});

// Process recommendations endpoint
router.post("/process-recommendations", async (req, res) => {
	log(`=== Processing Recommendation Request ===`);
	try {
		// Get user input from session
		const userText = req.session.userText || null;
		const userImage = req.session.uploadedImage || null;
		log(`Session data retrieved`, "INFO", {
			hasText: !!userText,
			hasImage: !!userImage,
		});

		if (!userText && !userImage) {
			log("No input provided in session", "WARN");
			return res.status(400).json({ error: "No input provided" });
		}

		// Call FastAPI endpoint to use the Python model
		log("Calling FastAPI endpoint for recommendations");
		const response = await axios.post(
			"http://localhost:5002/process-recommendations",
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
				response.data.detail ||
					"Failed to process recommendations. Failed axios.post()"
			);
		}

		// Store the results in session
		log("Storing recommendation results in session");
		req.session.recommendationResults = response.data;
		log(`Recommendation results stored`, "INFO", {
			similarProductsCount: response.data.similarProducts?.length || 0,
			recommendProductsCount:
				response.data.recommendProducts?.length || 0,
		});

		log(`=== Recommendation Processing Complete ===`);
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
	log(`=== Processing Product Details Request ===`);
	log(`Requested product IDs:`, "INFO", req.body.product_ids);
	try {
		const { product_ids } = req.body;

		if (!product_ids || !Array.isArray(product_ids)) {
			log("Invalid product IDs provided", "WARN", { product_ids });
			return res.status(400).json({ error: "Invalid product IDs" });
		}

		// Call FastAPI endpoint to get product details
		log("Calling FastAPI endpoint for product details");
		const response = await axios.post(
			"http://localhost:5002/get-product-details",
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

		log(`Product details retrieved successfully`, "INFO", {
			productsCount: response.data.products?.length || 0,
		});
		log(`=== Product Details Request Complete ===`);
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
	log(`=== Processing Health Check Request ===`);
	try {
		const response = await axios.get("http://localhost:5002/health", {
			headers: {
				Accept: "application/json",
			},
		});
		log("Health check successful", "INFO", response.data);
		log(`=== Health Check Complete ===`);
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
	log(`=== Processing Recommend Page Request ===`);
	// Get the processed results from session
	const results = req.session.recommendationResults;

	if (!results) {
		log("No recommendation results found in session", "WARN");
		return res.redirect("/"); // Redirect to home if no results
	}

	// Process similar products to ensure proper image paths
	log("Processing similar products");
	const processedSimilarProducts = (results?.similarProducts || []).map(
		(product) => ({
			...product,
			image: `/images/similarProducts/${product.productId}.jpg`,
		})
	);

	// Process recommended products to ensure proper image paths
	log("Processing recommended products");
	const processedRecommendProducts = (results?.recommendProducts || []).map(
		(product) => ({
			...product,
			image: `/images/recommendProducts/${product.productId}.jpg`,
		})
	);

	// Log detailed product information
	log("Similar Products Details:", "INFO", processedSimilarProducts);
	log("Recommended Products Details:", "INFO", processedRecommendProducts);

	log(`Rendering recommend page`, "INFO", {
		similarProductsCount: processedSimilarProducts.length,
		recommendProductsCount: processedRecommendProducts.length,
	});
	log(`=== Recommend Page Processing Complete ===`);

	res.render("recommend", {
		products: products,
		similarProducts: processedSimilarProducts,
		recommendProducts: processedRecommendProducts,
	});
});

router.get("/product/:id", (req, res) => {
	log(`=== Processing Product Details Page Request ===`);
	log(`Requested product ID: ${req.params.id}`);

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
		log("Product not found", "WARN", {
			availableIds: {
				products: products.map((p) => p.productId),
				similarProducts: similarProducts.map((p) => p.productId),
				recommendProducts: recommendProducts.map(
					(p) => p.productId
				),
			},
		});
		return res.status(404).send("Product not found");
	}

	log(`Product found in ${foundInArray} array`, "INFO", product);

	// Get similar products based on category
	log("Filtering similar products by category");
	const filteredSimilarProducts = similarProducts.filter((p) => {
		const matches =
			p.productId !== product.productId &&
			p.category === product.category;
		return matches;
	});

	// Show all recommended products except the current one
	log("Filtering recommended products");
	const filteredRecommendProducts = recommendProducts.filter(
		(p) => p.productId !== product.productId
	);

	log(`Filtered products`, "INFO", {
		similarProductsCount: filteredSimilarProducts.length,
		recommendProductsCount: filteredRecommendProducts.length,
	});
	log(`=== Product Details Page Processing Complete ===`);

	res.render("product", {
		product,
		similarProducts: filteredSimilarProducts,
		recommendProducts: filteredRecommendProducts,
	});
});

router.get("/cart", (req, res) => {
	log(`=== Processing Cart Page Request ===`);
	const sessionCart = req.session.cart || [];

	log(`Processing ${sessionCart.length} cart items`);
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
		.filter((item) => item !== null);

	log(`Cart items processed`, "INFO", {
		totalItems: cartItems.length,
		items: cartItems,
	});
	log(`=== Cart Page Processing Complete ===`);

	res.render("cart", { cartItems });
});

router.get("/checkout/:id", (req, res) => {
	const product = products.find((p) => p.productId === req.params.id);
	res.render("payment", { product });
});

router.post("/cart/add", (req, res) => {
	log(`=== Processing Add to Cart Request ===`);
	log(`Request body:`, "INFO", req.body);

	const { productId, quantity } = req.body;

	if (!productId || !quantity) {
		log("Missing required fields", "WARN", { productId, quantity });
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
		log(`Updated existing cart item`, "INFO", req.session.cart[index]);
	} else {
		req.session.cart.push({ productId, quantity: parseInt(quantity) });
		log(`Added new item to cart`, "INFO", { productId, quantity });
	}

	log(`Cart updated`, "INFO", req.session.cart);
	log(`=== Add to Cart Request Complete ===`);
	res.json({ success: true, message: "Product added to cart successfully" });
});

router.post("/cart/remove", (req, res) => {
	log(`=== Processing Remove from Cart Request ===`);
	log(`Request body:`, "INFO", req.body);
	const { productId } = req.body;

	if (!req.session.cart) {
		req.session.cart = [];
	}

	req.session.cart = req.session.cart.filter(
		(item) => item.productId !== productId
	);

	log(`Cart updated after removal`, "INFO", req.session.cart);
	log(`=== Remove from Cart Request Complete ===`);
	res.json({ success: true, message: "Item removed from cart" });
});

router.post("/chatbot/image", (req, res) => {
	log(`=== Processing Chatbot Image Upload ===`);
	const { image } = req.body;
	if (image) {
		req.session.uploadedImage = image;
		log(`Image stored in session`, "INFO", { hasImage: true });
		res.json({ success: true });
	} else {
		log("No image provided", "WARN");
		res.json({ success: false });
	}
	log(`=== Chatbot Image Upload Complete ===`);
});

router.post("/chatbot/text", (req, res) => {
	log(`=== Processing Chatbot Text Input ===`);
	const { text } = req.body;
	if (text) {
		req.session.userText = text;
		log(`Text stored in session`, "INFO", { hasText: true });
		res.json({ success: true });
	} else {
		log("No text provided", "WARN");
		res.json({ success: false });
	}
	log(`=== Chatbot Text Input Complete ===`);
});

router.post("/chatbot/clear-history", (req, res) => {
	log(`=== Processing Clear History Request ===`);
	try {
		// Clear all chatbot-related session data
		delete req.session.userText;
		delete req.session.uploadedImage;
		delete req.session.recommendationResults;

		// Also clear session history and summary in Python recommender
		const axios = require("axios");
		axios.post("http://localhost:5002/clear-session-history").then(() => {
			log("Session history and summary cleared in Python recommender", "INFO");
		}).catch((err) => {
			log(`Failed to clear session history in Python recommender: ${err.message}`, "ERROR");
		});

		log(`Session cleared successfully`, "INFO");
		log(`=== Clear History Request Complete ===`);
		res.json({ success: true, message: "History cleared successfully" });
	} catch (error) {
		log(`Error clearing history: ${error.message}`, "ERROR");
		res.status(500).json({
			success: false,
			message: "Failed to clear history",
			error: error.message,
		});
	}
});

module.exports = router;