const express = require("express");
const path = require("path");
const fs = require("fs");
const axios = require("axios");

const { homeProducts } = require(path.join(__dirname, "../data/homeProducts"));

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

// Middleware to check if user is authenticated
const isAuthenticated = (req, res, next) => {
	if (req.session.user) {
		next();
	} else {
		// Redirect to login page with error message
		return res.redirect(
			"/login?error=Please login to proceed to payment"
		);
	}
};

// Middleware to check if user is admin
const isAdmin = (req, res, next) => {
	if (req.session.user && req.session.user.role === "admin") {
		next();
	} else {
		res.status(403).json({
			success: false,
			message: "Access denied. Admin privileges required.",
		});
	}
};

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
	log(`[HOME] === Processing Home Route Request ===`);
	const bannerDir = path.join(
		__dirname,
		"../../client/public/images/banner"
	);
	const dataImagesDir = path.join(__dirname, "../data/images");
	let bannerImages = [];

	try {
		log(`[HOME] Reading banner images from: ${bannerDir}`);
		bannerImages = fs.readdirSync(bannerDir);
		log(
			`[HOME] Found ${bannerImages.length} banner images`,
			"INFO",
			bannerImages
		);
	} catch (err) {
		log(
			`[HOME] Error reading banner images folder: ${err.message}`,
			"ERROR"
		);
	}

	// Get only the images from the root of data/images directory
	log(`[HOME] Reading root images from: ${dataImagesDir}`);
	const rootImages = getImagesFromDirectory(dataImagesDir);

	log(`[HOME] homeProducts = `, "INFO", homeProducts);
	// Create featured products using product details from products.js and root images
	log(
		`[HOME] Creating featured products from ${homeProducts.length} available products`
	);
	const featuredProducts = homeProducts
		.map((product, index) => {
			const imageName = rootImages[index % rootImages.length];
			return {
				...product,
				image: `/images/${imageName}`,
			};
		})
		.slice(0, 8);

	log(
		`[HOME] Created ${featuredProducts.length} featured products`,
		"INFO",
		featuredProducts
	);
	log(`[HOME] === Home Route Processing Complete ===`);

	res.render("home", {
		products: featuredProducts,
		bannerImages,
	});
});

// Process recommendations endpoint
router.post("/process-recommendations", async (req, res) => {
	log(`[PROCESS_RECOMMENDATIONS] === Processing Recommendation Request ===`);
	try {
		// Get user input from session
		const userText = req.session.chatHistory?.userText?.content || null;
		const userImage = req.session.chatHistory?.image?.content || null;
		log(`[PROCESS_RECOMMENDATIONS] Session data retrieved`, "INFO", {
			hasText: !!userText,
			hasImage: !!userImage,
		});

		if (!userText && !userImage) {
			log(
				`[PROCESS_RECOMMENDATIONS] No input provided in session`,
				"WARN"
			);
			return res.status(400).json({ error: "No input provided" });
		}

		// Call FastAPI endpoint to use the Python model
		log(
			`[PROCESS_RECOMMENDATIONS] Calling FastAPI endpoint for recommendations`
		);
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
		log(
			`[PROCESS_RECOMMENDATIONS] Storing recommendation results in session`
		);
		// === Dynamically reload updated products.js ===
		const productsPath = path.join(__dirname, "../data/products.js");
		delete require.cache[require.resolve(productsPath)];
		const { similarProducts, recommendProducts } = require(productsPath);

		// Store the meaningful caption from the response
		const meaningfulCaption = response.data.meaningfulCaption;
		req.session.recommendationResults = {
			similarProducts: similarProducts || [],
			recommendProducts: recommendProducts || [],
			meaningfulCaption: meaningfulCaption || "",
		};

		log(
			`[PROCESS_RECOMMENDATIONS] Recommendation results stored`,
			"INFO",
			{
				similarProductsCount:
					req.session.recommendationResults.similarProducts
						.length,
				recommendProductsCount:
					req.session.recommendationResults.recommendProducts
						.length,
				hasCaption: !!meaningfulCaption,
			}
		);

		log(
			`[PROCESS_RECOMMENDATIONS] === Recommendation Processing Complete ===`
		);
		res.json({ success: true });
	} catch (error) {
		log(
			`[PROCESS_RECOMMENDATIONS] Error processing recommendations: ${
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
	log(`[GET_PRODUCT_DETAILS] === Processing Product Details Request ===`);
	log(
		`[GET_PRODUCT_DETAILS] Requested product IDs:`,
		"INFO",
		req.body.product_ids
	);
	try {
		const { product_ids } = req.body;

		if (!product_ids || !Array.isArray(product_ids)) {
			log(
				`[GET_PRODUCT_DETAILS] Invalid product IDs provided`,
				"WARN",
				{ product_ids }
			);
			return res.status(400).json({ error: "Invalid product IDs" });
		}

		// Call FastAPI endpoint to get product details
		log(
			`[GET_PRODUCT_DETAILS] Calling FastAPI endpoint for product details`
		);
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
			`[GET_PRODUCT_DETAILS] Product details retrieved successfully`,
			"INFO",
			{
				productsCount: response.data.products?.length || 0,
			}
		);
		log(`[GET_PRODUCT_DETAILS] === Product Details Request Complete ===`);
		res.json(response.data);
	} catch (error) {
		log(
			`[GET_PRODUCT_DETAILS] Error getting product details: ${
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
	log(`[MODEL_HEALTH] === Processing Health Check Request ===`);
	try {
		const response = await axios.get("http://localhost:5001/health", {
			headers: {
				Accept: "application/json",
			},
		});
		log(`[MODEL_HEALTH] Health check successful`, "INFO", response.data);
		log(`[MODEL_HEALTH] === Health Check Complete ===`);
		res.json(response.data);
	} catch (error) {
		log(
			`[MODEL_HEALTH] Error checking model health: ${
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
	log(`[RECOMMEND] === Processing Recommend Page Request ===`);
	// Get the processed results from session
	const results = req.session.recommendationResults;

	if (!results) {
		log(`[RECOMMEND] No recommendation results found in session`, "WARN");
		return res.redirect("/"); // Redirect to home if no results
	}

	// Process similar products to ensure proper image paths
	log(`[RECOMMEND] Processing similar products`);
	const processedSimilarProducts = (results?.similarProducts || []).map(
		(product) => ({
			...product,
			image: `/images/similarProducts/${product.productId}.jpg`,
		})
	);

	// Process recommended products to ensure proper image paths
	log(`[RECOMMEND] Processing recommended products`);
	const processedRecommendProducts = (results?.recommendProducts || []).map(
		(product) => ({
			...product,
			image: `/images/recommendProducts/${product.productId}.jpg`,
		})
	);

	// Get the meaningful caption
	const meaningfulCaption =
		results.meaningfulCaption || "Here are your recommendations!";

	// Log detailed product information
	log(
		`[RECOMMEND] Similar Products Details:`,
		"INFO",
		processedSimilarProducts
	);
	log(
		`[RECOMMEND] Recommended Products Details:`,
		"INFO",
		processedRecommendProducts
	);
	log(`[RECOMMEND] Meaningful Caption:`, "INFO", meaningfulCaption);

	log(`[RECOMMEND] Rendering recommend page`, "INFO", {
		similarProductsCount: processedSimilarProducts.length,
		recommendProductsCount: processedRecommendProducts.length,
		hasCaption: !!meaningfulCaption,
	});
	log(`[RECOMMEND] === Recommend Page Processing Complete ===`);

	res.render("recommend", {
		similarProducts: processedSimilarProducts,
		recommendProducts: processedRecommendProducts,
		meaningfulCaption: meaningfulCaption,
	});
});

router.get("/product/:id", (req, res) => {
	log(`[PRODUCT] === Processing Product Details Page Request ===`);
	log(`[PRODUCT] Requested product ID: ${req.params.id}`);

	// Get products from session
	const sessionSimilarProducts =
		req.session.recommendationResults?.similarProducts || [];
	const sessionRecommendProducts =
		req.session.recommendationResults?.recommendProducts || [];

	// Search for the product across all product arrays
	let foundInArray = "";
	const product =
		sessionSimilarProducts.find((p) => {
			if (
				String(p.productId).trim() === String(req.params.id).trim()
			) {
				foundInArray = "similarProducts";
				return true;
			}
			return false;
		}) ||
		sessionRecommendProducts.find((p) => {
			if (
				String(p.productId).trim() === String(req.params.id).trim()
			) {
				foundInArray = "recommendProducts";
				return true;
			}
			return false;
		});

	if (!product) {
		log(`[PRODUCT] Product not found`, "WARN", {
			availableIds: {
				similarProducts: sessionSimilarProducts.map(
					(p) => p.productId
				),
				recommendProducts: sessionRecommendProducts.map(
					(p) => p.productId
				),
			},
		});
		return res.status(404).send("Product not found");
	}

	log(`[PRODUCT] Product found in ${foundInArray} array`, "INFO", product);

	// Get similar products based on category
	log(`[PRODUCT] Filtering similar products by category`);
	const filteredSimilarProducts = sessionSimilarProducts.filter((p) => {
		const matches =
			String(p.productId).trim() !==
				String(product.productId).trim() &&
			p.category == product.category;
		return matches;
	});

	// Show all recommended products except the current one
	log(`[PRODUCT] Filtering recommended products`);
	const filteredRecommendProducts = sessionRecommendProducts.filter(
		(p) => String(p.productId).trim() !== String(product.productId).trim()
	);

	log(`[PRODUCT] Filtered products`, "INFO", {
		similarProductsCount: filteredSimilarProducts.length,
		recommendProductsCount: filteredRecommendProducts.length,
	});
	log(`[PRODUCT] === Product Details Page Processing Complete ===`);

	res.render("product", {
		product,
		similarProducts: filteredSimilarProducts,
		recommendProducts: filteredRecommendProducts,
	});
});

// Cart page route
router.get("/cart", (req, res) => {
	log(`[CART] === Processing Cart Page Request ===`);

	const sessionCart = req.session.cart || [];
	const user = req.session.user;

	// Get products from session and homeProducts
	const sessionSimilarProducts =
		req.session.recommendationResults?.similarProducts || [];
	const sessionRecommendProducts =
		req.session.recommendationResults?.recommendProducts || [];

	log(`[CART] Processing ${sessionCart.length} cart items`);
	log(`[CART] Session cart data:`, "INFO", sessionCart);
	log(`[CART] Using products from session:`, "INFO", {
		hasSessionSimilarProducts: sessionSimilarProducts.length > 0,
		hasSessionRecommendProducts: sessionRecommendProducts.length > 0,
		hasHomeProducts: homeProducts.length > 0,
		similarProductsCount: sessionSimilarProducts.length,
		recommendProductsCount: sessionRecommendProducts.length,
		homeProductsCount: homeProducts.length,
	});

	const cartItems = sessionCart
		.map((cartItem) => {
			log(`[CART] Processing cart item:`, "INFO", cartItem);

			// Convert productId to string for consistent comparison
			const cartItemId = String(cartItem.productId).trim();

			// Search for the product in similar, recommended, and home products
			let product = sessionSimilarProducts.find(
				(p) => String(p.productId).trim() === cartItemId
			);
			let imagePath = product?.image;

			if (!product) {
				log(
					`[CART] Product ${cartItemId} not found in session similarProducts array`,
					"INFO"
				);
				product = sessionRecommendProducts.find(
					(p) => String(p.productId).trim() === cartItemId
				);
				if (product) {
					imagePath = `/images/recommendProducts/${product.productId}.jpg`;
					log(
						`[CART] Product found in session recommendProducts array`,
						"INFO",
						product
					);
				}
			}

			if (!product) {
				log(
					`[CART] Product ${cartItemId} not found in session arrays, checking homeProducts`,
					"INFO"
				);
				product = homeProducts.find(
					(p) => String(p.productId).trim() === cartItemId
				);
				if (product) {
					imagePath = `/images/${product.productId}.png`;
					log(
						`[CART] Product found in homeProducts array`,
						"INFO",
						product
					);
				}
			}

			if (!product) {
				log(
					`[CART] Product ${cartItemId} not found in any array`,
					"WARN"
				);
				return null;
			}

			// Create cart item with all necessary data
			const cartItemData = {
				productId: product.productId,
				productName: product.productName,
				articleType: product.articleType,
				subCategory: product.subCategory,
				season: product.season,
				usage: product.usage,
				image: imagePath,
				price: parseFloat(product.price),
				quantity: parseInt(cartItem.quantity),
			};

			log(`[CART] Created cart item data:`, "INFO", cartItemData);
			return cartItemData;
		})
		.filter((item) => item !== null);

	// Calculate total price and total quantity
	const totalPrice = cartItems.reduce(
		(total, item) => total + item.price * item.quantity,
		0
	);
	const totalQuantity = cartItems.reduce(
		(total, item) => total + item.quantity,
		0
	);

	// Store totals in session
	req.session.cartTotals = {
		totalPrice,
		totalQuantity,
	};

	log(`[CART] Cart items processed`, "INFO", {
		totalItems: cartItems.length,
		items: cartItems,
		totalPrice: totalPrice,
	});
	log(`[CART] === Cart Page Processing Complete ===`);

	res.render("cart", {
		cartItems,
		totalPrice: totalPrice.toFixed(2),
		user: req.session.user, // Pass user data to the template
	});
});

router.get("/checkout/:id", (req, res) => {
	const product = products.find((p) => p.productId == req.params.id);
	res.render("payment", { product });
});

router.post("/cart/add", (req, res) => {
	log(`[CART_ADD] === Processing Add to Cart Request ===`);
	log(`[CART_ADD] Request body:`, "INFO", req.body);

	const { productId, quantity } = req.body;

	if (!productId || !quantity) {
		log(`[CART_ADD] Missing required fields`, "WARN", {
			productId,
			quantity,
		});
		return res.status(400).json({
			success: false,
			message: "Missing required fields",
		});
	}

	// Get products from session
	const sessionSimilarProducts =
		req.session.recommendationResults?.similarProducts || [];
	const sessionRecommendProducts =
		req.session.recommendationResults?.recommendProducts || [];

	log(`[CART_ADD] Session products:`, "INFO", {
		hasSessionSimilarProducts: sessionSimilarProducts.length > 0,
		hasSessionRecommendProducts: sessionRecommendProducts.length > 0,
		similarProductsCount: sessionSimilarProducts.length,
		recommendProductsCount: sessionRecommendProducts.length,
	});

	// Convert productId to string for consistent comparison
	const searchId = String(productId).trim();
	log(`[CART_ADD] Searching for product with ID: ${searchId}`);

	// Verify product exists in one of the arrays
	const product =
		sessionSimilarProducts.find(
			(p) => String(p.productId).trim() === searchId
		) ||
		sessionRecommendProducts.find(
			(p) => String(p.productId).trim() === searchId
		) ||
		homeProducts.find((p) => String(p.productId).trim() === searchId);

	if (!product) {
		log(`[CART_ADD] Product ${searchId} not found in any array`, "WARN");
		return res.status(404).json({
			success: false,
			message: "Product not found",
		});
	}

	// Initialize cart if not present
	if (!req.session.cart) {
		req.session.cart = [];
	}

	// Add or update product
	const index = req.session.cart.findIndex(
		(item) => String(item.productId).trim() === searchId
	);
	if (index !== -1) {
		req.session.cart[index].quantity += parseInt(quantity);
		log(
			`[CART_ADD] Updated existing cart item`,
			"INFO",
			req.session.cart[index]
		);
	} else {
		req.session.cart.push({
			productId: searchId,
			quantity: parseInt(quantity),
		});
		log(`[CART_ADD] Added new item to cart`, "INFO", {
			productId: searchId,
			quantity,
		});
	}

	// Save session
	req.session.save((err) => {
		if (err) {
			log(`[CART_ADD] Error saving session: ${err.message}`, "ERROR");
			return res.status(500).json({
				success: false,
				message: "Failed to update cart",
				error: err.message,
			});
		}
		log(`[CART_ADD] Cart updated`, "INFO", req.session.cart);
		log(`[CART_ADD] === Add to Cart Request Complete ===`);
		res.json({
			success: true,
			message: "Product added to cart successfully",
			cart: req.session.cart,
		});
	});
});

router.post("/cart/add-multiple", (req, res) => {
	log(`[CART_ADD_MULTIPLE] === Processing Add Multiple to Cart Request ===`);
	log(`[CART_ADD_MULTIPLE] Request body:`, "INFO", req.body);

	const { products } = req.body;

	if (!products || !Array.isArray(products)) {
		log(`[CART_ADD_MULTIPLE] Invalid products data`, "WARN", {
			products,
		});
		return res.status(400).json({
			success: false,
			message: "Invalid products data",
		});
	}

	// Initialize cart if not present
	if (!req.session.cart) {
		req.session.cart = [];
	}

	// Add or update products
	products.forEach(({ productId, quantity }) => {
		const index = req.session.cart.findIndex(
			(item) => item.productId == productId
		);
		if (index !== -1) {
			req.session.cart[index].quantity += parseInt(quantity);
		} else {
			req.session.cart.push({
				productId,
				quantity: parseInt(quantity),
			});
		}
	});

	// Save session
	req.session.save((err) => {
		if (err) {
			log(
				`[CART_ADD_MULTIPLE] Error saving session: ${err.message}`,
				"ERROR"
			);
			return res.status(500).json({
				success: false,
				message: "Failed to update cart",
				error: err.message,
			});
		}
		log(
			`[CART_ADD_MULTIPLE] Cart updated with multiple items`,
			"INFO",
			req.session.cart
		);
		log(
			`[CART_ADD_MULTIPLE] === Add Multiple to Cart Request Complete ===`
		);
		res.json({
			success: true,
			message: "Products added to cart successfully",
			cart: req.session.cart,
		});
	});
});

router.post("/cart/remove", (req, res) => {
	log(`[CART_REMOVE] === Processing Remove from Cart Request ===`);
	log(`[CART_REMOVE] Request body:`, "INFO", req.body);
	const { productId } = req.body;

	if (!req.session.cart) {
		req.session.cart = [];
	}

	const initialLength = req.session.cart.length;
	req.session.cart = req.session.cart.filter(
		(item) => item.productId !== productId
	);

	// Save session
	req.session.save((err) => {
		if (err) {
			log(
				`[CART_REMOVE] Error saving session: ${err.message}`,
				"ERROR"
			);
			return res.status(500).json({
				success: false,
				message: "Failed to remove item from cart",
				error: err.message,
			});
		}

		const removed = initialLength > req.session.cart.length;
		log(`[CART_REMOVE] Cart updated after removal`, "INFO", {
			removed,
			newCartLength: req.session.cart.length,
			cart: req.session.cart,
		});
		log(`[CART_REMOVE] === Remove from Cart Request Complete ===`);
		res.json({
			success: true,
			message: removed
				? "Item removed from cart"
				: "Item not found in cart",
			cart: req.session.cart,
		});
	});
});

// Express route for setting quantity
router.post("/cart/set-quantity", (req, res) => {
	const { productId, quantity } = req.body;

	// Get user's cart from session or DB
	const cart = req.session.cart || [];

	// Find the item
	const item = cart.find((p) => p.productId === productId);

	if (item) {
		// Replace the quantity
		item.quantity = quantity;
	} else {
		// Add as new item
		cart.push({ productId, quantity });
	}

	req.session.cart = cart;
	res.json({ success: true });
});

router.post("/chatbot/image", (req, res) => {
	log(`[CHATBOT_IMAGE] === Processing Chatbot Image Upload ===`);
	const { image } = req.body;
	if (image) {
		// Initialize chat history if it doesn't exist
		if (!req.session.chatHistory) {
			req.session.chatHistory = {
				image: null,
				userText: null,
			};
		}

		// Store image in chat history
		req.session.chatHistory.image = {
			content: image,
			timestamp: new Date().toISOString(),
		};

		res.json({ success: true });
	} else {
		log(`[CHATBOT_IMAGE] No image provided`, "WARN");
		res.json({ success: false });
	}
	log(`[CHATBOT_IMAGE] === Chatbot Image Upload Complete ===`);
});

router.post("/chatbot/text", (req, res) => {
	log(`[CHATBOT_TEXT] === Processing Chatbot Text Input ===`);
	const { text } = req.body;
	if (text) {
		// Initialize chat history if it doesn't exist
		if (!req.session.chatHistory) {
			req.session.chatHistory = {
				image: null,
				userText: null,
			};
		}

		// Store text in chat history
		req.session.chatHistory.userText = {
			content: text,
			timestamp: new Date().toISOString(),
		};

		log(`[CHATBOT_TEXT] Text stored in session`, "INFO", {
			hasText: true,
		});
		res.json({ success: true });
	} else {
		log(`[CHATBOT_TEXT] No text provided`, "WARN");
		res.json({ success: false });
	}
	log(`[CHATBOT_TEXT] === Chatbot Text Input Complete ===`);
});

// Initialize chat session with Q&A flow
router.post("/chatbot/init-session", async (req, res) => {
	log(`[CHATBOT_INIT] === Processing Chat Session Initialization ===`);
	try {
		// Initialize chat history with the new structure
		if (!req.session.chatHistory) {
			req.session.chatHistory = {
				image: null,
				userText: null,
			};
		}

		// Initialize recommendation results with the new structure
		if (!req.session.recommendationResults) {
			req.session.recommendationResults = {
				similarProducts: [],
				recommendProducts: [],
				meaningfulCaption: "",
			};
		}

		log(`[CHATBOT_INIT] === Chat Session Initialization Complete ===`);
		res.json({ success: true });
	} catch (error) {
		log(
			`[CHATBOT_INIT] Error initializing chat session: ${error.message}`,
			"ERROR"
		);
		res.status(500).json({
			success: false,
			message: "Failed to initialize chat session",
			error: error.message,
		});
	}
});

// Get current question
router.get("/chatbot/current-question", (req, res) => {
	log(`[CHATBOT_QUESTION] === Processing Current Question Request ===`);
	try {
		const { chatbotState, chatbotQuestions } = req.session;

		if (
			!chatbotState ||
			!chatbotQuestions ||
			chatbotQuestions.length === 0
		) {
			return res.json({
				success: false,
				message: "No questions available",
			});
		}

		const currentQuestion =
			chatbotQuestions[chatbotState.currentQuestionIndex];
		log(`[CHATBOT_QUESTION] Retrieved current question`, "INFO", {
			questionIndex: chatbotState.currentQuestionIndex,
			question: currentQuestion,
		});

		res.json({
			success: true,
			question: currentQuestion,
			isComplete: chatbotState.isComplete,
			currentIndex: chatbotState.currentQuestionIndex,
			totalQuestions: chatbotQuestions.length,
		});
	} catch (error) {
		log(
			`[CHATBOT_QUESTION] Error getting current question: ${error.message}`,
			"ERROR"
		);
		res.status(500).json({
			success: false,
			message: "Failed to get current question",
			error: error.message,
		});
	}
});

// Process answer and get next question
router.post("/chatbot/answer", async (req, res) => {
	log(`[CHATBOT_ANSWER] === Processing Answer ===`);
	try {
		const { answer } = req.body;
		const { chatbotState, chatbotQuestions } = req.session;

		if (!chatbotState || !chatbotQuestions) {
			return res.status(400).json({
				success: false,
				message: "Chat session not initialized",
			});
		}

		// Store the answer
		chatbotState.answers[chatbotState.currentQuestionIndex] = answer;

		// Add answer to chat history
		req.session.chatHistory.push({
			type: "text",
			content: answer,
			sender: "user",
			timestamp: new Date().toISOString(),
		});

		// Get next question or recommendations
		try {
			const response = await axios.post(
				"http://localhost:5001/chatbot/process-answer",
				{
					questionIndex: chatbotState.currentQuestionIndex,
					answer: answer,
					answers: chatbotState.answers,
				}
			);

			// Add bot response to chat history
			if (response.data.botResponse) {
				req.session.chatHistory.push({
					type: "text",
					content: response.data.botResponse,
					sender: "bot",
					timestamp: new Date().toISOString(),
				});
			}

			// Update state
			chatbotState.currentQuestionIndex++;
			chatbotState.isComplete = response.data.isComplete;

			if (response.data.recommendations) {
				req.session.recommendationResults =
					response.data.recommendations;
			}

			log(`[CHATBOT_ANSWER] Processed answer`, "INFO", {
				questionIndex: chatbotState.currentQuestionIndex - 1,
				isComplete: chatbotState.isComplete,
			});

			res.json({
				success: true,
				isComplete: chatbotState.isComplete,
				botResponse: response.data.botResponse,
				hasRecommendations: !!response.data.recommendations,
			});
		} catch (error) {
			log(
				`[CHATBOT_ANSWER] Error processing answer: ${error.message}`,
				"ERROR"
			);
			res.status(500).json({
				success: false,
				message: "Failed to process answer",
				error: error.message,
			});
		}
	} catch (error) {
		log(
			`[CHATBOT_ANSWER] Error handling answer: ${error.message}`,
			"ERROR"
		);
		res.status(500).json({
			success: false,
			message: "Failed to handle answer",
			error: error.message,
		});
	}
});

// Modify clear history to match new structure
router.post("/chatbot/clear-history", (req, res) => {
	log(`[CHATBOT_CLEAR] === Processing Clear History Request ===`);
	try {
		// Clear all chatbot-related session data
		req.session.chatHistory = {
			image: null,
			userText: null,
		};
		req.session.recommendationResults = {
			similarProducts: [],
			recommendProducts: [],
			meaningfulCaption: "",
		};

		log(`[CHATBOT_CLEAR] Session cleared successfully`, "INFO");
		log(`[CHATBOT_CLEAR] === Clear History Request Complete ===`);
		res.json({ success: true, message: "History cleared successfully" });
	} catch (error) {
		log(
			`[CHATBOT_CLEAR] Error clearing history: ${error.message}`,
			"ERROR"
		);
		res.status(500).json({
			success: false,
			message: "Failed to clear history",
			error: error.message,
		});
	}
});

// Payment page route
router.get("/payment", isAuthenticated, (req, res) => {
	log(`[PAYMENT] === Processing Payment Page Request ===`);
	// Check if user is authenticated
	if (!req.session.user) {
		log(`[PAYMENT] Unauthorized access attempt`, "WARN");
		return res.redirect(
			"/login?error=Please login to proceed to payment"
		);
	}
	// Get cart totals from session
	const cartTotals = req.session.cartTotals || {
		totalPrice: 0,
		totalQuantity: 0,
	};

	log(`[PAYMENT] Cart totals:`, "INFO", cartTotals);
	log(`[PAYMENT] === Payment Page Processing Complete ===`);

	res.render("payment", {
		cartTotals,
	});
});

// Update process payment route to require authentication
router.post("/process-payment", isAuthenticated, (req, res) => {
	log(`[PROCESS_PAYMENT] === Processing Payment Request ===`);
	if (!req.session.user) {
		log(`[PROCESS-PAYMENT] Unauthorized access attempt`, "WARN");
		return res.redirect(
			"/login?error=Please login to proceed to payment"
		);
	}
	try {
		const paymentData = req.body;

		// Log payment data (excluding sensitive information)
		const safePaymentData = {
			...paymentData,
			paymentDetails:
				paymentData.paymentMethod === "upi"
					? { upiId: "***" }
					: {
							cardNumber: "**** **** **** ****",
							expiryDate: "**/**",
							cvv: "***",
					  },
		};
		log(
			`[PROCESS_PAYMENT] Payment data received`,
			"INFO",
			safePaymentData
		);

		// Here you would typically:
		// 1. Validate the payment data
		// 2. Process the payment through a payment gateway
		// 3. Create an order in the database
		// 4. Clear the cart
		// 5. Send confirmation emails

		// For now, we'll simulate a successful payment
		req.session.cartItems = []; // Clear the cart
		req.session.orderId = `ORD${Date.now()}`; // Generate a mock order ID

		log(`[PROCESS_PAYMENT] Payment processed successfully`, "INFO", {
			orderId: req.session.orderId,
		});

		res.json({
			success: true,
			message: "Payment processed successfully",
			orderId: req.session.orderId,
		});
	} catch (error) {
		log(
			`[PROCESS_PAYMENT] Error processing payment: ${error.message}`,
			"ERROR"
		);
		res.status(500).json({
			success: false,
			message: "Error processing payment",
		});
	}
});

// Order confirmation page route
router.get("/order-confirmation", (req, res) => {
	log(
		`[ORDER_CONFIRMATION] === Processing Order Confirmation Page Request ===`
	);
	if (!req.session.user) {
		log(`[ORDER_CONFIRMATION] Unauthorized access attempt`, "WARN");
		return res.redirect(
			"/login?error=Please login to proceed to payment"
		);
	}
	try {
		const orderId = req.session.orderId;

		if (!orderId) {
			log(`[ORDER_CONFIRMATION] No order ID found in session`, "WARN");
			return res.redirect("/");
		}

		log(
			`[ORDER_CONFIRMATION] Rendering order confirmation page`,
			"INFO",
			{
				orderId,
			}
		);

		res.render("order-confirmation", {
			orderId,
		});
	} catch (error) {
		log(
			`[ORDER_CONFIRMATION] Error rendering order confirmation page: ${error.message}`,
			"ERROR"
		);
		res.status(500).send("Error loading order confirmation page");
	}
});

router.get("/session-debug", (req, res) => {
	log(`[SESSION_DEBUG] === Processing Session Debug Request ===`);
	log(`[SESSION_DEBUG] === Session Debug Complete ===`);
	res.json(req.session);
});

// Mock user database (in a real app, this would be in a database)
const users = [
	{
		id: 1,
		email: "ee24b024@smail.iitm.ac.in",
		username: "user",
		password: "password", // In a real app, this would be hashed
		name: "Test User",
		role: "admin",
	},
	{
		id: 2,
		email: "mm24b024@smail.iitm.ac.in",
		username: "admin",
		password: "password",
		name: "Admin User",
		role: "admin",
	},
	{
		id: 3,
		email: "ce24b119@smail.iitm.ac.in",
		username: "admin",
		password: "password",
		name: "Admin User",
		role: "admin",
	},
	{
		id: 4,
		email: "cs24b033@smail.iitm.ac.in",
		username: "admin",
		password: "password",
		name: "Admin User",
		role: "admin",
	},
	{
		id: 5,
		email: "username",
		username: "admin",
		password: "password",
		name: "Admin User",
		role: "user",
	},
];

// Login endpoint
router.post("/login", async (req, res) => {
	log(`[LOGIN] === Processing Login Request ===`);
	try {
		const { email, password } = req.body;

		// Validate input
		if (!email || !password) {
			log(`[LOGIN] Missing credentials`, "WARN");
			return res.status(400).json({
				success: false,
				message: "Please provide both email and password",
			});
		}

		// Find user by email or username
		const user = users.find(
			(u) =>
				u.email.toLowerCase() === email.toLowerCase() ||
				u.username.toLowerCase() === email.toLowerCase()
		);

		// Check if user exists and password matches
		if (!user || user.password !== password) {
			log(`[LOGIN] Invalid credentials`, "WARN");
			return res.status(401).json({
				success: false,
				message: "Invalid email/username or password",
			});
		}

		// Create user session
		req.session.user = {
			id: user.id,
			email: user.email,
			username: user.username,
			name: user.name,
			role: user.role,
		};

		// Initialize cart if not exists
		if (!req.session.cart) {
			req.session.cart = [];
		}

		// Save session
		req.session.save((err) => {
			if (err) {
				log(
					`[LOGIN] Error saving session: ${err.message}`,
					"ERROR"
				);
				return res.status(500).json({
					success: false,
					message: "Error creating session",
				});
			}

			log(`[LOGIN] User logged in successfully`, "INFO", {
				userId: user.id,
				username: user.username,
			});

			// Return success response with user data (excluding sensitive info)
			res.json({
				success: true,
				message: "Login successful",
				user: {
					id: user.id,
					email: user.email,
					username: user.username,
					name: user.name,
					role: user.role,
				},
				redirectUrl: user.role === "admin" ? "/" : "/",
			});
		});
	} catch (error) {
		log(`[LOGIN] Error during login: ${error.message}`, "ERROR");
		res.status(500).json({
			success: false,
			message: "An error occurred during login",
		});
	}
	log(`[LOGIN] === Login Request Processing Complete ===`);
});

// Logout endpoint
router.post("/logout", (req, res) => {
	log(`[LOGOUT] === Processing Logout Request ===`);
	try {
		// Clear user session
		req.session.user = null;

		// Save session
		req.session.save((err) => {
			if (err) {
				log(
					`[LOGOUT] Error saving session: ${err.message}`,
					"ERROR"
				);
				return res.status(500).json({
					success: false,
					message: "Error clearing session",
				});
			}

			log(`[LOGOUT] User logged out successfully`);
			res.json({
				success: true,
				message: "Logged out successfully",
			});
		});
	} catch (error) {
		log(`[LOGOUT] Error during logout: ${error.message}`, "ERROR");
		res.status(500).json({
			success: false,
			message: "An error occurred during logout",
		});
	}
	log(`[LOGOUT] === Logout Request Processing Complete ===`);
});

// Get current user endpoint
router.get("/me", (req, res) => {
	log(`[GET_USER] === Processing Get Current User Request ===`);
	try {
		if (!req.session.user) {
			log(`[GET_USER] No user session found`, "WARN");
			return res.status(401).json({
				success: false,
				message: "Not authenticated",
			});
		}

		log(`[GET_USER] User data retrieved`, "INFO", {
			userId: req.session.user.id,
			username: req.session.user.username,
		});

		res.json({
			success: true,
			user: req.session.user,
		});
	} catch (error) {
		log(`[GET_USER] Error getting user data: ${error.message}`, "ERROR");
		res.status(500).json({
			success: false,
			message: "Error retrieving user data",
		});
	}
	log(`[GET_USER] === Get Current User Request Complete ===`);
});

// Login page route
router.get("/login", (req, res) => {
	log(`[LOGIN_PAGE] === Processing Login Page Request ===`);
	try {
		// Check if user is already logged in
		if (req.session.user) {
			log(
				`[LOGIN_PAGE] User already logged in, redirecting to home`,
				"INFO",
				{
					userId: req.session.user.id,
					username: req.session.user.username,
				}
			);

			// Redirect based on user role
			return res.redirect(
				req.session.user.role === "admin" ? "/" : "/"
			);
		}

		// Get any error messages from query parameters
		const error = req.query.error;
		const message = req.query.message;

		log(`[LOGIN_PAGE] Rendering login page`, "INFO", {
			hasError: !!error,
			hasMessage: !!message,
		});

		// Render login page with any error messages
		res.render("login", {
			error: error || null,
			message: message || null,
			layout: false, // Don't use the default layout
		});
	} catch (error) {
		log(
			`[LOGIN_PAGE] Error rendering login page: ${error.message}`,
			"ERROR"
		);
		res.status(500).render("error", {
			message: "Error loading login page",
			error: process.env.NODE_ENV === "development" ? error : {},
		});
	}
	log(`[LOGIN_PAGE] === Login Page Request Complete ===`);
});

module.exports = router;
