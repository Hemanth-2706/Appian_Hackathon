const express = require("express");
const path = require("path");
const fs = require("fs");
const router = express.Router();

const products = [
	{
		_id: "1",
		name: "Fastrack Smartwatch",
		description: "Stylish and functional",
		price: 1399,
		image: "/images/watch.png",
	},
	{
		_id: "2",
		name: "Men Regular Fit Self Design Built-up Collar Casual Shirt",
		description: "Comfortable and stylish",
		price: 355,
		image: "/images/shirt.png",
	},
	{
		_id: "3",
		name: "Shoes",
		description: "Comfortable and stylish shoes",
		price: 1499,
		image: "/images/shoes.png",
	},
];

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
	const product = products.find((p) => p._id === req.params.id);
	res.render("product", { product });
});

router.get("/checkout/:id", (req, res) => {
	const product = products.find((p) => p._id === req.params.id);
	res.render("payment", { product });
});

module.exports = router;
