const express = require("express");
const router = express.Router();

const products = [
	{
		_id: "1",
		name: "Stylish Jacket",
		description: "Warm and stylish",
		price: 59.99,
		image: "https://via.placeholder.com/150",
	},
	{
		_id: "2",
		name: "Cool Sneakers",
		description: "Trendy sneakers",
		price: 49.99,
		image: "https://via.placeholder.com/150",
	},
	{
		_id: "3",
		name: "Desk Lamp",
		description: "Modern lamp",
		price: 24.99,
		image: "https://via.placeholder.com/150",
	},
];

router.get("/", (req, res) => {
	res.render("home", { products });
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
