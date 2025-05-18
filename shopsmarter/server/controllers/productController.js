const Product = require('../models/Product');

exports.getHomePage = async (req, res) => {
  const products = await Product.find().limit(6);
  res.render('home', { products });
};

exports.getRecommendations = async (req, res) => {
  const recommended = await Product.find({ category: 'jacket' });
  res.render('recommendations', { recommended });
};

exports.getProductPage = async (req, res) => {
  const product = await Product.findById(req.params.id);
  res.render('product', { product });
};

exports.getCheckoutPage = async (req, res) => {
  const product = await Product.findById(req.params.id);
  res.render('payment', { product });
};