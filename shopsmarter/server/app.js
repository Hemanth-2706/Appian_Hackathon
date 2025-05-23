process.on("warning", (warning) => {
	console.log(warning.name); // 'DeprecationWarning'
	console.log(warning.message); // warning message
	console.trace();
});

const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });
const express = require("express");
const app = express();
const session = require("express-session");
const routes = require("./routes/mockRoutes");

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "../client/views"));
app.use(express.static(path.join(__dirname, "../client/public")));

app.use(
	session({
		secret: process.env.SECRET_KEY, // 🔐 Use a strong secret in production
		resave: false, // Don't save session if unmodified
		saveUninitialized: false, // Don't create session until something stored
		cookie: { secure: false }, // Set to true if using HTTPS
	})
);

app.use("/", routes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
