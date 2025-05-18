const express = require("express");
const path = require("path");
const app = express();
const routes = require("./routes/mockRoutes");

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "../client/views"));
app.use(express.static(path.join(__dirname, "../client/public")));

app.use("/", routes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
