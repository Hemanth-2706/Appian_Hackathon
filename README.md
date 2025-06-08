# <p align="center">🛍️ ShopSmarter 🧠</p>

## <p align="center">AI-Powered Personal Shopping Assistant for E-Commerce 🛒</p>

Welcome to **ShopSmarter**, your intelligent shopping companion!  
Combining the power of **Computer Vision**, **Recommendation Systems**, and **Conversational AI**, ShopSmarter personalizes and automates your shopping experience like never before.

---

## 🚀 What is ShopSmarter?

Imagine seeing an outfit you love on Instagram, snapping a photo, and instantly getting similar (or even complementary!) products from your favorite online store.  
**ShopSmarter** makes that vision a reality by turning image inputs into smart, personalized recommendations across an e-commerce platform.

<<<<<<< Updated upstream
---
=======
-    Node.js (v22.11.0)
-    Python 3.9 or higher
-    GPU (recommended for better performance, but will run on CPU also)

### Python Dependencies

-    FastAPI
-    Uvicorn
-    PyTorch (>=1.10.0)
-    FAISS-CPU (>=1.7.2)
-    NumPy (>=1.21.0)
-    Pandas (>=1.3.0)
-    Pillow (>=8.3.0)
-    Transformers (>=4.30.0)
-    Google Generative AI (>=0.3.0)
-    Requests (>=2.26.0)
-    PyQt5 (>=5.15.0)
-    Sentence Transformers

### Node.js Dependencies

-    Express.js
-    EJS
-    Axios
-    Dotenv
-    Express-session
-    Mongoose

### Backend Setup (Server)

1. Install Node.js dependencies:

```bash
npm install
```

2. For the chatbot recommendation system to work, you also have to start a python server in port 5001:

```bash
cd server
python api_server.py
```

Wait for all the model to load and the server to start

3. Then start the Node server in port 5003:
   Open another terminal simultaneously and run this

```bash
cd server
node app.js
```

You can use the website in http://localhost:5003

### Recommendation System Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Unix/macOS
# or
env\Scripts\activate     # On Windows
```

2. Navigate to the recommendation system directory:

```bash
cd MainRecommendationSystem
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

4. Start the recommendation service:

```bash
python service.py
```
>>>>>>> Stashed changes

## 🎯 Key Features

✨ **Image Uploads**  
Users can upload photos of products (clothes, accessories, furniture, gadgets, etc.).

🧠 **Visual Understanding**  
AI models analyze the image to extract features like:

-    Color 🎨
-    Texture 🔍
-    Category 🧥📱🛋️
-    Style 👗
-    Brand-like attributes 👟

🔁 **Smart Recommendations**  
Get visually similar items or complementary product suggestions pulled directly from the store's catalog.

❤️ **Personalized Shopping**  
Recommendations adapt based on user:

-    Preferences
-    Shopping behavior
-    Optional style choices

💬 **Multimodal Interaction**  
Combine image uploads with prompts like:

> "Show me similar jackets"  
> "Find matching lamps for this room"

🤖 **Agentic AI for Queries & Checkout**  
The assistant can handle follow-up questions, modifications, and even **automate the checkout** process!

🧩 **Modular & Scalable**  
Built to scale with flexible components and a robust backend for product ranking and retrieval.

---

## 🧪 Example Use Cases

📸 _User uploads a picture of sneakers_  
→ ShopSmarter suggests similar sneakers + matching athletic wear 👕🧦

🏡 _User snaps their living room_  
→ Suggestions for matching lamps, decor, and furniture 🛋️💡

📱 _User uploads a product photo from social media_  
→ Find look-alikes or alternatives in the store instantly 🔎🛍️

---

## 🛠️ Tech Stack

| Layer                    | Technology Used                                                         |
| ------------------------ | ----------------------------------------------------------------------- |
| 👁️ Image Processing      | CNN / Vision Transformers (ViT), OpenCV                                 |
| 🧠 Feature Extraction    | CLIP / ResNet / Custom Embeddings                                       |
| 🔍 Recommendation Engine | KNN / Cosine Similarity / Matrix Factorization / Reranking with XGBoost |
| 🗣️ Conversational Layer  | LangChain / LLM APIs / RAG Pipelines                                    |
| 🧰 Backend Services      | FastAPI / Flask / Node.js                                               |
| 🗃️ Database              | MongoDB / PostgreSQL / Redis                                            |
| 💅 Frontend              | React / Tailwind / Next.js                                              |
| ☁️ Hosting & Infra       | AWS / GCP / Vercel / Docker                                             |

---

## 🧠 Future Ideas

-    🧑‍🎨 Style Quiz for preference learning
-    📦 AR try-on or room placement previews
-    🧵 Cross-category bundles (e.g., whole outfits or room setups)
-    🌐 Browser extension for direct image capture

---

## 🎉 Team ShopSmarter

Made with ❤️ for the [Your Hackathon Name]  
By: [Team Members]

---

## 📸 Try It Out!

1. Upload a product photo 📤
2. Chat with the assistant 💬
3. Discover and shop your style 🛍️

---

## 🗂️ Project Structure

shop-smarter/
│
├── backend/ # FastAPI/Flask backend
│ ├── api/
│ ├── models/
│ └── recommender/
│
├── frontend/ # React or Next.js UI
│ ├── components/
│ └── pages/
│
├── data/ # Product catalog, embeddings
├── notebooks/ # Prototyping CV/ML models
└── README.md # You're here!

---

## 📬 Contact Us

For questions, suggestions, or collabs:  
📧 yourteam@email.com  
🐦 @yourhandle

---

Let’s make shopping smarter, simpler, and more **YOU**. ✨
