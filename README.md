# 🛍️ ShopSmarter 🧠

An AI-Powered Personal Shopping Assistant for E-Commerce that combines Computer Vision, Recommendation Systems, and Conversational AI to enhance your shopping experience.

## 🗂️ Project Structure

```
ShopSmarter/
├── client/                      # Frontend EJS application
│   ├── public/                  # Static assets
│   │   ├── css/                # Stylesheets
│   │   ├── js/                 # Client-side JavaScript
│   │   └── images/             # Image assets
│   └── views/                  # EJS templates
│       └── partials/           # Reusable template parts
│
├── server/                      # Express.js backend server
│   ├── controllers/            # Route controllers
│   ├── models/                 # Data models
│   ├── routes/                 # API routes
│   ├── data/                   # Mock data and images
│   └── app.js                  # Main server file
│
└── MainRecommendationSystem/    # ML recommendation engine
    ├── models/                 # Trained ML models
    └── data/                   # Training data
```

## 🚀 Setup Instructions

### Prerequisites

- Node.js (v14 or higher)
- Python 3.9 or higher
- npm or yarn
- Virtual environment tool (venv, conda)

### Frontend Setup (Client)

1. Install Node.js dependencies:
```bash
npm install
```

2. Create `.env` file in the root directory:
```env
PORT=3000
```

3. Start the frontend development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

### Backend Setup (Server)

1. Navigate to the server directory:
```bash
cd server
```

2. Install dependencies:
```bash
npm install
```

3. Create `.env` file in the server directory:
```env
PORT=5000
MONGODB_URI=your_mongodb_uri
```

4. Start the server:
```bash
npm start
```

The API will be available at `http://localhost:5000`

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

## 🎯 Key Features

✨ **Image Uploads**  
- Upload photos of products (clothes, accessories, furniture, gadgets)
- Supported formats: JPG, PNG, WEBP

🧠 **Visual Understanding**  
AI models analyze images to extract:
- Color schemes
- Textures
- Product categories
- Style attributes
- Brand characteristics

🔁 **Smart Recommendations**  
- Similar product suggestions
- Complementary item recommendations
- Style-based matching

❤️ **Personalized Shopping**  
Recommendations adapt based on:
- User preferences
- Shopping history
- Style choices

💬 **Multimodal Interaction**  
Natural language queries with image support:
- "Show me similar jackets"
- "Find matching accessories"
- "What goes well with this?"

## 🧪 Testing

### Frontend Tests
```bash
cd client
npm test
```

### Backend Tests
```bash
cd server
npm test
```

### Recommendation System Tests
```bash
cd MainRecommendationSystem
python -m pytest
```

## 👥 Team

- Smitali Bhandari
- Hemanth
- Shaurya

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
