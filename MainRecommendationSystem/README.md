# 👗 AI Fashion Stylist

Welcome to the **AI Fashion Stylist**! This is a smart desktop app that helps you find similar and complementary fashion items using AI. Upload a clothing image or describe what you want, and get instant recommendations! ✨

## 🚀 Features
- **Image & Text Search:** Upload a photo or type a description to find matching fashion items.
- **AI Recommendations:** Get both similar and complementary outfit suggestions.
- **Chatbot Stylist:** Chat with your AI stylist for personalized advice.
- **Beautiful UI:** Modern, user-friendly interface built with PyQt5.

## 🛠️ How It Works
- Uses **CLIP** and **BLIP** models for image and text understanding.
- Finds similar items using pre-computed embeddings and FAISS for fast search.
- Suggests complementary items using Google Gemini AI.
- Shows product images and details from a local dataset.

## 📦 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd appian_round_2
   ```

2. **Install dependencies**
   Make sure you have Python 3.8+ and pip installed.
   ```bash
   pip install -r requirements.txt
   ```
   
   **Main dependencies:**
   - torch
   - faiss
   - numpy, pandas
   - Pillow
   - transformers
   - google-generativeai
   - requests
   - PyQt5

3. **Prepare the data**
   - Place the following files in the project directory:
     - `img_embs.npy`, `txt_embs.npy`, `filtered_df.csv`, `valid_indices.npy`
     - `fashion-dataset/images.csv` (for product images)
   - Make sure image files are accessible as referenced in `images.csv`.

4. **Run the app**
   ```bash
   python assistant_11.py
   ```

## 🖼️ Usage
- **Upload an image** or **type a description** in the app.
- Click **Find Recommendations** to see similar and complementary items.
- Use the **chatbot** to interact with your AI stylist for more help.

## 💡 Notes
- Requires a GPU for best performance, but will run on CPU (slower).
- Needs internet access for Gemini API (Google Generative AI).
- All data files must be in the correct paths as referenced in the code.

## 🙋‍♂️ Need Help?
If you have issues or questions, feel free to open an issue or contact the author!

---

Enjoy your personalized fashion journey! 👠👜👚 