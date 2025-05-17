import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import random

API_URL = "http://localhost:8000/recommend"

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="🛍️ ShopSmarter AI", layout="wide")
st.markdown("<h1 style='text-align: center;'>🛒 ShopSmarter: AI Personal Shopping Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Input Area (Image + Text)
# -------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h4>📷 Upload an Image</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

with col2:
    st.markdown("<h4>🔎 Or Describe What You're Looking For</h4>", unsafe_allow_html=True)
    search_query = st.text_input("", placeholder="e.g., red leather jacket")

search_button = st.button("🚀 Search")

# -------------------------------
# Fetch Recommendations
# -------------------------------
if search_button and (uploaded_file or search_query):
    with st.spinner("Fetching smart recommendations for you..."):
        files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)} if uploaded_file else None
        data = {'query': search_query} if search_query else {}

        response = requests.post(API_URL, files=files, data=data)

        if response.status_code == 200:
            results = response.json().get("results", [])
            image_urls = [f"http://localhost:8000{img_path}" for img_path in results]

            st.markdown("## 🧠 Smart Recommendations for You:")
            st.markdown("")

            cols = st.columns(5)  # 5 images per row
            image_size = 200  # fixed square size in pixels

            for idx, url in enumerate(image_urls):
                col = cols[idx % 5]
                with col:
                    response = requests.get(url)
                    img = Image.open(BytesIO(response.content))

                    # Center and crop to square
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) / 2
                    top = (height - min_dim) / 2
                    right = (width + min_dim) / 2
                    bottom = (height + min_dim) / 2
                    img = img.crop((left, top, right, bottom))

                    # Resize to square for consistency
                    img = img.resize((image_size, image_size))

                    # st.image(img, caption=f"Product {idx+1}", use_container_width=False)

                    # Render a styled card
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:10px; padding:10px; margin-bottom:15px; background-color:#fff;">
                            <img src="{url}" width="220" style="border-radius:5px; object-fit:cover;" />
                            <h5 style="margin-top:10px;">Product {idx+1}</h5>
                            <p style="margin:0; color:green; font-weight:bold;">₹{random.randint(799, 1999)}</p>
                            <button style="
                                padding:6px 12px;
                                background-color:#ff9900;
                                border:none;
                                border-radius:5px;
                                color:white;
                                margin-top:8px;
                                cursor:pointer;">
                                🛒 Add to Cart
                            </button>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.error("❌ Failed to retrieve recommendations. Please try again.")
