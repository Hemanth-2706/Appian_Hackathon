import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import random

API_URL = "http://localhost:8000/recommend"

# Set up page
st.set_page_config(page_title="🛍️ ShopSmarter AI", layout="wide")
st.title("🛒 ShopSmarter: Your Personal Shopping Assistant")
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("📷 Upload an image to find similar products", type=["jpg", "jpeg", "png"])

# Show uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)

    if st.button("🔍 Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            response = requests.post(API_URL, files={'file': (uploaded_file.name, uploaded_file, uploaded_file.type)})

            if response.status_code == 200:
                data = response.json()
                image_urls = [f"http://localhost:8000{path}" for path in data["results"]]

                st.success("🧠 AI-Powered Recommendations:")

                # Product grid
                cols = st.columns(4)  # 4 products per row
                for idx, url in enumerate(image_urls):
                    col = cols[idx % 4]
                    with col:
                        # Fetch image
                        img_response = requests.get(url)
                        img = Image.open(BytesIO(img_response.content))

                        # Square cropping (optional)
                        img = img.resize((250, 250))

                        # Card-style display
                        st.markdown(
                            f"""
                            <div style="border:1px solid #ddd; border-radius:10px; padding:10px; margin-bottom:20px; background-color:#fafafa;">
                                <img src="{url}" width="250" style="object-fit: cover; border-radius: 8px;"/>
                                <h5 style="margin-top:10px;">Product {idx+1}</h5>
                                <p style="color:green;"><strong>₹{random.randint(499, 1499)}</strong></p>
                                <button style="padding:8px 16px; background-color:#ff9900; border:none; border-radius:5px; color:white; cursor:pointer;">
                                    Buy Now
                                </button>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.error("❌ Something went wrong while getting recommendations.")
