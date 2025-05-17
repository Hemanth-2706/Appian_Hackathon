# frontend/app.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:8000/recommend"

st.set_page_config(page_title="🛍️ ShopSmarter AI", layout="wide")
st.title("🛍️ ShopSmarter: Personal Shopping Assistant")

st.markdown("---")

uploaded_file = st.file_uploader("📷 Upload an image to find similar products", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)

    if st.button("🔍 Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            response = requests.post(API_URL, files={'file': (uploaded_file.name, uploaded_file, uploaded_file.type)})

            if response.status_code == 200:
                data = response.json()
                image_urls = [f"http://localhost:8000{path}" for path in data["results"]]

                st.success("Here are your recommended products:")

                cols = st.columns(5)  # Show 3 images per row (Amazon-style)
                for idx, url in enumerate(image_urls):
                    col = cols[idx % 5]
                    with col:
                        response = requests.get(url)
                        img = Image.open(BytesIO(response.content))
                        st.image(img, caption=f"Product {idx+1}", use_column_width=True)
            else:
                st.error("❌ Something went wrong while getting recommendations.")
