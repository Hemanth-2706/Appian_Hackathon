# Branch: sbhandari
import streamlit as st
from PIL import Image

st.set_page_config(page_title="ShopSmarter", layout="centered")
st.title("🛍️ ShopSmarter: AI-Powered Personal Shopping Assistant")

st.markdown("Upload an image of a product (like shoes, clothes, decor), and get smart suggestions!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Generating Recommendations...")

    # Simulate AI recommendation output
    with st.spinner("Analyzing..."):
        st.success("Here are some similar products:")

        # Placeholder images for now
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://via.placeholder.com/150?text=Product+1", caption="Product 1")
        with col2:
            st.image("https://via.placeholder.com/150?text=Product+2", caption="Product 2")
        with col3:
            st.image("https://via.placeholder.com/150?text=Product+3", caption="Product 3")
