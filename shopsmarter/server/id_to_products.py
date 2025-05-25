import pandas as pd
import requests
import os
import json

# === LOAD CSV FILES ===
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

df_recommendations = pd.read_csv(os.path.join(data_dir, "recommendProducts.csv"))
df_similar = pd.read_csv(os.path.join(data_dir, "similarProducts.csv"))
df = pd.read_csv(os.path.join(data_dir, "dataset.csv"))

# === CONFIGURATION ===
recommendation_ids = df_recommendations["RecommendationProdId"].astype(str).tolist()
similar_ids = df_similar["SimilarProdId"].astype(str).tolist()
output_recommendation_folder = os.path.join(data_dir, "images", "recommendProducts")
output_similar_folder = os.path.join(data_dir, "images", "similarProducts")
products_js_path = os.path.join(data_dir, "products.js")

# === Ensure 'productId' is string for matching ===
df["productId"] = df["productId"].astype(str)

# === Create output folders if they don't exist ===
os.makedirs(output_recommendation_folder, exist_ok=True)
os.makedirs(output_similar_folder, exist_ok=True)

def download_images(product_ids, output_folder):
    filtered_df = df[df["productId"].isin(product_ids)]
    image_extensions = {}  # Store product ID to image extension mapping

    for _, row in filtered_df.iterrows():
        pid = row["productId"]
        url = row["link"]

        print(f"📥 Downloading image for Product ID {pid}...")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Determine image extension
            ext = os.path.splitext(url)[-1]
            if not ext or len(ext) > 5:
                ext = ".jpg"

            image_path = os.path.join(output_folder, f"{pid}{ext}")
            image_extensions[pid] = ext

            with open(image_path, "wb") as f:
                f.write(response.content)

            print(f"✅ Saved: {image_path}")

        except requests.RequestException as e:
            print(f"❌ Failed to download {url}: {e}")
            image_extensions[pid] = ".jpg"  # Default extension if download fails

    return image_extensions

def create_product_object(row, image_path):
    """Create a product object with all features from the dataset"""
    return {
        "productId": str(row["productId"]),
        "productName": row.get("productDisplayName", ""),
        "gender": row.get("gender", ""),
        "masterCategory": row.get("masterCategory", ""),
        "subCategory": row.get("subCategory", ""),
        "articleType": row.get("articleType", ""),
        "baseColour": row.get("baseColour", ""),
        "season": row.get("season", ""),
        "usage": row.get("usage", ""),
        "image": image_path,
        "price": float(row.get("price", 0))
    }

def create_products_js(similar_extensions, recommend_extensions):
    """Create the products.js file with dynamic data"""
    # Get base products (first 3 products from dataset)
    base_products = df.head(3).apply(
        lambda row: create_product_object(row, f"/images/products/{row['productId']}.jpg"),
        axis=1
    ).tolist()

    # Get similar products
    similar_products = df[df["productId"].isin(similar_ids)].apply(
        lambda row: create_product_object(
            row,
            f"/images/similarProducts/{row['productId']}{similar_extensions.get(str(row['productId']), '.jpg')}"
        ),
        axis=1
    ).tolist()

    # Get recommended products
    recommend_products = df[df["productId"].isin(recommendation_ids)].apply(
        lambda row: create_product_object(
            row,
            f"/images/recommendProducts/{row['productId']}{recommend_extensions.get(str(row['productId']), '.jpg')}"
        ),
        axis=1
    ).tolist()

    # Create the JavaScript content
    js_content = f"""const products = {json.dumps(base_products, indent=2)};

const similarProducts = {json.dumps(similar_products, indent=2)};

const recommendProducts = {json.dumps(recommend_products, indent=2)};

module.exports = {{
    products,
    similarProducts,
    recommendProducts,
}};
"""
    
    # Write to products.js
    with open(products_js_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"✅ Created products.js at {products_js_path}")

# === Download images for both sets ===
print("\n--- Downloading Recommendation Product Images ---")
recommend_extensions = download_images(recommendation_ids, output_recommendation_folder)

print("\n--- Downloading Similar Product Images ---")
similar_extensions = download_images(similar_ids, output_similar_folder)

# === Create products.js file ===
print("\n--- Creating products.js file ---")
create_products_js(similar_extensions, recommend_extensions)
