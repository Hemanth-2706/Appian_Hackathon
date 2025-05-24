import pandas as pd
import requests
import os

# === LOAD CSV FILES ===
df_recommendations = pd.read_csv("/data/recommendationProducts.csv")
df_similar = pd.read_csv("/data/similarProducts.csv")
df = pd.read_csv("/data/dataset.csv")

# === CONFIGURATION ===
recommendation_ids = df_recommendations["RecommendationProdId"].astype(str).tolist()
similar_ids = df_similar["SimilarProdId"].astype(str).tolist()
output_recommendation_folder = r"C:\Not Synced Storage\Hackathons\Appian Hackathon\Round 2\Github Repo\Appian_Hackathon\shopsmarter\server\data\images\recommendationProducts"
output_similar_folder = r"C:\Not Synced Storage\Hackathons\Appian Hackathon\Round 2\Github Repo\Appian_Hackathon\shopsmarter\server\data\images\similarProducts"

# === Ensure 'productId' is string for matching ===
df["productId"] = df["productId"].astype(str)

# === Create output folders if they don't exist ===
os.makedirs(output_recommendation_folder, exist_ok=True)
os.makedirs(output_similar_folder, exist_ok=True)

def download_images(product_ids, output_folder):
    filtered_df = df[df["productId"].isin(product_ids)]

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

            with open(image_path, "wb") as f:
                f.write(response.content)

            print(f"✅ Saved: {image_path}")

        except requests.RequestException as e:
            print(f"❌ Failed to download {url}: {e}")

# === Download images for both sets ===
print("\n--- Downloading Recommendation Product Images ---")
download_images(recommendation_ids, output_recommendation_folder)

print("\n--- Downloading Similar Product Images ---")
download_images(similar_ids, output_similar_folder)
