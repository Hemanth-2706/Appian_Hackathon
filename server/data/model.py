import os, torch, faiss
import numpy as np
import pandas as pd
from PIL import Image
import requests
import json
import logging
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model.log')
    ]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# 3a) CLIP for visual search (lighter model + quantization)
logger.info("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).half()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
logger.info("CLIP model loaded successfully")

# 3b) SBERT for text indexing (lighter model)
logger.info("Loading SBERT model...")
txt_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device=DEVICE)
logger.info("SBERT model loaded successfully")

# 3c) BLIP for captioning (quantized)
logger.info("Loading BLIP model...")
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE).half()
logger.info("BLIP model loaded successfully")

# Configuration
K = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR)
OUTPUT_RECOMMENDATION_FOLDER = os.path.join(DATA_DIR, "images", "recommendProducts")
OUTPUT_SIMILAR_FOLDER = os.path.join(DATA_DIR, "images", "similarProducts")
PRODUCTS_JS_PATH = os.path.join(DATA_DIR, "products.js")

# Set up Gemini API
logger.info("Configuring Gemini API...")
os.environ["GOOGLE_API_KEY"] = "AIzaSyAM_mhTB1qe4-7QgNy7ONjw9mSob7x5qdw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
logger.info("Gemini API configured successfully")

class FashionRecommender:
    def __init__(self, embeddings_dir="./", output_dir="./"):
        """Initialize the recommender by loading pre-computed embeddings"""
        logger.info("Initializing FashionRecommender...")
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.models_loaded = False
        self.load_models()
        self.load_embeddings()
        self.build_indexes()
        self.load_dataset()
        logger.info("FashionRecommender initialized successfully")
    
    def load_dataset(self):
        """Load the main dataset"""
        logger.info("Loading dataset...")
        self.df = pd.read_csv(os.path.join(DATA_DIR, "filtered_df.csv"), dtype=str)
        # Robust column renaming for productId
        if "productId" not in self.df.columns and "id" in self.df.columns:
            self.df = self.df.rename(columns={"id": "productId"})
        self.df["productId"] = self.df["productId"].astype(str)
        logger.info(f"DF shape: {self.df.shape}")
        logger.info(f"DF head: {self.df.head()}\n")
        logger.info(f"DF columns: {self.df.columns.tolist()}")
        logger.info("Dataset loaded with %d records", len(self.df))
    
    def download_images(self, product_ids, output_folder):
        """Download images for given product IDs"""
        logger.info(f"Downloading images for {len(product_ids)} products to {output_folder}")
        filtered_df = self.df[self.df["productId"].isin(product_ids)]
        image_extensions = {}

        for _, row in filtered_df.iterrows():
            pid = row["productId"]
            url = row["link"]
            logger.info(f"Downloading image for Product ID {pid}...")

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                ext = os.path.splitext(url)[-1]
                if not ext or len(ext) > 5:
                    ext = ".jpg"

                image_path = os.path.join(output_folder, f"{pid}{ext}")
                image_extensions[pid] = ext

                with open(image_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Successfully saved image: {image_path}")

            except requests.RequestException as e:
                logger.error(f"Failed to download image for {pid}: {str(e)}")
                image_extensions[pid] = ".jpg"

        return image_extensions

    def create_product_object(self, row, image_path):
        """Create a product object with all features"""
        try:
            return {
                "productId": str(row["productId"]),
                "productName": str(row.get("productDisplayName", "")),
                "gender": str(row.get("gender", "")),
                "masterCategory": str(row.get("masterCategory", "")),
                "subCategory": str(row.get("subCategory", "")),
                "articleType": str(row.get("articleType", "")),
                "baseColour": str(row.get("baseColour", "")),
                "season": str(row.get("season", "")),
                "usage": str(row.get("usage", "")),
                "image": image_path,
                "price": float(row.get("price", 0))
            }
        except Exception as e:
            logger.error(f"Error creating product object: {str(e)}")
            raise

    def create_products_js(self, similar_ids, recommend_ids, similar_extensions, recommend_extensions):
        """Create the products.js file with dynamic data"""
        logger.info("Creating products.js file...")
        try:
            # Get base products (first 3 products from dataset)
            base_products = self.df.head(3).apply(
                lambda row: self.create_product_object(
                    row,
                    os.path.join("/images", "products", f"{row['productId']}.jpg").replace("\\", "/")
                ),
                axis=1
            ).tolist()
            logger.info(f"Created {len(base_products)} base products")

            # Get similar products
            similar_products = self.df[self.df["productId"].isin(similar_ids)].apply(
                lambda row: self.create_product_object(
                    row,
                    os.path.join("/images", "similarProducts", f"{row['productId']}{similar_extensions.get(str(row['productId']), '.jpg')}").replace("\\", "/")
                ),
                axis=1
            ).tolist()
            logger.info(f"Created {len(similar_products)} similar products")

            # Get recommended products
            recommend_products = self.df[self.df["productId"].isin(recommend_ids)].apply(
                lambda row: self.create_product_object(
                    row,
                    os.path.join("/images", "recommendProducts", f"{row['productId']}{recommend_extensions.get(str(row['productId']), '.jpg')}").replace("\\", "/")
                ),
                axis=1
            ).tolist()
            logger.info(f"Created {len(recommend_products)} recommended products")

            # Create the JavaScript content with proper formatting
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
            with open(PRODUCTS_JS_PATH, 'w', encoding='utf-8') as f:
                f.write(js_content)
            
            logger.info(f"✅ Created products.js at {PRODUCTS_JS_PATH}")
        except Exception as e:
            logger.error(f"Error creating products.js: {str(e)}")
            raise

    def clear_output_directories(self):
        """Clear existing images from output directories"""
        logger.info("Clearing output directories...")
        try:
            # Clear similarProducts directory
            if os.path.exists(OUTPUT_SIMILAR_FOLDER):
                for file in os.listdir(OUTPUT_SIMILAR_FOLDER):
                    file_path = os.path.join(OUTPUT_SIMILAR_FOLDER, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleared {OUTPUT_SIMILAR_FOLDER}")
            
            # Clear recommendProducts directory
            if os.path.exists(OUTPUT_RECOMMENDATION_FOLDER):
                for file in os.listdir(OUTPUT_RECOMMENDATION_FOLDER):
                    file_path = os.path.join(OUTPUT_RECOMMENDATION_FOLDER, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logger.info(f"Cleared {OUTPUT_RECOMMENDATION_FOLDER}")
        except Exception as e:
            logger.error(f"Error clearing directories: {str(e)}")
            raise

    def process_recommendations(self, sim_results, comp_results):
        """Process recommendation results and create products.js"""
        try:
            # Clear existing images first
            self.clear_output_directories()
            
            # Ensure output directories exist
            os.makedirs(OUTPUT_RECOMMENDATION_FOLDER, exist_ok=True)
            os.makedirs(OUTPUT_SIMILAR_FOLDER, exist_ok=True)

            # Extract product IDs from results
            similar_ids = sim_results["productId"].astype(str).tolist() if sim_results is not None else []
            recommend_ids = comp_results["productId"].astype(str).tolist() if comp_results is not None else []

            # Download images
            logger.info("\n--- Downloading Recommendation Product Images ---")
            recommend_extensions = self.download_images(recommend_ids, OUTPUT_RECOMMENDATION_FOLDER)

            logger.info("\n--- Downloading Similar Product Images ---")
            similar_extensions = self.download_images(similar_ids, OUTPUT_SIMILAR_FOLDER)

            # Create products.js
            logger.info("\n--- Creating products.js file ---")
            self.create_products_js(similar_ids, recommend_ids, similar_extensions, recommend_extensions)
            
            logger.info("Successfully processed recommendations")
        except Exception as e:
            logger.error(f"Error processing recommendations: {str(e)}")
            raise

    def load_models(self):
        """Load all models needed for inference"""
        if self.models_loaded:
            return
            
        logger.info("Loading models...")
        
        # CLIP for embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).half()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # BLIP for captioning
        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE).half()
        
        self.models_loaded = True
        logger.info("Models loaded successfully!")
    
    def load_embeddings(self):
        """Load pre-computed embeddings and metadata"""
        logger.info("Loading embeddings and metadata...")
        self.img_embs = np.load(os.path.join(DATA_DIR, "img_embs.npy"))
        self.txt_embs = np.load(os.path.join(DATA_DIR, "txt_embs.npy"))
        self.valid_indices = np.load(os.path.join(DATA_DIR, "valid_indices.npy"))
        logger.info(f"img_embs shape: {self.img_embs.shape}, sample: {self.img_embs[0][:5]}")
        logger.info(f"txt_embs shape: {self.txt_embs.shape}, sample: {self.txt_embs[0][:5]}")
        logger.info(f"valid_indices shape: {self.valid_indices.shape}, sample: {self.valid_indices[:10]}")
    
    def build_indexes(self):
        """Build FAISS indexes from loaded embeddings"""
        logger.info("Building FAISS indexes...")
        
        # Fused index for multimodal search
        fused_embs = np.concatenate([self.img_embs, self.txt_embs], axis=1).astype("float32")
        faiss.normalize_L2(fused_embs)
        self.sim_index = faiss.IndexFlatIP(fused_embs.shape[1])
        self.sim_index.add(fused_embs)
        
        # Text-only index
        self.txt_index = faiss.IndexFlatIP(self.txt_embs.shape[1])
        self.txt_index.add(self.txt_embs)
        
        logger.info("Indexes built successfully!")
    
    def embed_image(self, image_path):
        """Generate embedding for a new image"""
        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.warning(f"Warning: Image not found: {image_path}")
            return None
        
        inp = self.clip_proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inp)
            emb /= emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().astype('float32')
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        inputs = self.clip_proc(text=[text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            txt_feats = self.clip_model.get_text_features(**inputs)
        arr = txt_feats.cpu().numpy().astype("float32")
        faiss.normalize_L2(arr)
        return arr
    
    def generate_caption(self, image_path: str):
        """Generate caption for an image using BLIP"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.blip_proc(images=image, return_tensors="pt").to(DEVICE)
            out = self.blip_model.generate(**inputs, max_new_tokens=64)
            caption = self.blip_proc.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return ""
    
    def generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return ""
    
    def ask_complements_local(self, caption, user_prompt, k=K):
        """Get complementary item recommendations using Gemini"""
        prompt = (
            f"You are a professional fashion stylist."
            f"\nYou are given a product: \"{caption}\"."
            f"\nCustomer said: \"{user_prompt}\"."
            f"\nList exactly 5 complementary and matching items for this."
            f"\nEach item MUST follow this format strictly:"
            f"\nCategory: <category>; Article Type: <article_type>; Color/Style: <color_or_style>; Usage: <usage>"
            f"\nSeparate each item with '//' on a single line."
            f"\nDO NOT include any explanations or extra text. Only output the 5 formatted items."
            f"\nDo not give same item as shown in the image in the recommendations."
        )
        
        out = self.generate_with_gemini(prompt)
        items = [itm.strip() for itm in out.split('//') if itm.strip()][:k]
        numbered_items = [f"{i+1}. {itm}" for i, itm in enumerate(items)]
        return numbered_items
    
    def recommend(self, img=None, prompt=None, k=K):
        """Complete recommendation function"""
        logger.info(f"recommend called with img: {img}, prompt: {prompt}")
        has_img = img is not None
        has_txt = prompt is not None
        
        if not has_img and not has_txt:
            logger.error("Both image and prompt are missing")
            return None, None
        
        sim_df = pd.DataFrame()
        rec_df = pd.DataFrame()
        
        # Visual or Text Similarity
        if has_img:
            logger.info("Processing image input...")
            img_emb = self.embed_image(img)
            if img_emb is None:
                logger.error("Could not process image")
                return None, None
            img_emb = img_emb.astype('float32')
            if len(img_emb.shape) == 1:
                img_emb = img_emb.reshape(1, -1)
            faiss.normalize_L2(img_emb)
            if has_txt:
                logger.info("Processing combined image and text input...")
                txt_emb = self.embed_text(prompt)
                txt_emb = txt_emb.astype('float32')
                if len(txt_emb.shape) == 1:
                    txt_emb = txt_emb.reshape(1, -1)
                faiss.normalize_L2(txt_emb)
                qv = np.concatenate([img_emb, txt_emb], axis=1).astype("float32")
                Dv, Iv = self.sim_index.search(qv, k)
            else:
                logger.info("Processing image-only input...")
                zero_txt = np.zeros((1, self.txt_embs.shape[1]), dtype=np.float32)
                qv = np.concatenate([img_emb, zero_txt], axis=1).astype("float32")
                faiss.normalize_L2(qv)
                Dv, Iv = self.sim_index.search(qv, k)
            sim_df = self.df.iloc[Iv[0]][["productId", "text"]].copy()
            sim_df["score_img"] = Dv[0]
        elif has_txt:
            logger.info("Processing text-only input...")
            txt_emb = self.embed_text(prompt)
            txt_emb = txt_emb.astype('float32')
            if len(txt_emb.shape) == 1:
                txt_emb = txt_emb.reshape(1, -1)
            faiss.normalize_L2(txt_emb)
            Dt, It = self.txt_index.search(txt_emb, k)
            sim_df = self.df.iloc[It[0]][["productId", "text"]].copy()
            sim_df["score_txt"] = Dt[0]
        logger.info(f"Similar product IDs: {sim_df['productId'].tolist() if not sim_df.empty else 'None'}")
        
        # Caption + Complementary Retrieval
        if has_txt or has_img:
            logger.info("Generating complementary recommendations...")
            caption = self.generate_caption(img) if has_img else ""
            if caption:
                logger.info(f"Generated caption: {caption}")
            cats = self.ask_complements_local(caption, prompt if has_txt else "")
            logger.info(f"Generated {len(cats)} complementary categories")
            cand = []
            for cat in cats:
                q_t = self.embed_text(cat)
                q_t = q_t.astype('float32')
                if len(q_t.shape) == 1:
                    q_t = q_t.reshape(1, -1)
                faiss.normalize_L2(q_t)
                Dt, It = self.txt_index.search(q_t, k)
                dfc = self.df.iloc[It[0]][["productId", "text"]].copy()
                dfc["score_txt"] = Dt[0]
                cand.append(dfc)
            if cand:
                all_rec = pd.concat(cand, ignore_index=True)
                unique_rec = (
                    all_rec
                    .sort_values("score_txt", ascending=False)
                    .drop_duplicates(subset="productId", keep="first")
                )
                rec_df = unique_rec.head(k)
        logger.info(f"Recommended product IDs: {rec_df['productId'].tolist() if not rec_df.empty else 'None'}")
        
        # Process recommendations and download images
        self.process_recommendations(sim_df, rec_df)
        
        logger.info("Recommendation process completed successfully")
        return sim_df, rec_df
    
    def get_product_details(self, product_ids):
        """Get detailed information for a list of product IDs"""
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        
        results = []
        for pid in product_ids:
            mask = self.df['productId'] == str(pid)
            if mask.any():
                results.append(self.df[mask].iloc[0])
        
        return pd.DataFrame(results) if results else pd.DataFrame()

# Usage example
if __name__ == "__main__":
    logger.info("Starting Fashion Recommender in standalone mode...")
    recommender = FashionRecommender()
    logger.info("Fashion Recommender initialized successfully")
    
    logger.info("\n" + "="*60)
    logger.info("FASHION RECOMMENDATION SYSTEM - INFERENCE MODE")
    logger.info("="*60)
    
    # Get user input from session (if available)
    user_text = None  # Will be set from session
    user_image = None  # Will be set from session
    
    # Example: Using session data
    logger.info("\n### Processing User Input ###")
    sim_results, comp_results = recommender.recommend(
        img=user_image,  # Will be None if no image in session
        prompt=user_text,  # Will be None if no text in session
        k=10
    )
    
    logger.info("\n" + "="*60)
    logger.info("INFERENCE COMPLETE - Check output files!")
    logger.info("="*60)