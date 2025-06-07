import os, torch, faiss
import numpy as np
import pandas as pd
from PIL import Image
import requests
import json
import logging
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
import regex as re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server/logs/model.log'),
        logging.FileHandler('server/logs/all_logs.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== Starting Model Initialization ===")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# 3a) CLIP for visual search (lighter model + quantization)
logger.info("=== Loading CLIP Model ===")
logger.info("Initializing CLIP model for visual search...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).half()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
logger.info("CLIP model loaded successfully")
logger.info("=== CLIP Model Loading Complete ===")

# # 3b) SBERT for text indexing (lighter model)
# logger.info("Loading SBERT model...")
# txt_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device=DEVICE)
# logger.info("SBERT model loaded successfully")

# 3c) BLIP for captioning (quantized)
logger.info("=== Loading BLIP Model ===")
logger.info("Initializing BLIP model for image captioning...")
blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE).half()
logger.info("BLIP model loaded successfully")
logger.info("=== BLIP Model Loading Complete ===")

# Configuration
K = 10
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR)
OUTPUT_RECOMMENDATION_FOLDER = os.path.join(DATA_DIR, "images", "recommendProducts")
OUTPUT_SIMILAR_FOLDER = os.path.join(DATA_DIR, "images", "similarProducts")
PRODUCTS_JS_PATH = os.path.join(DATA_DIR, "products.js")

logger.info("=== Configuration Settings ===")
logger.info(f"Base Directory: {BASE_DIR}")
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Output Recommendation Folder: {OUTPUT_RECOMMENDATION_FOLDER}")
logger.info(f"Output Similar Folder: {OUTPUT_SIMILAR_FOLDER}")
logger.info(f"Products JS Path: {PRODUCTS_JS_PATH}")
logger.info("=== Configuration Settings Complete ===")

# Set up Gemini API
logger.info("=== Configuring Gemini API ===")
os.environ["GOOGLE_API_KEY"] = "AIzaSyAM_mhTB1qe4-7QgNy7ONjw9mSob7x5qdw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
logger.info("Gemini API configured successfully")
logger.info("=== Gemini API Configuration Complete ===")

class FashionRecommender:
    def __init__(self, embeddings_dir="./", output_dir="./"):
        """Initialize the recommender by loading pre-computed embeddings"""
        logger.info("=== Initializing FashionRecommender ===")
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.models_loaded = False
        logger.info(f"Embeddings directory: {embeddings_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        logger.info("Loading models...")
        self.load_models()
        logger.info("Loading embeddings...")
        self.load_embeddings()
        logger.info("Building indexes...")
        self.build_indexes()
        logger.info("Loading dataset...")
        self.load_dataset()
        logger.info("=== FashionRecommender Initialization Complete ===")
    
    def load_dataset(self):
        """Load the main dataset"""
        logger.info("=== Loading Dataset ===")
        logger.info(f"Loading dataset from: {os.path.join(DATA_DIR, 'filtered_df.csv')}")
        self.df = pd.read_csv(os.path.join(DATA_DIR, "filtered_df.csv"), dtype=str)
        # Robust column renaming for productId
        if "productId" not in self.df.columns and "id" in self.df.columns:
            logger.info("Renaming 'id' column to 'productId'")
            self.df = self.df.rename(columns={"id": "productId"})
        self.df["productId"] = self.df["productId"].astype(str)
        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"Dataset columns: {self.df.columns.tolist()}")
        logger.info(f"First few records: {self.df.head().to_dict()}")
        logger.info(f"Loaded {len(self.df)} records")
        logger.info("=== Dataset Loading Complete ===")
    
    def download_images(self, product_ids, output_folder):
        """Download images for given product IDs"""
        logger.info(f"=== Starting Image Download for {len(product_ids)} Products ===")
        logger.info(f"Output folder: {output_folder}")
        filtered_df = self.df[self.df["productId"].isin(product_ids)]
        image_extensions = {}

        for _, row in filtered_df.iterrows():
            pid = row["productId"]
            url = row["link"]
            logger.info(f"Processing Product ID {pid}...")

            try:
                logger.info(f"Downloading image from URL: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                ext = os.path.splitext(url)[-1]
                if not ext or len(ext) > 5:
                    ext = ".jpg"
                logger.info(f"Using file extension: {ext}")

                image_path = os.path.join(output_folder, f"{pid}{ext}")
                image_extensions[pid] = ext

                with open(image_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Successfully saved image: {image_path}")

            except requests.RequestException as e:
                logger.error(f"Failed to download image for {pid}: {str(e)}")
                image_extensions[pid] = ".jpg"

        logger.info(f"=== Image Download Complete - Processed {len(image_extensions)} Images ===")
        return image_extensions

    def create_product_object(self, row, image_path):
        """Create a product object with all features"""
        logger.info(f"=== Creating Product Object for ID: {row['productId']} ===")
        try:
            product = {
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
            logger.info(f"Created product object: {product}")
            logger.info("=== Product Object Creation Complete ===")
            return product
        except Exception as e:
            logger.error(f"Error creating product object: {str(e)}")
            raise

    def create_products_js(self, similar_ids, recommend_ids, similar_extensions, recommend_extensions):
        """Create the products.js file with dynamic data"""
        logger.info("=== Creating Products.js File ===")
        try:
            # Get base products (first 3 products from dataset)
            logger.info("Processing base products...")
            base_products = self.df.head(3).apply(
                lambda row: self.create_product_object(
                    row,
                    os.path.join("/images", "products", f"{row['productId']}.jpg").replace("\\", "/")
                ),
                axis=1
            ).tolist()
            logger.info(f"Created {len(base_products)} base products")

            # Get similar products
            logger.info("Processing similar products...")
            similar_products = self.df[self.df["productId"].isin(similar_ids)].apply(
                lambda row: self.create_product_object(
                    row,
                    os.path.join("/images", "similarProducts", f"{row['productId']}{similar_extensions.get(str(row['productId']), '.jpg')}").replace("\\", "/")
                ),
                axis=1
            ).tolist()
            logger.info(f"Created {len(similar_products)} similar products")

            # Get recommended products
            logger.info("Processing recommended products...")
            recommend_products = self.df[self.df["productId"].isin(recommend_ids)].apply(
                lambda row: self.create_product_object(
                    row,
                    os.path.join("/images", "recommendProducts", f"{row['productId']}{recommend_extensions.get(str(row['productId']), '.jpg')}").replace("\\", "/")
                ),
                axis=1
            ).tolist()
            logger.info(f"Created {len(recommend_products)} recommended products")

            # Create the JavaScript content with proper formatting
            logger.info("Generating JavaScript content...")
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
            logger.info(f"Writing to products.js at {PRODUCTS_JS_PATH}")
            with open(PRODUCTS_JS_PATH, 'w', encoding='utf-8') as f:
                f.write(js_content)
            
            logger.info("✅ Successfully created products.js")
            logger.info("=== Products.js Creation Complete ===")
        except Exception as e:
            logger.error(f"Error creating products.js: {str(e)}")
            raise

    def clear_output_directories(self):
        """Clear existing images from output directories"""
        logger.info("=== Clearing Output Directories ===")
        try:
            # Clear similarProducts directory
            if os.path.exists(OUTPUT_SIMILAR_FOLDER):
                logger.info(f"Clearing {OUTPUT_SIMILAR_FOLDER}")
                for file in os.listdir(OUTPUT_SIMILAR_FOLDER):
                    file_path = os.path.join(OUTPUT_SIMILAR_FOLDER, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed file: {file_path}")
                logger.info(f"Cleared {OUTPUT_SIMILAR_FOLDER}")
            
            # Clear recommendProducts directory
            if os.path.exists(OUTPUT_RECOMMENDATION_FOLDER):
                logger.info(f"Clearing {OUTPUT_RECOMMENDATION_FOLDER}")
                for file in os.listdir(OUTPUT_RECOMMENDATION_FOLDER):
                    file_path = os.path.join(OUTPUT_RECOMMENDATION_FOLDER, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed file: {file_path}")
                logger.info(f"Cleared {OUTPUT_RECOMMENDATION_FOLDER}")
            logger.info("=== Output Directories Clearing Complete ===")
        except Exception as e:
            logger.error(f"Error clearing directories: {str(e)}")
            raise

    def process_recommendations(self, sim_results, comp_results):
        """Process recommendation results and create products.js"""
        logger.info("=== Processing Recommendations ===")
        try:
            # Clear existing images first
            logger.info("Clearing existing images...")
            self.clear_output_directories()
            
            # Ensure output directories exist
            logger.info("Creating output directories...")
            os.makedirs(OUTPUT_RECOMMENDATION_FOLDER, exist_ok=True)
            os.makedirs(OUTPUT_SIMILAR_FOLDER, exist_ok=True)

            # Extract product IDs from results
            similar_ids = sim_results["productId"].astype(str).tolist() if sim_results is not None else []
            recommend_ids = comp_results["productId"].astype(str).tolist() if comp_results is not None else []
            logger.info(f"Found {len(similar_ids)} similar products and {len(recommend_ids)} recommended products")

            # Download images
            logger.info("\n--- Downloading Recommendation Product Images ---")
            recommend_extensions = self.download_images(recommend_ids, OUTPUT_RECOMMENDATION_FOLDER)

            logger.info("\n--- Downloading Similar Product Images ---")
            similar_extensions = self.download_images(similar_ids, OUTPUT_SIMILAR_FOLDER)

            # Create products.js
            logger.info("\n--- Creating products.js file ---")
            self.create_products_js(similar_ids, recommend_ids, similar_extensions, recommend_extensions)
            
            logger.info("=== Recommendation Processing Complete ===")
        except Exception as e:
            logger.error(f"Error processing recommendations: {str(e)}")
            raise

    def load_models(self):
        """Load all models needed for inference"""
        if self.models_loaded:
            logger.info("Models already loaded, skipping...")
            return
            
        logger.info("=== Loading Models ===")
        
        # CLIP for embeddings
        logger.info("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).half()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        logger.info("CLIP model loaded successfully")
        
        # BLIP for captioning
        logger.info("Loading BLIP model...")
        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE).half()
        logger.info("BLIP model loaded successfully")
        
        self.models_loaded = True
        logger.info("=== Model Loading Complete ===")
    
    def load_embeddings(self):
        """Load pre-computed embeddings and metadata"""
        logger.info("=== Loading Embeddings ===")
        logger.info("Loading image embeddings...")
        self.img_embs = np.load(os.path.join(DATA_DIR, "img_embs.npy"))
        logger.info(f"Image embeddings shape: {self.img_embs.shape}")
        logger.info(f"Image embeddings sample: {self.img_embs[0][:5]}")
        
        logger.info("Loading text embeddings...")
        self.txt_embs = np.load(os.path.join(DATA_DIR, "txt_embs.npy"))
        logger.info(f"Text embeddings shape: {self.txt_embs.shape}")
        logger.info(f"Text embeddings sample: {self.txt_embs[0][:5]}")
        
        logger.info("Loading valid indices...")
        self.valid_indices = np.load(os.path.join(DATA_DIR, "valid_indices.npy"))
        logger.info(f"img_embs shape: {self.img_embs.shape}, sample: {self.img_embs[0][:5]}")
        logger.info(f"txt_embs shape: {self.txt_embs.shape}, sample: {self.txt_embs[0][:5]}")
        logger.info(f"valid_indices shape: {self.valid_indices.shape}, sample: {self.valid_indices[:10]}")
    
    def build_indexes(self):
        """Build FAISS indexes from loaded embeddings"""
        logger.info("=== Building FAISS Indexes ===")
        
        # Fused index for multimodal search
        logger.info("Building fused index for multimodal search...")
        fused_embs = np.concatenate([self.img_embs, self.txt_embs], axis=1).astype("float32")
        faiss.normalize_L2(fused_embs)
        self.sim_index = faiss.IndexFlatIP(fused_embs.shape[1])
        self.sim_index.add(fused_embs)
        logger.info(f"Fused index built with dimension: {fused_embs.shape[1]}")
        
        # Text-only index
        logger.info("Building text-only index...")
        self.txt_index = faiss.IndexFlatIP(self.txt_embs.shape[1])
        self.txt_index.add(self.txt_embs)
        logger.info(f"Text index built with dimension: {self.txt_embs.shape[1]}")
        
        logger.info("=== FAISS Index Building Complete ===")
    
    def embed_image(self, image_path):
        """Generate embedding for a new image"""
        logger.info(f"=== Generating Image Embedding for: {image_path} ===")
        try:
            img = Image.open(image_path).convert("RGB")
            logger.info("Image loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Warning: Image not found: {image_path}")
            return None
        
        logger.info("Processing image with CLIP...")
        inp = self.clip_proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inp)
            emb /= emb.norm(p=2, dim=-1, keepdim=True)
        logger.info("Image embedding generated successfully")
        logger.info("=== Image Embedding Generation Complete ===")
        return emb.cpu().numpy().astype('float32')
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        logger.info(f"=== Generating Text Embedding for: {text[:100]}... ===")
        inputs = self.clip_proc(text=[text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            txt_feats = self.clip_model.get_text_features(**inputs)
        arr = txt_feats.cpu().numpy().astype("float32")
        faiss.normalize_L2(arr)
        logger.info("Text embedding generated successfully")
        logger.info("=== Text Embedding Generation Complete ===")
        return arr
    
    def generate_caption(self, image_path: str):
        """Generate caption for an image using BLIP"""
        logger.info(f"=== Generating Caption for: {image_path} ===")
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info("Image loaded successfully")
            
            logger.info("Processing image with BLIP...")
            inputs = self.blip_proc(images=image, return_tensors="pt").to(DEVICE)
            out = self.blip_model.generate(**inputs, max_new_tokens=64)
            caption = self.blip_proc.decode(out[0], skip_special_tokens=True)
            logger.info(f"Generated caption: {caption}")
            logger.info("=== Caption Generation Complete ===")
            return caption
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return ""
    
    def generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        logger.info(f"=== Generating Response with Gemini for: {prompt[:100]}... ===")
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            result = response.text if hasattr(response, 'text') else str(response)
            logger.info(f"Generated response: {result[:100]}...")
            logger.info("=== Gemini Response Generation Complete ===")
            return result
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            return ""
    
    def ask_complements_local(self, caption, user_prompt, k=K):
        """Get complementary item recommendations using Gemini"""
        logger.info("=== Generating Complementary Recommendations ===")
        logger.info(f"Caption: {caption}")
        logger.info(f"User prompt: {user_prompt}")
        
        prompt = (
            f"You are a professional fashion stylist."
            f"\nYou are given a product: \"{caption}\"."
            f"\nCustomer said: \"{user_prompt}\"."
            f"\nList exactly 5 complementary and matching items for this."
            f"\nEach item MUST follow this format strictly:"
            f"\nCategory: <category>; subCategory: <subcategory> ;Article Type: <article_type>; Color/Style: <color_or_style>; Usage: <usage>"
            f"\nChoose <category> from ['Apparel','Accessories','Footwear','Personal Care','Free Items','Sporting Goods','Home']"
            f"\nChoose <subcategory> from ['Topwear','Shoes','Bags','Bottomwear','Watches','Innerwear','Jewellery','Eyewear','Fragrance','Sandal','Wallets','Flip Flops','Belts','Socks','Lips','Dress','Loungewear and Nightwear','Saree','Nails','Makeup','Headwear','Ties','Accessories','Scarves','Cufflinks','Apparel Set','Free Gifts','Stoles','Skin Care','Skin','Eyes','Mufflers','Shoe Accessories','Sports Equipment','Gloves','Hair','Bath and Body','Water Bottle','Perfumes','Umbrellas','Beauty Accessories','Wristbands','Sports Accessories','Home Furnishing','Vouchers']"
            f"\nSeparate each item with '//' on a single line."
            f"\nDO NOT include any explanations or extra text. Only output the 5 formatted items."
            f"\nDo not give same item as shown in the image in the recommendations."
        )
        
        logger.info("Generating complementary items with Gemini...")
        out = self.generate_with_gemini(prompt)
        items = [itm.strip() for itm in out.split('//') if itm.strip()][:k]
        numbered_items = [f"{i+1}. {itm}" for i, itm in enumerate(items)]
        logger.info("=== Complementary Recommendations Generation Complete ===")

        print(numbered_items)  # Debugging output
        logger.info(f"Generated {len(numbered_items)} complementary items: {numbered_items}")
        # items = []
        # for line in out.splitlines():
        #     line = line.strip()
        #     if re.match(r"^\d+\.\s", line):
        #         items.append(line.split(". ", 1)[1])
        #     if len(items) >= k:
        #         break
        # numbered_items = [f"{i+1}. {itm}" for i, itm in enumerate(items)]

        return numbered_items
    
    def recommend(self, img=None, prompt=None, k=K):
        """Complete recommendation function"""
        logger.info("=== Starting Recommendation Process ===")
        logger.info(f"Input parameters - Image: {'Present' if img else 'None'}, Prompt: {'Present' if prompt else 'None'}")
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
            
            # Get the category of the input image
            caption = self.generate_caption(img)
            input_category = None
            if caption:
                # Extract category from caption using Gemini
                category_prompt = f"Given this product description: '{caption}', what is the sub category of this product? Respond with just the category name.choose any one from:['Topwear','Shoes','Bags','Bottomwear','Watches','Innerwear','Jewellery','Eyewear','Fragrance','Sandal','Wallets','Flip Flops','Belts','Socks','Lips','Dress','Loungewear and Nightwear','Saree','Nails','Makeup','Headwear','Ties','Accessories','Scarves','Cufflinks','Apparel Set','Free Gifts','Stoles','Skin Care','Skin','Eyes','Mufflers','Shoe Accessories','Sports Equipment','Gloves','Hair','Bath and Body','Water Bottle','Perfumes','Umbrellas','Beauty Accessories','Wristbands','Sports Accessories','Home Furnishing','Vouchers']"
                input_category = self.generate_with_gemini(category_prompt).strip()
                logger.info(f"Detected input category: {input_category}")
            
            if has_txt:
                logger.info("Processing combined image and text input...")
                txt_emb = self.embed_text(prompt)
                txt_emb = txt_emb.astype('float32')
                if len(txt_emb.shape) == 1:
                    txt_emb = txt_emb.reshape(1, -1)
                faiss.normalize_L2(txt_emb)
                qv = np.concatenate([img_emb, txt_emb], axis=1).astype("float32")
                Dv, Iv = self.sim_index.search(qv, k*3)  # Increased search size for better filtering
            else:
                logger.info("Processing image-only input...")
                zero_txt = np.zeros((1, self.txt_embs.shape[1]), dtype=np.float32)
                qv = np.concatenate([img_emb, zero_txt], axis=1).astype("float32")
                faiss.normalize_L2(qv)
                Dv, Iv = self.sim_index.search(qv, k*3)  # Increased search size for better filtering
            
            # Filter results by category if available
            sim_df = self.df.iloc[Iv[0]][["productId", "text", "masterCategory", "subCategory"]].copy()
            sim_df["score_img"] = Dv[0]
            
            if input_category:
                # More flexible category matching
                category_matches = (
                    sim_df["masterCategory"].str.contains(input_category, case=False, na=False) |
                    sim_df["subCategory"].str.contains(input_category, case=False, na=False)
                )
                
                if category_matches.any():
                    sim_df = sim_df[category_matches]
                    logger.info(f"Filtered results by category: {input_category}")
                else:
                    logger.warning(f"No exact category matches found for {input_category}, using top similarity scores")
            
            # Ensure we have enough results
            if len(sim_df) < k:
                logger.warning(f"Not enough category-filtered results ({len(sim_df)}), using top similarity scores")
                sim_df = self.df.iloc[Iv[0]][["productId", "text", "masterCategory", "subCategory"]].copy()
                sim_df["score_img"] = Dv[0]
            
            sim_df = sim_df.head(k)  # Take top k after filtering
            
        elif has_txt:
            logger.info("Processing text-only input...")
            txt_emb = self.embed_text(prompt)
            txt_emb = txt_emb.astype('float32')
            if len(txt_emb.shape) == 1:
                txt_emb = txt_emb.reshape(1, -1)
            faiss.normalize_L2(txt_emb)
            Dt, It = self.txt_index.search(txt_emb, k)
            sim_df = self.df.iloc[It[0]][["productId", "text", "masterCategory", "subCategory"]].copy()
            sim_df["score_txt"] = Dt[0]
        
        logger.info(f"Similar product IDs: {sim_df['productId'].tolist() if not sim_df.empty else 'None'}")
        
        # Caption + Complementary Retrieval
        if has_txt or has_img:
            logger.info("Generating complementary recommendations...")
            caption = self.generate_caption(img) if has_img else ""
            if caption:
                logger.info(f"Generated caption: {caption}")
            
            # Get complementary categories
            cats = self.ask_complements_local(caption, prompt if has_txt else "")
            logger.info(f"Generated {len(cats)} complementary categories")
            
            # Extract categories from the complementary items
            complementary_categories = []
            for cat in cats:
                try:
                    category = cat.split("Category: ")[1].split(";")[0].strip()
                    complementary_categories.append(category)
                except:
                    continue
            
            logger.info(f"Complementary categories: {complementary_categories}")
            
            cand = []
            for cat in cats:
                q_t = self.embed_text(cat)
                q_t = q_t.astype('float32')
                if len(q_t.shape) == 1:
                    q_t = q_t.reshape(1, -1)
                faiss.normalize_L2(q_t)
                Dt, It = self.txt_index.search(q_t, k*3)  # Increased search size
                dfc = self.df.iloc[It[0]][["productId", "text", "masterCategory", "subCategory"]].copy()
                dfc["score_txt"] = Dt[0]
                
                # Filter by complementary categories
                if complementary_categories:
                    category_matches = (
                        dfc["masterCategory"].isin(complementary_categories) |
                        dfc["subCategory"].isin(complementary_categories)
                    )
                    if category_matches.any():
                        dfc = dfc[category_matches]
                        logger.info(f"Filtered complementary results by categories")
                    else:
                        logger.warning(f"No exact category matches found for complementary categories, using top similarity scores")
                
                cand.append(dfc)
            
            if cand:
                all_rec = pd.concat(cand, ignore_index=True)
                unique_rec = (
                    all_rec
                    .sort_values("score_txt", ascending=False)
                    .drop_duplicates(subset="productId", keep="first")
                )
                rec_df = unique_rec.head(k)
                logger.info(f"Found {len(rec_df)} recommended products")
        
        logger.info(f"Recommended product IDs: {rec_df['productId'].tolist() if not rec_df.empty else 'None'}")
        
        # Process recommendations and download images
        if not sim_df.empty or not rec_df.empty:
            logger.info("Processing final recommendations...")
            self.process_recommendations(sim_df, rec_df)
            logger.info("Recommendation process completed successfully")
        else:
            logger.warning("No recommendations generated")
        
        logger.info("=== Recommendation Process Complete ===")
        return sim_df, rec_df
    
    def get_product_details(self, product_ids):
        """Get detailed information for a list of product IDs"""
        logger.info(f"=== Getting Product Details for {len(product_ids)} Products ===")
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        
        results = []
        for pid in product_ids:
            logger.info(f"Looking up product ID: {pid}")
            mask = self.df['productId'] == str(pid)
            if mask.any():
                results.append(self.df[mask].iloc[0])
                logger.info(f"Found product: {pid}")
            else:
                logger.warning(f"Product not found: {pid}")
        
        logger.info(f"Retrieved details for {len(results)} products")
        logger.info("=== Product Details Retrieval Complete ===")
        return pd.DataFrame(results) if results else pd.DataFrame()

# Usage example
if __name__ == "__main__":
    logger.info("=== Starting Fashion Recommender in Standalone Mode ===")
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