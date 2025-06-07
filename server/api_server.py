from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
import os
import uvicorn
import json
import pandas as pd
import logging
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server/logs/api_server.log'),
        logging.FileHandler('server/logs/all_logs.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the server directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.model import FashionRecommender

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # Your Node.js server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the recommender
logger.info("=== Starting API Server Initialization ===")
logger.info("Initializing FashionRecommender...")
recommender = FashionRecommender()
logger.info("FashionRecommender initialized successfully")
logger.info("=== API Server Initialization Complete ===")

class RecommendationRequest(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None  # This will be a base64 encoded image

class ProductDetailsRequest(BaseModel):
    product_ids: List[str]

def convert_dataframe_to_dict(df):
    """Convert DataFrame to a list of dictionaries with proper column names"""
    logger.info("=== Starting DataFrame Conversion ===")
    if df is None or df.empty:
        logger.warning("Empty or None DataFrame received")
        return []
    
    try:
        logger.info(f"Original DataFrame shape: {df.shape}")
        logger.info(f"Original DataFrame columns: {df.columns.tolist()}")
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Map the columns to the expected format
        column_mapping = {
            'id': 'productId',
            'text': 'productName',
            'articleType': 'articleType',
            'subCategory': 'subCategory',
            'season': 'season',
            'usage': 'usage',
            'price': 'price'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in result_df.columns:
                logger.info(f"Renaming column {old_col} to {new_col}")
                result_df = result_df.rename(columns={old_col: new_col})
        
        # Ensure all required columns exist
        required_columns = ['productId', 'productName', 'articleType', 'subCategory', 'season', 'usage', 'image', 'price']
        logger.info(f"Required columns: {required_columns}")
        
        # Create a new DataFrame with default values
        final_df = pd.DataFrame()
        
        # Add productId and productName from the original DataFrame
        if 'productId' in result_df.columns:
            final_df['productId'] = result_df['productId']
            logger.info("Using existing productId column")
        elif 'id' in result_df.columns:
            final_df['productId'] = result_df['id']
            logger.info("Using 'id' column as productId")
        
        if 'productName' in result_df.columns:
            final_df['productName'] = result_df['productName']
            logger.info("Using existing productName column")
        elif 'text' in result_df.columns:
            final_df['productName'] = result_df['text']
            logger.info("Using 'text' column as productName")
        
        # Add other required columns with default values
        for col in required_columns:
            if col not in final_df.columns:
                logger.info(f"Adding missing column: {col}")
                if col == 'image':
                    final_df[col] = f"/images/products/{final_df['productId']}.jpg"
                elif col == 'price':
                    final_df[col] = 0.0
                else:
                    final_df[col] = ""
        
        # Convert to dictionary format
        result = final_df[required_columns].to_dict('records')
        logger.info(f"Successfully converted DataFrame to dictionary with {len(result)} records")
        logger.info("=== DataFrame Conversion Complete ===")
        return result
    except Exception as e:
        logger.error(f"Error converting DataFrame: {str(e)}", exc_info=True)
        return []

def save_base64_image(base64_string: str) -> str:
    """Save base64 image to a temporary file and return the path"""
    logger.info("=== Starting Base64 Image Processing ===")
    try:
        # Remove data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]
            logger.info("Removed data URL prefix from base64 string")
        logger.info(f"First 100 chars of base64 image: {base64_string[:100]}")
        
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Ensured temp directory exists at: {temp_dir}")
        
        # Generate unique filename
        temp_path = os.path.join(temp_dir, f"temp_image_{hash(base64_string)}.jpg")
        logger.info(f"Generated temp file path: {temp_path}")
        
        # Decode and save image
        image_data = base64.b64decode(base64_string)
        with open(temp_path, "wb") as f:
            f.write(image_data)
        logger.info(f"Successfully saved image to: {temp_path}")
        
        logger.info("=== Base64 Image Processing Complete ===")
        return temp_path
    except Exception as e:
        logger.error(f"Error saving base64 image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process image")

@app.post("/process-recommendations")
async def process_recommendations(request: RecommendationRequest):
    logger.info("=== Starting Recommendation Processing ===")
    logger.info(f"Request received - Text present: {bool(request.text)}, Image present: {bool(request.image)}")
    
    try:
        # Process image if present
        image_path = None
        if request.image:
            logger.info("Processing image from request...")
            image_path = save_base64_image(request.image)
            logger.info(f"Image processed and saved to: {image_path}")
        
        # Get recommendations using the model
        logger.info("Calling recommender.recommend() with parameters:")
        logger.info(f"- Image path: {image_path}")
        logger.info(f"- Text prompt: {request.text}")
        logger.info(f"- K value: 10")
        
        sim_results, comp_results = recommender.recommend(
            img=image_path,
            prompt=request.text,
            k=10
        )
        logger.info("Received results from recommender")

        # Convert results to JSON-serializable format
        logger.info("Converting similar products to dictionary format...")
        similar_products = convert_dataframe_to_dict(sim_results)
        logger.info(f"Converted {len(similar_products)} similar products")
        
        logger.info("Converting recommended products to dictionary format...")
        recommend_products = convert_dataframe_to_dict(comp_results)
        logger.info(f"Converted {len(recommend_products)} recommended products")

        logger.info(f"Total processed products - Similar: {len(similar_products)}, Recommended: {len(recommend_products)}")

        # Clean up temporary image if it exists
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            logger.info(f"Cleaned up temporary image: {image_path}")

        logger.info("=== Recommendation Processing Complete ===")
        return JSONResponse(
            content={
                "success": True,
                "similarProducts": similar_products,
                "recommendProducts": recommend_products
            },
            media_type="application/json"
        )

    except Exception as e:
        logger.error(f"Error in process_recommendations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-product-details")
async def get_product_details(request: ProductDetailsRequest):
    logger.info("=== Starting Product Details Retrieval ===")
    logger.info(f"Request received for {len(request.product_ids)} product IDs: {request.product_ids}")
    try:
        # Get product details using the model
        logger.info("Calling recommender.get_product_details()...")
        details = recommender.get_product_details(request.product_ids)
        logger.info(f"Retrieved details for {len(details)} products")
        
        # Convert to JSON-serializable format
        logger.info("Converting product details to dictionary format...")
        products = convert_dataframe_to_dict(details)
        logger.info(f"Converted {len(products)} products to dictionary format")
        
        logger.info("=== Product Details Retrieval Complete ===")
        return JSONResponse(
            content={
                "success": True,
                "products": products
            },
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Error in get_product_details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    logger.info("=== Health Check Requested ===")
    logger.info("Checking system health...")
    return JSONResponse(
        content={"status": "healthy", "model_loaded": True},
        media_type="application/json"
    )

if __name__ == "__main__":
    logger.info("=== Starting FastAPI Server ===")
    logger.info("Server will be available at http://0.0.0.0:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001) 