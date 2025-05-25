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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')
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
logger.info("Initializing FashionRecommender...")
recommender = FashionRecommender()
logger.info("FashionRecommender initialized successfully")

class RecommendationRequest(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None

class ProductDetailsRequest(BaseModel):
    product_ids: List[str]

def convert_dataframe_to_dict(df):
    """Convert DataFrame to a list of dictionaries with proper column names"""
    logger.info("Starting DataFrame conversion")
    if df is None or df.empty:
        logger.warning("Empty or None DataFrame received")
        return []
    
    try:
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
        elif 'id' in result_df.columns:
            final_df['productId'] = result_df['id']
        
        if 'productName' in result_df.columns:
            final_df['productName'] = result_df['productName']
        elif 'text' in result_df.columns:
            final_df['productName'] = result_df['text']
        
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
        return result
    except Exception as e:
        logger.error(f"Error converting DataFrame: {str(e)}", exc_info=True)
        return []

@app.post("/process-recommendations")
async def process_recommendations(request: RecommendationRequest):
    logger.info("Received recommendation request")
    logger.info(f"Request data - Text: {request.text}, Image: {'Present' if request.image else 'None'}")
    
    try:
        # Get recommendations using the model
        logger.info("Calling recommender.recommend()")
        sim_results, comp_results = recommender.recommend(
            img=request.image,
            prompt=request.text,
            k=10
        )
        logger.info("Received results from recommender")

        # Convert results to JSON-serializable format
        logger.info("Converting similar products to dictionary")
        similar_products = convert_dataframe_to_dict(sim_results)
        logger.info("Converting recommended products to dictionary")
        recommend_products = convert_dataframe_to_dict(comp_results)

        logger.info(f"Processed {len(similar_products)} similar products and {len(recommend_products)} recommended products")

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
    logger.info(f"Received product details request for IDs: {request.product_ids}")
    try:
        # Get product details using the model
        logger.info("Calling recommender.get_product_details()")
        details = recommender.get_product_details(request.product_ids)
        logger.info(f"Retrieved details for {len(details)} products")
        
        # Convert to JSON-serializable format
        logger.info("Converting product details to dictionary")
        products = convert_dataframe_to_dict(details)
        
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
    logger.info("Health check requested")
    return JSONResponse(
        content={"status": "healthy", "model_loaded": True},
        media_type="application/json"
    )

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=5001) 