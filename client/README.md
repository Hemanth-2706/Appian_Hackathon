# AI Fashion Stylist

An intelligent fashion recommendation system that combines computer vision, natural language processing, and AI to provide personalized fashion recommendations.

## Overview

The AI Fashion Stylist is a desktop application that helps users find fashion items based on either images or text descriptions. It uses multiple AI models to provide both similar items and complementary fashion recommendations.

## Key Features

1. **Multimodal Search**
   - Image-based search: Upload images of clothing items
   - Text-based search: Describe what you're looking for
   - Combined search: Use both image and text for more precise results

2. **AI-Powered Recommendations**
   - Similar items: Find items visually or semantically similar to your input
   - Complementary items: Get suggestions for items that go well with your selection
   - AI chat interface: Interact with an AI stylist for personalized assistance

3. **User-Friendly Interface**
   - Modern GUI built with PyQt5
   - Real-time image preview
   - Interactive chat system
   - Clear results display with product images

## Technical Architecture

### Core Components

1. **FashionRecommender Class**
   - Handles all AI model operations
   - Manages embeddings and similarity search
   - Generates recommendations

2. **GUI Components**
   - WelcomeWidget: Initial welcome screen
   - FashionRecommenderGUI: Main application window
   - ImageLoader: Asynchronous image loading
   - RecommendationThread: Background processing

### AI Models Used

1. **CLIP (Contrastive Language-Image Pre-training)**
   - Purpose: Generate embeddings for both images and text
   - Used for: Similarity search and matching

2. **BLIP (Bootstrapping Language-Image Pre-training)**
   - Purpose: Generate image captions
   - Used for: Understanding image content

3. **Gemini AI**
   - Purpose: Natural language understanding and generation
   - Used for: Chat interface and complementary item suggestions

## Data Flow

1. **Input Processing**
   - User uploads image or enters text
   - System processes input through appropriate AI models
   - Generates embeddings and captions

2. **Recommendation Generation**
   - Similar items: Uses FAISS for efficient similarity search
   - Complementary items: Uses Gemini AI for style matching
   - Results are ranked and filtered

3. **Result Display**
   - Asynchronous image loading
   - Grid-based result presentation
   - Interactive chat feedback

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Required Python packages:
  ```
  torch
  transformers
  faiss-cpu
  pandas
  Pillow
  PyQt5
  google-generativeai
  ```

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Google API key for Gemini:
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   ```

### Data Requirements

- Pre-computed embeddings (img_embs.npy, txt_embs.npy)
- Product metadata (filtered_df.csv)
- Image dataset (images.csv)

## Usage

1. Launch the application:
   ```bash
   python main_chatbot.py
   ```

2. Use the interface to:
   - Upload images
   - Enter text descriptions
   - Chat with the AI stylist
   - View and interact with recommendations

## Performance Considerations

- Initial model loading may take time
- Image processing is done asynchronously
- Results are cached for better performance
- GPU acceleration is used when available

## Future Improvements

1. Enhanced recommendation algorithms
2. More sophisticated chat interactions
3. Additional fashion categories
4. Improved image processing
5. Better error handling and recovery

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 