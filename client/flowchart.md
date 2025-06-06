```mermaid
graph TD
    A[Start Application] --> B[Load Welcome Screen]
    B --> C[Initialize FashionRecommender]
    C --> D[Load Models & Data]
    
    subgraph User Interface
        E[Upload Image] --> F[Process Input]
        G[Enter Text] --> F
        H[Chat Interface] --> F
    end
    
    subgraph AI Processing
        F --> I[Generate Embeddings]
        I --> J[Similar Items Search]
        I --> K[Complementary Items Search]
        
        J --> L[FAISS Index Search]
        K --> M[Gemini AI Processing]
        
        L --> N[Rank Results]
        M --> N
    end
    
    subgraph Result Display
        N --> O[Load Product Images]
        O --> P[Display Results]
        P --> Q[Update Chat Interface]
    end
    
    D --> E
    D --> G
    D --> H
    
    Q --> E
    Q --> G
    Q --> H

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style N fill:#bfb,stroke:#333,stroke-width:2px
    style Q fill:#fbb,stroke:#333,stroke-width:2px
```

## Process Flow Explanation

1. **Application Initialization**
   - Application starts and loads the welcome screen
   - FashionRecommender is initialized
   - Required AI models and data are loaded

2. **User Input Processing**
   - Users can interact through three main channels:
     * Image upload
     * Text input
     * Chat interface
   - All inputs are processed through the AI pipeline

3. **AI Processing Pipeline**
   - Input is converted to embeddings using CLIP
   - Two parallel processes:
     * Similar items search using FAISS
     * Complementary items using Gemini AI
   - Results are ranked and combined

4. **Result Display**
   - Product images are loaded asynchronously
   - Results are displayed in a grid layout
   - Chat interface is updated with recommendations
   - User can continue interaction

5. **Feedback Loop**
   - User can provide feedback through chat
   - System can refine recommendations
   - Process can be repeated with new inputs 