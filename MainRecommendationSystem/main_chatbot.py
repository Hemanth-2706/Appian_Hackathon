import sys
import os
import torch
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai
import requests
from io import BytesIO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit,
                            QFileDialog, QScrollArea, QFrame, QGridLayout, QTabWidget,
                            QMessageBox, QProgressBar, QSplitter, QGroupBox, QSpacerItem,
                            QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QImage

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 10

# Set up Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyAM_mhTB1qe4-7QgNy7ONjw9mSob7x5qdw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class FashionRecommender:
    def __init__(self, embeddings_dir="./", output_dir="./"):
        """Initialize the recommender by loading pre-computed embeddings"""
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.models_loaded = False
        
    def load_models(self):
        """Load all models needed for inference"""
        if self.models_loaded:
            return
            
        print("Loading models...")
        
        # CLIP for embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).half()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        # BLIP for captioning
        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE).half()
        
        self.models_loaded = True
        print("Models loaded successfully!")
    
    def load_embeddings(self):
        """Load pre-computed embeddings and metadata"""
        print("Loading embeddings and metadata...")
        
        # Load embeddings
        self.img_embs = np.load(os.path.join(self.embeddings_dir, r"F:\appian_round_2\img_embs.npy"))
        self.txt_embs = np.load(os.path.join(self.embeddings_dir, r"F:\appian_round_2\txt_embs.npy"))
        
        # Load metadata
        self.df = pd.read_csv(os.path.join(self.embeddings_dir, r"F:\appian_round_2\filtered_df.csv"), dtype=str)
        self.valid_indices = np.load(os.path.join(self.embeddings_dir, r"F:\appian_round_2\valid_indices.npy"))
        
        print(f"Loaded {len(self.df)} products with embeddings")
    
    def build_indexes(self):
        """Build FAISS indexes from loaded embeddings"""
        print("Building FAISS indexes...")
        
        # Fused index for multimodal search
        fused_embs = np.concatenate([self.img_embs, self.txt_embs], axis=1).astype("float32")
        faiss.normalize_L2(fused_embs)
        self.sim_index = faiss.IndexFlatIP(fused_embs.shape[1])
        self.sim_index.add(fused_embs)
        
        # Text-only index
        self.txt_index = faiss.IndexFlatIP(self.txt_embs.shape[1])
        self.txt_index.add(self.txt_embs)
        
        print("Indexes built successfully!")
    
    def embed_image(self, image_path):
        """Generate embedding for a new image"""
        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found: {image_path}")
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
            print(f"Error generating caption: {e}")
            return ""
    
    def generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API"""
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            print(f"Error with Gemini API: {e}")
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
        has_img = img is not None
        has_txt = prompt is not None
        
        if not has_img and not has_txt:
            return None, None
        
        sim_df = pd.DataFrame()
        rec_df = pd.DataFrame()
        
        # Visual or Text Similarity
        if has_img:
            img_emb = self.embed_image(img)
            if img_emb is None:
                return None, None
            
            img_emb = img_emb.astype('float32')
            if len(img_emb.shape) == 1:
                img_emb = img_emb.reshape(1, -1)
            faiss.normalize_L2(img_emb)
            
            if has_txt:
                txt_emb = self.embed_text(prompt)
                txt_emb = txt_emb.astype('float32')
                if len(txt_emb.shape) == 1:
                    txt_emb = txt_emb.reshape(1, -1)
                faiss.normalize_L2(txt_emb)
                qv = np.concatenate([img_emb, txt_emb], axis=1).astype("float32")
                Dv, Iv = self.sim_index.search(qv, k)
            else:
                zero_txt = np.zeros((1, self.txt_embs.shape[1]), dtype=np.float32)
                qv = np.concatenate([img_emb, zero_txt], axis=1).astype("float32")
                faiss.normalize_L2(qv)
                Dv, Iv = self.sim_index.search(qv, k)
            
            sim_df = self.df.iloc[Iv[0]][["id", "text"]].copy()
            sim_df["score_img"] = Dv[0]
            
        elif has_txt:
            txt_emb = self.embed_text(prompt)
            txt_emb = txt_emb.astype('float32')
            if len(txt_emb.shape) == 1:
                txt_emb = txt_emb.reshape(1, -1)
            faiss.normalize_L2(txt_emb)
            Dt, It = self.txt_index.search(txt_emb, k)
            
            sim_df = self.df.iloc[It[0]][["id", "text"]].copy()
            sim_df["score_txt"] = Dt[0]
        
        # Caption + Complementary Retrieval
        if has_txt or has_img:
            caption = self.generate_caption(img) if has_img else ""
            cats = self.ask_complements_local(caption, prompt if has_txt else "")
            
            cand = []
            for cat in cats:
                q_t = self.embed_text(cat)
                q_t = q_t.astype('float32')
                if len(q_t.shape) == 1:
                    q_t = q_t.reshape(1, -1)
                faiss.normalize_L2(q_t)
                
                Dt, It = self.txt_index.search(q_t, k)
                dfc = self.df.iloc[It[0]][["id", "text"]].copy()
                dfc["score_txt"] = Dt[0]
                cand.append(dfc)
            
            if cand:
                all_rec = pd.concat(cand, ignore_index=True)
                unique_rec = (
                    all_rec
                    .sort_values("score_txt", ascending=False)
                    .drop_duplicates(subset="id", keep="first")
                )
                rec_df = unique_rec.head(k)
        
        return sim_df, rec_df

class ImageLoader(QThread):
    """Thread for loading images from URLs"""
    image_loaded = pyqtSignal(str, object)  # product_id, pixmap
    error_occurred = pyqtSignal(str, str)  # product_id, error_message
    
    def __init__(self, product_id, image_url):
        super().__init__()
        self.product_id = product_id
        self.image_url = image_url
    
    def run(self):
        try:
            response = requests.get(self.image_url, timeout=10)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            
            # Convert PIL image to QPixmap
            img_data = img.tobytes("raw", "RGB")
            qimg = QImage(img_data, img.size[0], img.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            self.image_loaded.emit(self.product_id, pixmap)
            
        except Exception as e:
            self.error_occurred.emit(self.product_id, str(e))

class RecommendationThread(QThread):
    """Thread for running recommendations without blocking UI"""
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)
    
    def __init__(self, recommender, img_path, text_prompt):
        super().__init__()
        self.recommender = recommender
        self.img_path = img_path
        self.text_prompt = text_prompt
    
    def run(self):
        try:
            sim_results, comp_results = self.recommender.recommend(
                img=self.img_path if self.img_path else None,
                prompt=self.text_prompt if self.text_prompt else None,
                k=10
            )
            self.finished.emit(sim_results, comp_results)
        except Exception as e:
            self.error.emit(str(e))

class WelcomeWidget(QWidget):
    """Welcome screen with Gemini-powered introduction"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.load_welcome_message()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Welcome to AI Fashion Stylist")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin: 20px;")
        
        # Welcome message area
        self.welcome_text = QTextEdit()
        self.welcome_text.setReadOnly(True)
        self.welcome_text.setMaximumHeight(200)
        self.welcome_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
                line-height: 1.5;
            }
        """)
        
        # Get Started button
        self.get_started_btn = QPushButton("Get Started with Fashion Recommendations")
        self.get_started_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.get_started_btn.clicked.connect(self.start_recommendations)
        
        layout.addWidget(title)
        layout.addWidget(self.welcome_text)
        layout.addWidget(self.get_started_btn, 0, Qt.AlignCenter)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def load_welcome_message(self):
        """Load welcome message using Gemini"""
        try:
            prompt = """
            You are the AI assistant for a cutting-edge Fashion Recommendation Store. Write a warm, engaging welcome message (2-3 paragraphs) that explains:
            1. What our AI Fashion Stylist does
            2. How customers can upload images or describe what they're looking for
            3. How we provide both similar items and complementary fashion recommendations
            4. Make it sound exciting and personalized
            
            Keep it friendly, professional, and under 150 words.
            """
            
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            welcome_msg = response.text if hasattr(response, 'text') else "Welcome to AI Fashion Stylist!"
            
            self.welcome_text.setPlainText(welcome_msg)
            
        except Exception as e:
            default_msg = """
            Welcome to AI Fashion Stylist - Your Personal AI Shopping Assistant!
            
            Our advanced AI technology combines computer vision and natural language processing to provide you with personalized fashion recommendations. Simply upload an image of a clothing item you like, or describe what you're looking for in words.
            
            We'll help you find similar items and suggest complementary pieces that perfectly match your style. Whether you're building a complete outfit or looking for that perfect accessory, our AI stylist is here to guide you through your fashion journey.
            
            Ready to discover your next favorite fashion find?
            """
            self.welcome_text.setPlainText(default_msg)
    
    def start_recommendations(self):
        """Switch to recommendation interface"""
        if self.parent:
            self.parent.show_recommendation_interface()

class FashionRecommenderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recommender = None
        self.current_image_path = None
        self.images_df = None  # For storing image URLs
        self.image_loaders = []  # Keep track of image loading threads
        self.init_ui()
        self.setup_recommender()
        self.load_images_csv()
    
    def init_ui(self):
        self.setWindowTitle("AI Fashion Stylist")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
        """)
        
        # Create central widget with tab system
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        
        # Welcome tab
        self.welcome_widget = WelcomeWidget(self)
        self.tab_widget.addTab(self.welcome_widget, "Welcome")
        
        # Recommendation tab
        self.recommendation_widget = self.create_recommendation_widget()
        self.tab_widget.addTab(self.recommendation_widget, "Fashion Recommendations")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_recommendation_widget(self):
        """Create the main recommendation interface"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Left panel - Input controls
        left_panel = QGroupBox("Search & Upload")
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Image upload section
        img_group = QGroupBox("Upload Image")
        img_layout = QVBoxLayout(img_group)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(200)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #007bff;
                border-radius: 8px;
                background-color: #f8f9fa;
                color: #6c757d;
            }
        """)
        
        self.upload_btn = QPushButton("Choose Image")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_image)
        
        img_layout.addWidget(self.image_label)
        img_layout.addWidget(self.upload_btn)
        
        # Text input section
        text_group = QGroupBox("Describe What You're Looking For")
        text_layout = QVBoxLayout(text_group)
        
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(100)
        self.text_input.setPlaceholderText("E.g., 'blue casual shirt for summer' or 'elegant evening dress'")
        
        text_layout.addWidget(self.text_input)
        
        # Search button
        self.search_btn = QPushButton("Find Recommendations")
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        self.search_btn.clicked.connect(self.search_recommendations)
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a71d2a;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_all)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        left_layout.addWidget(img_group)
        left_layout.addWidget(text_group)
        left_layout.addWidget(self.search_btn)
        left_layout.addWidget(self.clear_btn)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch()

        # --- Chatbot area ---
        chat_group = QGroupBox("Chat with your AI Stylist")
        chat_layout = QVBoxLayout(chat_group)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(120)
        self.chat_history.setStyleSheet("background-color: #f8f9fa; border-radius: 6px; padding: 8px;")
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message here...")
        self.chat_send_btn = QPushButton("Send")
        self.chat_send_btn.setStyleSheet("background-color: #007bff; color: white; border-radius: 4px; padding: 6px 16px;")
        self.chat_send_btn.clicked.connect(self.handle_chat_send)
        chat_input_layout = QHBoxLayout()
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.chat_send_btn)
        chat_layout.addWidget(self.chat_history)
        chat_layout.addLayout(chat_input_layout)
        left_layout.addWidget(chat_group)
        # --- End chatbot area ---
        
        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Similar items tab
        self.similar_scroll = QScrollArea()
        self.similar_widget = QWidget()
        self.similar_layout = QGridLayout(self.similar_widget)
        self.similar_scroll.setWidget(self.similar_widget)
        self.similar_scroll.setWidgetResizable(True)
        self.results_tabs.addTab(self.similar_scroll, "Similar Items")
        
        # Complementary items tab
        self.complement_scroll = QScrollArea()
        self.complement_widget = QWidget()
        self.complement_layout = QGridLayout(self.complement_widget)
        self.complement_scroll.setWidget(self.complement_widget)
        self.complement_scroll.setWidgetResizable(True)
        self.results_tabs.addTab(self.complement_scroll, "Complementary Items")
        
        right_layout.addWidget(self.results_tabs)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        return widget
    
    def setup_recommender(self):
        """Initialize the fashion recommender"""
        try:
            self.recommender = FashionRecommender(
                embeddings_dir="./",
                output_dir="./"
            )
            # Note: Models will be loaded when first needed to avoid blocking UI
        except Exception as e:
            QMessageBox.warning(self, "Initialization Error", f"Could not initialize recommender: {e}")
    
    def load_images_csv(self):
        """Load the images.csv file for displaying product images"""
        try:
            self.images_df = pd.read_csv(r"F:\appian_round_2\fashion-dataset\fashion-dataset\images.csv")
            print(f"Loaded {len(self.images_df)} image records")
        except FileNotFoundError:
            print("Warning: images.csv not found. Product images will not be displayed.")
            QMessageBox.warning(self, "Images File Missing", 
                              "images.csv not found. Product images will not be displayed.")
        except Exception as e:
            print(f"Error loading images.csv: {e}")
            QMessageBox.warning(self, "Images Loading Error", f"Error loading images.csv: {e}")
    
    def get_image_url_by_id(self, product_id):
        """Get image URL for a product ID"""
        if self.images_df is None:
            return None
        
        filename = f"{product_id}.jpg"
        row = self.images_df[self.images_df['filename'] == filename]
        
        if row.empty:
            return None
        
        return row['link'].values[0]
    
    def show_recommendation_interface(self):
        """Switch to the recommendation tab"""
        self.tab_widget.setCurrentIndex(1)
    
    def upload_image(self):
        """Handle image upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Display image preview
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")
    
    def search_recommendations(self):
        """Perform recommendation search"""
        if not self.current_image_path and not self.text_input.toPlainText().strip():
            QMessageBox.warning(self, "Input Required", "Please upload an image or enter a text description.")
            return
        
        if not self.recommender:
            QMessageBox.warning(self, "System Error", "Recommender system not initialized.")
            return
        
        # Load models if not already loaded
        if not self.recommender.models_loaded:
            try:
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate progress
                self.statusBar().showMessage("Loading AI models...")
                
                self.recommender.load_models()
                self.recommender.load_embeddings()
                self.recommender.build_indexes()
                
                self.progress_bar.setVisible(False)
                self.statusBar().showMessage("Models loaded successfully")
            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Loading Error", f"Failed to load models: {e}")
                return
        
        # Start recommendation thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.statusBar().showMessage("Searching for recommendations...")
        self.search_btn.setEnabled(False)
        
        self.recommendation_thread = RecommendationThread(
            self.recommender,
            self.current_image_path,
            self.text_input.toPlainText().strip()
        )
        self.recommendation_thread.finished.connect(self.display_results)
        self.recommendation_thread.error.connect(self.handle_error)
        self.recommendation_thread.start()
    
    def display_results(self, similar_df, complement_df):
        """Display recommendation results"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.statusBar().showMessage("Recommendations ready")
        
        # Clear previous results
        self.clear_results()
        
        # Display similar items
        if similar_df is not None and not similar_df.empty:
            for i, row in enumerate(similar_df.iterrows()):
                idx, data = row
                self.add_result_item(self.similar_layout, i, data['id'], data['text'])
        
        # Display complementary items
        if complement_df is not None and not complement_df.empty:
            for i, row in enumerate(complement_df.iterrows()):
                idx, data = row
                self.add_result_item(self.complement_layout, i, data['id'], data['text'])
    
    def add_result_item(self, layout, index, product_id, description):
        """Add a result item to the layout, including product image"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setStyleSheet("""
            QFrame {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
                background-color: #f8f9fa;
            }
        """)
        item_layout = QVBoxLayout(frame)

        # Product image
        img_label = QLabel()
        img_label.setFixedSize(120, 120)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("background-color: #e9ecef; border-radius: 6px;")
        # Set placeholder pixmap
        placeholder = QPixmap(120, 120)
        placeholder.fill(QColor('#e9ecef'))
        img_label.setPixmap(placeholder)

        # Start image loading thread
        image_url = self.get_image_url_by_id(product_id)
        if image_url:
            loader = ImageLoader(str(product_id), image_url)
            loader.image_loaded.connect(lambda pid, pixmap, lbl=img_label: lbl.setPixmap(pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)) if pid == str(product_id) else None)
            loader.error_occurred.connect(lambda pid, err: img_label.setToolTip(f"Image load error: {err}") if pid == str(product_id) else None)
            loader.start()
            self.image_loaders.append(loader)  # Keep reference
        else:
            img_label.setToolTip("No image available")

        # Product ID
        id_label = QLabel(f"ID: {product_id}")
        id_label.setFont(QFont("Arial", 10, QFont.Bold))
        id_label.setStyleSheet("color: #007bff;")
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setMaximumWidth(250)
        desc_label.setStyleSheet("color: #495057; margin-top: 5px;")

        item_layout.addWidget(img_label)
        item_layout.addWidget(id_label)
        item_layout.addWidget(desc_label)

        row = index // 3
        col = index % 3
        layout.addWidget(frame, row, col)
    
    def clear_results(self):
        """Clear previous results"""
        # Clear similar items
        for i in reversed(range(self.similar_layout.count())):
            self.similar_layout.itemAt(i).widget().setParent(None)
        
        # Clear complementary items
        for i in reversed(range(self.complement_layout.count())):
            self.complement_layout.itemAt(i).widget().setParent(None)
    
    def handle_error(self, error_msg):
        """Handle errors from recommendation thread"""
        self.progress_bar.setVisible(False)
        self.search_btn.setEnabled(True)
        self.statusBar().showMessage("Error occurred")
        QMessageBox.critical(self, "Recommendation Error", f"An error occurred: {error_msg}")

    # --- Chatbot logic and state ---
    def handle_chat_send(self):
        user_msg = self.chat_input.text().strip()
        if not user_msg:
            return
        self.append_chat("You", user_msg)
        self.chat_input.clear()
        self.process_chat_message(user_msg)

    def append_chat(self, sender, message):
        self.chat_history.append(f"<b>{sender}:</b> {message}")

    def process_chat_message(self, user_msg):
        # Simple stateful conversation logic (to be expanded with Gemini)
        if not hasattr(self, 'chat_state'):
            self.chat_state = 'greeting'
        if self.chat_state == 'greeting':
            self.append_chat("AI Stylist", "Hello! How can I help you with your fashion needs today?")
            self.chat_state = 'waiting_for_input'
        elif self.chat_state == 'waiting_for_input':
            # Here, trigger recommendation search based on user_msg
            self.text_input.setText(user_msg)
            self.search_recommendations()
            self.chat_state = 'showing_recommendations'
        elif self.chat_state == 'showing_recommendations':
            # After showing recommendations, ask for feedback
            self.append_chat("AI Stylist", "Are you happy with my suggestion?")
            self.chat_state = 'awaiting_feedback'
        elif self.chat_state == 'awaiting_feedback':
            if any(word in user_msg.lower() for word in ['yes', 'happy', 'good', 'great', 'love']):
                self.append_chat("AI Stylist", "Thank you! Would you like to search for something else?")
                self.chat_state = 'end_or_restart'
            else:
                self.append_chat("AI Stylist", "I'm sorry! Let me try to find better recommendations for you.")
                # Optionally, refine search (e.g., increase k or change prompt)
                self.search_recommendations()
                self.chat_state = 'showing_recommendations'
        elif self.chat_state == 'end_or_restart':
            if any(word in user_msg.lower() for word in ['yes', 'again', 'search', 'more']):
                self.append_chat("AI Stylist", "Sure! Please tell me what you're looking for.")
                self.chat_state = 'waiting_for_input'
            else:
                self.append_chat("AI Stylist", "Thank you for using AI Fashion Stylist! Have a great day!")
                self.chat_state = 'greeting'

    def clear_all(self):
        """Clear uploaded image, text input, and all displayed results"""
        self.current_image_path = None
        self.image_label.clear()
        self.image_label.setText("No image selected")
        self.text_input.clear()
        self.clear_results()
        self.statusBar().showMessage("Cleared. Ready for new input.")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Fashion Stylist")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = FashionRecommenderGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()