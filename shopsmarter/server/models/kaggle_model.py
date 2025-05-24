# run_kaggle_pipeline.py
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

api = KaggleApi()
api.authenticate()

# Zip session data
with zipfile.ZipFile("upload.zip", "w") as zipf:
    zipf.write("session_data/input.txt")
    zipf.write("session_data/image.jpg")

# Push to dataset or notebook, then download output
api.kernels_output("username/notebook-name", path="output/")
