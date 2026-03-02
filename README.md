# Handwritten OCR System

This project is a Handwritten OCR System that uses a CRNN (CNN + BiLSTM + CTC) architecture to recognize text from scanned handwritten documents.

# App Link 
https://pen2text.streamlit.app
## Features
- **Image Preprocessing**: Grayscale conversion, denoising, thresholding, and skew correction.
- **Deep Learning Model**: CRNN for sequence-to-sequence text recognition.
- **Web Interface**: A Streamlit app to upload images and view OCR results.
- **Evaluation**: Metrics like CER and WER included.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
