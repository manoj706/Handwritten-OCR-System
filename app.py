import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os
import easyocr

# Import project modules
from src.preprocess import preprocess_image
from src.segmentation import segment_lines

# Page Config
st.set_page_config(page_title="Handwritten OCR System", layout="wide")

# Title and sidebar
st.title("Handwritten Text Recognition System ✍️")
st.markdown("---")

st.sidebar.header("OCR Engine Selection")
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["Gemini Vision API (Best)", "TrOCR (High-Accuracy Local)", "EasyOCR (Lightweight)"],
    index=0
)

# Model specific settings
gemini_api_key = ""
if model_choice == "Gemini Vision API (Best)":
    st.sidebar.markdown("### Gemini Settings")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get a free key at aistudio.google.com/app/apikey")
    st.sidebar.info("Gemini offers 'ChatGPT-level' accuracy by using Google's latest Vision-Language models.")

if model_choice == "TrOCR (High-Accuracy Local)":
    st.sidebar.markdown("### TrOCR Settings")
    st.sidebar.info("TrOCR is a local transformer specifically trained for handwriting.")
    trocr_size = st.sidebar.selectbox("Model Size", ["Base", "Large"], index=0, help="Large is more accurate but takes more RAM and time to download (~1.4GB).")
    trocr_segmentation = st.sidebar.checkbox("Line-by-Line Processing", value=True, help="Recommended for multi-line notes.")
    trocr_model_id = "microsoft/trocr-large-handwritten" if trocr_size == "Large" else "microsoft/trocr-base-handwritten"

# Preprocessing Settings
st.sidebar.markdown("---")
st.sidebar.header("Image Preprocessing")
denoise_strength = st.sidebar.slider("Denoise Strength", 0, 5, 1)
use_preprocessed = st.sidebar.checkbox("Apply Preprocessing before OCR", value=True)

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Preprocessing
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    processed_gray = preprocess_image(image_bgr, denoise_strength=denoise_strength)
    
    with col2:
        st.subheader("Binarized Image")
        st.image(processed_gray, use_container_width=True, channels="GRAY")
        
    if st.button("🚀 Perform OCR"):
        st.info(f"Running OCR with {model_choice}...")
        full_text = ""
        
        # Decide which image to feed to OCR
        ocr_input = processed_gray if use_preprocessed else image_np
        
        # --- 1. Gemini Vision Path ---
        if model_choice == "Gemini Vision API (Best)":
            if not gemini_api_key:
                st.error("Please enter your Gemini API Key in the sidebar.")
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key)
                    # Use gemini-1.5-flash for speed and cost effectiveness
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Convert to PIL for Gemini tool
                    # If using preprocessed, it's grayscale, Gemini prefers RGB
                    if use_preprocessed:
                        pil_input = Image.fromarray(cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB))
                    else:
                        pil_input = image
                        
                    with st.spinner("Gemini is reading..."):
                        response = model.generate_content([
                            pil_input,
                            "Transcribe all handwritten text in this image verbatim. "
                            "Preserve paragraph structure and line breaks. "
                            "Output ONLY the transcribed text, nothing else."
                        ])
                        full_text = response.text
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")

        # --- 2. TrOCR Path ---
        elif model_choice == "TrOCR (High-Accuracy Local)":
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                from PIL import Image as PILImage
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                @st.cache_resource
                def load_trocr(model_id):
                    print(f"DEBUG: Loading TrOCR Processor and Model ({model_id})...")
                    processor = TrOCRProcessor.from_pretrained(model_id)
                    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
                    return processor, model

                with st.spinner(f"Loading TrOCR {trocr_size} Model (this may take a minute)..."):
                    processor, model = load_trocr(trocr_model_id)
                
                def process_trocr_image(img):
                    # Convert to PIL RGB if needed
                    if len(img.shape) == 2: # Gray
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    pil_img = PILImage.fromarray(img)
                    
                    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
                    generated_ids = model.generate(pixel_values)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    return generated_text

                if trocr_segmentation:
                    with st.spinner("Segmenting lines and recognizing..."):
                        lines = segment_lines(ocr_input)
                        if not lines:
                            st.warning("No text lines detected. Attempting full image recognition.")
                            lines = [ocr_input]
                        
                        progress_bar = st.progress(0)
                        temp_text = []
                        for i, line_img in enumerate(lines):
                            line_out = process_trocr_image(line_img)
                            temp_text.append(line_out)
                            progress_bar.progress((i + 1) / len(lines))
                        
                        full_text = "\n".join(temp_text)
                else:
                    with st.spinner("TrOCR is reading full image..."):
                        full_text = process_trocr_image(ocr_input)
            except Exception as e:
                st.error(f"TrOCR Error: {str(e)}")
                print(f"DEBUG: TrOCR Error detail: {e}")

        # --- 3. EasyOCR Path ---
        elif model_choice == "EasyOCR (Lightweight)":
            try:
                with st.spinner("Initializing EasyOCR..."):
                    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                
                with st.spinner("EasyOCR is reading..."):
                    # Use original paragraph groupings
                    results = reader.readtext(
                        ocr_input,
                        paragraph=True,
                        contrast_ths=0.1,
                        adjust_contrast=0.7,
                        width_ths=1.2
                    )
                    # Sort top-to-bottom
                    results = sorted(results, key=lambda r: r[0][0][1])
                    full_text = "\n".join([r[1] for r in results])
            except Exception as e:
                st.error(f"EasyOCR Error: {str(e)}")

        # --- Results Display ---
        if full_text:
            st.success("Recognition Complete!")
            st.text_area("Final Transcription", full_text, height=350)
            st.download_button("Download Transcription", full_text, file_name="handwriting_transcription.txt")
        else:
            st.warning("OCR finished but no text was generated. Please try a different model or adjust settings.")

else:
    st.info("👋 Welcome! Please upload a JPG or PNG image of handwriting from the sidebar to start.")
