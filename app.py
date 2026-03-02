import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os

# Set Hugging Face cache to D: drive to avoid "No space left on device" error on C:
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface_cache"
# Optimize CUDA memory allocation to prevent fragmentation OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import project modules
from src.preprocess import preprocess_image
from src.segmentation import segment_lines

# Guarded imports for heavy models
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:
    pass

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
trocr_size = "Base"
trocr_segmentation = True
trocr_model_id = "microsoft/trocr-base-handwritten"

if model_choice == "Gemini Vision API (Best)":
    st.sidebar.markdown("### Gemini Settings")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Get a free key at aistudio.google.com/app/apikey")
    st.sidebar.info("Gemini offers 'ChatGPT-level' accuracy by using Google's latest Vision-Language models.")

if model_choice == "TrOCR (High-Accuracy Local)":
    st.sidebar.markdown("### TrOCR Settings")
    
    import psutil
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    if available_ram_gb < 3.0:
        st.sidebar.warning(f"⚠️ **Low RAM ({available_ram_gb:.1f}GB)**: 'Large' model is disabled to prevent crashes. Using 'Base'.")
        trocr_size = "Base"
    else:
        st.sidebar.warning("⚠️ **High Memory Usage**: Local TrOCR requires ~2GB+ free RAM. If the app crashes, use Gemini (Recommended).")
        trocr_size = st.sidebar.selectbox("Model Size", ["Base", "Large"], index=0)
    
    st.sidebar.info("TrOCR is a local transformer specifically trained for handwriting.")
    trocr_segmentation = st.sidebar.checkbox("Line-by-Line Processing", value=True, help="Recommended for multi-line notes.")
    # Calculate ID here to avoid NameError in other blocks
    trocr_model_id = "microsoft/trocr-large-handwritten" if trocr_size == "Large" else "microsoft/trocr-base-handwritten"

# Preprocessing Settings
st.sidebar.markdown("---")
st.sidebar.header("Image Preprocessing")
bin_method = st.sidebar.selectbox("Binarization Mode", ["Adaptive (Low Light)", "Clean (High Contrast)"], help="Use 'Clean' for white paper with dark ink.")
denoise_strength = st.sidebar.slider("Denoise Strength", 0, 5, 2)
use_preprocessed = st.sidebar.checkbox("Apply Preprocessing before OCR", value=True)

if use_preprocessed:
    st.sidebar.warning("💡 Tip: If your image is already clear, uncheck 'Apply Preprocessing' for better local OCR results.")

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
    method_key = 'clean' if "Clean" in bin_method else 'adaptive'
    processed_gray = preprocess_image(image_bgr, denoise_strength=denoise_strength, method=method_key)
    
    with col2:
        st.subheader("Binarized Image")
        st.image(processed_gray, use_container_width=True)
        
    st.markdown("---")
    ground_truth = st.text_area("Ground Truth Text (Optional for Accuracy Evaluation)", "", help="Enter the exact expected text here. After OCR runs, Word and Character Accuracy will be evaluated.")

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
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    
                    # Convert to PIL for Gemini tool
                    # If using preprocessed, it's grayscale, Gemini prefers RGB
                    if use_preprocessed:
                        pil_input = Image.fromarray(cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB))
                    else:
                        pil_input = image
                        
                    with st.spinner("Gemini is reading..."):
                        try:
                            model_id = "gemini-2.0-flash" # Updated based on diagnostic listing
                            model = genai.GenerativeModel(model_id)
                            response = model.generate_content([
                                pil_input,
                                "SYSTEM: You are a verbatim OCR engine. "
                                "TASK: Transcribe exactly what is written in the image. "
                                "STRICT RULE: Do not include ANY introductory text, metadata, descriptions, or explanations. "
                                "STRICT RULE: Only output the literal text found in the image. "
                                "STRICT RULE: Do not wrap the output in markdown code blocks. "
                                "Preserve line breaks as they appear."
                            ])
                            full_text = response.text.strip()
                        except Exception as e:
                            st.error(f"Gemini API Error: {str(e)}")
                            full_text = ""
                        # Final cleanliness check: remove markdown code block markers if they exist
                        if full_text.startswith("```"):
                            # This handles cases where Gemini ignores the strict rule
                            lines = full_text.split("\n")
                            if lines[0].startswith("```"):
                                lines = lines[1:]
                            if lines and lines[-1].startswith("```"):
                                lines = lines[:-1]
                            full_text = "\n".join(lines).strip()
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
                    # Final crash prevention: check if we have enough memory
                    import psutil
                    available_ram_gb = psutil.virtual_memory().available / (1024**3)
                    if available_ram_gb < 1.0:
                        raise MemoryError(f"Critically Low RAM: {available_ram_gb:.1f}GB. TrOCR requires at least 1.5GB to start safely.")
                    
                    print(f"DEBUG: Loading TrOCR Processor and Model ({model_id})...")
                    processor = TrOCRProcessor.from_pretrained(model_id, local_files_only=True)
                    # Load in FP16 to save memory on 6GB GPUs
                    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    model = VisionEncoderDecoderModel.from_pretrained(
                        model_id, 
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=True
                    ).to(device)
                    return processor, model

                with st.spinner(f"Loading TrOCR {trocr_size} Model (this may take a minute)..."):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    processor, model = load_trocr(trocr_model_id)
                
                def process_trocr_image(img):
                    # Convert to PIL RGB if needed
                    if len(img.shape) == 2: # Gray
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    pil_img = PILImage.fromarray(img)
                    
                    # Ensure input also uses correct dtype
                    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device, dtype=dtype)
                    generated_ids = model.generate(
                        pixel_values, 
                        max_new_tokens=64, 
                        num_beams=4, # Increased from 1 to 4 for better spelling accuracy
                        early_stopping=True
                    )
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    # Cleanup common TrOCR hallucinations
                    generated_text = generated_text.replace("#", "").replace("  ", " ").strip()
                    if generated_text.startswith('"') and generated_text.endswith('"'):
                        generated_text = generated_text[1:-1]
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
            except torch.cuda.OutOfMemoryError:
                st.error("🚨 CUDA Out of Memory! The model is too large for your GPU. Try switching to 'Base' model size in the sidebar.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                st.error(f"TrOCR Error: {str(e)}")
                print(f"DEBUG: TrOCR Error detail: {e}")

        # --- 3. EasyOCR Path ---
        elif model_choice == "EasyOCR (Lightweight)":
            try:
                import easyocr
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
            st.text_area("OCR Result (Verbatim)", full_text, height=450)
            
            # Compute Accuracies if Ground Truth is provided
            if ground_truth.strip():
                try:
                    from src.utils import compute_cer, compute_wer
                    cer = compute_cer(full_text, ground_truth)
                    wer = compute_wer(full_text, ground_truth)
                    
                    char_acc = max(0.0, 1.0 - cer) * 100
                    word_acc = max(0.0, 1.0 - wer) * 100
                    
                    st.markdown("### Accuracy Metrics")
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric(label="Character Accuracy", value=f"{char_acc:.2f}%", help="Based on Character Error Rate (CER). How many characters are correctly transcribed.")
                    col_m2.metric(label="Word Accuracy", value=f"{word_acc:.2f}%", help="Based on Word Error Rate (WER). How many full words are correctly transcribed.")
                    st.markdown("---")
                except ImportError:
                    st.warning("Could not calculate accuracy. Please ensure 'editdistance' is installed (`pip install editdistance`).")
            
            st.download_button("Download Text", full_text, file_name="handwriting_transcription.txt")
        else:
            st.warning("OCR finished but no text was generated. Please try a different model or adjust settings.")

else:
    st.info("👋 Welcome! Please upload a JPG or PNG image of handwriting from the sidebar to start.")
