import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import os

# Set Hugging Face cache to D: drive only if it exists (local setup)
# This prevents errors on Streamlit Cloud (Linux) where D: doesn't exist.
CACHE_PATH = r"D:\huggingface_cache"
if os.path.exists(CACHE_PATH):
    os.environ["HF_HOME"] = CACHE_PATH
    os.environ["TRANSFORMERS_CACHE"] = CACHE_PATH
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

# Load labels for automatic evaluation
LABELS_PATH = os.path.join("data", "dummy", "labels.txt")
USER_LABELS_PATH = "user_labels.json"
ground_truth_lookup = {}

# Load built-in labels
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    ground_truth_lookup[parts[0]] = parts[1]
    except Exception as e:
        print(f"Error loading labels: {e}")

# Load user-saved labels
import json
if os.path.exists(USER_LABELS_PATH):
    try:
        with open(USER_LABELS_PATH, "r") as f:
            user_data = json.load(f)
            ground_truth_lookup.update(user_data)
    except Exception as e:
        print(f"Error loading user labels: {e}")

# Page Config
st.set_page_config(page_title="Handwritten OCR System", layout="wide")

# Title and sidebar
st.title("Handwritten Text Recognition System ✍️")
st.markdown("---")

st.sidebar.header("OCR Engine Selection")
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["TrOCR (High-Accuracy Local)", "EasyOCR (Lightweight)", "Gemini Vision API (Cloud)"],
    index=0
)

# Reference / Evaluation Mode
st.sidebar.markdown("---")
st.sidebar.subheader("Evaluation Settings")
use_gemini_ref = False

# Only show Gemini Ref if Gemini isn't the primary and if key is provided
if model_choice != "Gemini Vision API (Cloud)":
    use_gemini_ref = st.sidebar.checkbox("Use Gemini as Reference", value=False, help="Uses Gemini to generate 'pseudo-ground truth' for accuracy metrics.")
    if use_gemini_ref:
        st.sidebar.warning("🤖 Gemini Reference active. Ensure API key is entered below.")

# Model specific settings
gemini_api_key = ""
trocr_size = "Base"
trocr_segmentation = True
trocr_model_id = "microsoft/trocr-base-handwritten"

if model_choice == "Gemini Vision API (Cloud)" or use_gemini_ref:
    st.sidebar.markdown("### Gemini Settings")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Required for Gemini main or reference mode.")
    st.sidebar.info("💡 Note: If you hit quota limits (429), use 'Consensus Mode' for local evaluation instead.")

if "TrOCR" in model_choice:
    st.sidebar.markdown("### TrOCR Settings")
    import psutil
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    if available_ram_gb < 3.0:
        st.sidebar.warning(f"⚠️ **Low RAM ({available_ram_gb:.1f}GB)**: Using 'Base' model size.")
        trocr_size = "Base"
    else:
        trocr_size = st.sidebar.selectbox("Model Size", ["Base", "Large"], index=0)
    
    st.sidebar.info("TrOCR is a local transformer specifically trained for handwriting.")
    trocr_segmentation = st.sidebar.checkbox("Line-by-Line Processing", value=True, help="Recommended for multi-line notes.")
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
        st.image(image, width='stretch') # The warning suggests replacing with width='stretch' but use_container_width is still common. Let's try width='stretch' if possible, or just keep it if it's not the cause of CRASH.
    
    # Preprocessing
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    method_key = 'clean' if "Clean" in bin_method else 'adaptive'
    processed_gray = preprocess_image(image_bgr, denoise_strength=denoise_strength, method=method_key)
    
    with col2:
        st.subheader("Binarized Image")
        st.image(processed_gray, width='stretch')
        
    st.markdown("---")
    st.subheader("🎯 Evaluation")
    
    # Automatic Ground Truth Lookup
    auto_gt = ground_truth_lookup.get(uploaded_file.name, "")
    if auto_gt:
        st.success(f"✅ Found ground truth for `{uploaded_file.name}` in dataset.")
    
    if use_gemini_ref:
        st.warning("🤖 **Reference Mode Active**: Gemini will be used as ground truth.")
        ground_truth = "" # Will be filled during OCR run
    else:
        st.info("Enter the **Expected Text** below (or upload a file from `data/dummy` for auto-fill).")
        ground_truth = st.text_area("Expected Text", value=auto_gt, help="Enter the exact expected text here. If left blank, accuracy won't be calculated unless Reference Mode is on.")

    if st.button("🚀 Perform OCR"):
        st.info(f"Running OCR with {model_choice}...")
        full_text = ""
        ref_text = "" # For pseudo-ground truth
        
        # Decide which image to feed to OCR
        ocr_input = processed_gray if use_preprocessed else image_np

        # --- PRE-STEP: Gemini Reference ---
        if use_gemini_ref:
            if not gemini_api_key:
                st.error("Please enter your Gemini API Key in the sidebar for Reference Mode.")
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key)
                    model_genai = genai.GenerativeModel('gemini-2.0-flash')
                    
                    if use_preprocessed:
                        pil_input = Image.fromarray(cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB))
                    else:
                        pil_input = image
                    
                    with st.spinner("🤖 Gemini is generating reference transcription..."):
                        response = model_genai.generate_content([
                            pil_input,
                            "SYSTEM: You are a verbatim OCR engine. Output only the literal text found in the image. No preamble."
                        ])
                        ref_text = response.text.strip()
                        st.success("🤖 Reference transcription acquired!")
                except Exception as e:
                    st.error(f"Gemini Reference Error: {str(e)}")

        # --- 1. Gemini Vision Path (Main) ---
        if model_choice == "Gemini Vision API (Cloud)":
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
                    # Load in FP16 to save memory on 6GB GPUs
                    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    try:
                        # Update: Using use_fast=False for better compatibility in cloudy environments
                        processor = TrOCRProcessor.from_pretrained(trocr_model_id, use_fast=False)
                        model = VisionEncoderDecoderModel.from_pretrained(
                            trocr_model_id, 
                            torch_dtype=dtype,
                            low_cpu_mem_usage=True
                        ).to(device)
                    except Exception as e:
                        st.error(f"Failed to load TrOCR model. Ensure you have an internet connection or the model is cached. Error: {e}")
                        st.stop() # Stop execution if model loading fails
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
                    
                    # Generate with scores to calculate confidence
                    outputs = model.generate(
                        pixel_values, 
                        max_new_tokens=64, 
                        num_beams=4,
                        early_stopping=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    generated_ids = outputs.sequences
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Calculate confidence using sequence scores
                    # For beam search, we use sequences_scores if available
                    if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                        # Log probability to probability
                        conf = torch.exp(outputs.sequences_scores).item()
                        # Normalize a bit as it can be very small
                        conf = min(1.0, conf * 1.5) # Heuristic for display
                    else:
                        conf = 0.85 # Fallback
                        
                    # Cleanup common TrOCR hallucinations
                    generated_text = generated_text.replace("#", "").replace("  ", " ").strip()
                    if generated_text.startswith('"') and generated_text.endswith('"'):
                        generated_text = generated_text[1:-1]
                    return generated_text, conf

                if trocr_segmentation:
                    with st.spinner("Segmenting lines and recognizing..."):
                        lines = segment_lines(ocr_input)
                        if not lines:
                            st.warning("No text lines detected. Attempting full image recognition.")
                            lines = [ocr_input]
                        
                        progress_bar = st.progress(0)
                        temp_text = []
                        confidences = []
                        for i, line_img in enumerate(lines):
                            line_out, line_conf = process_trocr_image(line_img)
                            temp_text.append(line_out)
                            confidences.append(line_conf)
                            progress_bar.progress((i + 1) / len(lines))
                        
                        full_text = "\n".join(temp_text)
                        model_confidence = sum(confidences) / len(confidences) if confidences else 0
                else:
                    with st.spinner("TrOCR is reading full image..."):
                        full_text, model_confidence = process_trocr_image(ocr_input)
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
                    results = reader.readtext(ocr_input, paragraph=True)
                    # For EasyOCR, result is [[box, text, confidence], ...]
                    # If paragraph=True, we might need to handle it differently to get individual scores
                    # Let's run it normally to get scores if needed
                    detailed_results = reader.readtext(ocr_input)
                    if detailed_results:
                        model_confidence = sum([r[2] for r in detailed_results]) / len(detailed_results)
                    else:
                        model_confidence = 0.0
                        
                    results = sorted(results, key=lambda r: r[0][0][1])
                    full_text = "\n".join([r[1] for r in results])
            except Exception as e:
                st.error(f"EasyOCR Error: {str(e)}")

        # --- 4. AUTO-REFERENCE LOGIC (Ensures Word/Char Accuracy) ---
        # If no Ground Truth and not using Gemini, run the "other" model as reference
        # This allows us to show Word/Char Accuracy relative to AI consensus
        auto_ref_text = ""
        if not ground_truth.strip() and not use_gemini_ref and model_choice != "Gemini Vision API (Cloud)":
            st.info("🔄 Running Auto-Reference to calculate Word/Character Accuracy...")
            
            # Use original image for reference if binarized fails/is too harsh
            ref_input = image_np # Always try original grayscale first for reference robustness
            if len(ref_input.shape) == 3:
                ref_input = cv2.cvtColor(ref_input, cv2.COLOR_RGB2GRAY)

            # Sub-step: Run the model NOT chosen as primary
            if model_choice == "TrOCR (High-Accuracy Local)":
                try:
                    import easyocr
                    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                    results = reader.readtext(ref_input)
                    if not results: # Try preprocessed as fallback
                        results = reader.readtext(ocr_input)
                    auto_ref_text = " ".join([r[1] for r in sorted(results, key=lambda r: r[0][0][1])])
                except Exception as e:
                    print(f"Auto-Ref EasyOCR Error: {e}")
            else:
                try:
                    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                    from PIL import Image as PILImage
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    @st.cache_resource
                    def load_ref_trocr(model_id):
                        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        try:
                            processor = TrOCRProcessor.from_pretrained(model_id)
                        except:
                            processor = TrOCRProcessor.from_pretrained(model_id, use_fast=False)
                        model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
                        return processor, model

                    ref_proc, ref_mod = load_ref_trocr(trocr_model_id)
                    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    
                    def run_ref_trocr(img):
                        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        pil_img = PILImage.fromarray(img)
                        pixel_values = ref_proc(images=pil_img, return_tensors="pt").pixel_values.to(device, dtype=dtype)
                        generated_ids = ref_mod.generate(pixel_values, max_new_tokens=64)
                        return ref_proc.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                    # For auto-ref (speed), we use full image first
                    auto_ref_text = run_ref_trocr(ref_input)
                    if not auto_ref_text.strip():
                        auto_ref_text = run_ref_trocr(ocr_input)
                except Exception as e:
                    print(f"Auto-Ref TrOCR Error: {e}")

        # --- Results Display ---
        if full_text:
            st.session_state["last_ocr"] = full_text
            st.text_area("OCR Result", full_text, height=450)
            
            # Save Label Feature
            if st.button("💾 Save as Ground Truth"):
                user_label_data = {}
                if os.path.exists(USER_LABELS_PATH):
                    with open(USER_LABELS_PATH, "r") as f:
                        user_label_data = json.load(f)
                
                user_label_data[uploaded_file.name] = full_text
                with open(USER_LABELS_PATH, "w") as f:
                    json.dump(user_label_data, f, indent=4)
                st.success(f"Saved text for `{uploaded_file.name}`! It will auto-populate next time.")
                st.rerun()

            # --- Evaluation Section ---
            # Prioritize: 1. Manual GT, 2. Gemini Ref, 3. Auto-Reference Local
            eval_target = ground_truth.strip()
            eval_source = "User Provided"
            
            if not eval_target:
                if use_gemini_ref and ref_text:
                    eval_target = ref_text
                    eval_source = "Gemini Reference"
                elif auto_ref_text:
                    eval_target = auto_ref_text
                    eval_source = "AI Reference (Cross-Model)"
            
            if eval_target:
                try:
                    from src.utils import compute_cer, compute_wer
                    cer = compute_cer(full_text, eval_target)
                    wer = compute_wer(full_text, eval_target)
                    
                    char_acc = max(0.0, 1.0 - cer) * 100
                    word_acc = max(0.0, 1.0 - wer) * 100
                    
                    st.markdown("### 🎯 Accuracy Metrics")
                    st.caption(f"*(Calculated relative to {eval_source})*")
                    
                    with st.expander("Show Comparison Data"):
                        st.text(f"Your Result: {full_text}")
                        st.text(f"Reference:  {eval_target}")
                    
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric(label="Character Accuracy", value=f"{char_acc:.2f}%")
                    col_m2.metric(label="Word Accuracy", value=f"{word_acc:.2f}%")
                    st.markdown("---")
                except ImportError:
                    st.warning("Could not calculate accuracy. Please ensure 'editdistance' is installed.")
            else:
                st.warning("⚠️ No ground truth or reference model available to calculate accurate Word/Character metrics.")
            
            st.download_button("Download Text", full_text, file_name="handwriting_transcription.txt")
        else:
            st.warning("OCR finished but no text was generated. Please try a different model or adjust settings.")
else:
    st.info("👋 Welcome! Please upload a JPG or PNG image of handwriting from the sidebar to start.")
