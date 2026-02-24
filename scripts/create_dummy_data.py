import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import string

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_image(text, width=128, height=32):
    # Create white image
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Load default font (or try to load a system font)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Text size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # Center text
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    
    draw.text((x, y), text, fill=0, font=font)
    
    # Convert to numpy
    img_np = np.array(img)
    
    # Add noise
    noise = np.random.normal(0, 10, img_np.shape).astype(np.uint8)
    img_np = cv2.add(img_np, noise)
    
    return img_np

def main():
    base_dir = "data/dummy"
    create_directory(base_dir)
    
    labels_file = os.path.join(base_dir, "labels.txt")
    
    characters = string.ascii_lowercase + string.digits
    
    with open(labels_file, "w") as f:
        for i in range(100): # Generate 100 samples
            # Random text length 3-8
            text_len = random.randint(3, 8)
            text = "".join(random.choices(characters, k=text_len))
            
            filename = f"sample_{i}.png"
            filepath = os.path.join(base_dir, filename)
            
            img = generate_image(text)
            cv2.imwrite(filepath, img)
            
            f.write(f"{filename} {text}\n")
    
    print(f"Generated 100 dummy samples in {base_dir}")

if __name__ == "__main__":
    main()
