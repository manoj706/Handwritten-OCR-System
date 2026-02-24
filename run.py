import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True)

def main():
    python_exe = sys.executable
    
    # 1. Generate Dummy Data
    print("--- Step 1: Generating Dummy Data ---")
    run_command(f"{python_exe} scripts/create_dummy_data.py")
    
    # 2. Train Model
    print("\n--- Step 2: Training Model ---")
    # Train for 5 epochs
    run_command(f"{python_exe} src/train.py --labels_path data/dummy/labels.txt --epochs 5 --save_dir .")
    
    # 3. Validation/Test (part of train.py, but we can have separate script)
    
    # 4. Launch App
    print("\n--- Step 3: Launching Web App ---")
    run_command(f"streamlit run app.py")

if __name__ == "__main__":
    main()
