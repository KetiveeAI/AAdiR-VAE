# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================
# Entrypoint for training high-quality/3D/video model (see scripts/train.py).
# ==============================================================

import os
import sys
import subprocess
import time

def main():
    print("Starting training and TensorBoard...")
    
    # Create required directories
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Set environment variables for CUDA optimization
    env = os.environ.copy()
    
    # Enhanced CUDA memory settings
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
      # Force smaller batch size and correct accumulation steps from config
    env["FORCE_BATCH_SIZE"] = "4"  # Increased from 2 to 4
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure using single GPU
    
    print("\nApplied memory optimization settings:")
    print("- Reduced batch size: 2")  # Updated to match
    print("- Gradient accumulation steps: 16")  # Updated to match config
    print("- Mixed precision training enabled")
    print("- Gradient checkpointing enabled") 
    print("- CUDA memory caching disabled")
    print("- Maximum CUDA split size: 512MB")
    
    # Start TensorBoard
    print("Starting TensorBoard...")
    try:
        tensorboard_proc = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/tensorboard", "--port", "6006"],
            cwd=os.getcwd(),
            env=env
        )
    except FileNotFoundError:
        print("TensorBoard not found. Installing TensorBoard...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"])
        tensorboard_proc = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/tensorboard", "--port", "6006"],
            cwd=os.getcwd(),
            env=env
        )

    # Give TensorBoard time to start
    time.sleep(3)
    
    # Start training script
    print("\nStarting training script...")
    training_script = os.path.join("scripts", "train.py")
    training_proc = subprocess.Popen(
        [sys.executable, training_script],
        cwd=os.getcwd(),
        env=env
    )
    
    print("\nTraining is running!")
    print("Access TensorBoard at: http://localhost:6006")
    print("\nPress Ctrl+C to stop both processes")

    try:
        training_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping processes...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        # Ensure both processes are terminated
        if training_proc.poll() is None:
            training_proc.terminate()
        if tensorboard_proc.poll() is None:
            tensorboard_proc.terminate()
        print("Processes stopped")

if __name__ == "__main__":
    main()
