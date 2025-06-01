# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================
# Entrypoint for training lightweight model (see scripts/train_light.py).
# ==============================================================

import os
import sys
import subprocess
import time

def main():
    print("Starting lightweight training and TensorBoard...")
    
    # Create required directories
    os.makedirs("logs/tensorboard_lite", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Set environment variables for CUDA optimization
    env = os.environ.copy()
    
    # Lightweight CUDA memory settings
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use single GPU
    
    print("\nApplied lightweight memory optimization settings:")
    print("- Max split size: 256MB")
    print("- CUDA memory caching disabled")
    print("- Mixed precision training enabled")
    print("- Using single GPU")
    
    # Start TensorBoard
    print("Starting TensorBoard...")
    try:
        tensorboard_proc = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/tensorboard_lite", "--port", "6006"],
            cwd=os.getcwd(), 
            env=env
        )
    except FileNotFoundError:
        print("TensorBoard not found. Installing TensorBoard...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"])
        tensorboard_proc = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/tensorboard_lite", "--port", "6006"],
            cwd=os.getcwd(),
            env=env
        )

    # Give TensorBoard time to start
    time.sleep(3)
    
    # Start lightweight training script
    print("\nStarting lightweight training script...")
    training_script = os.path.join("scripts", "train_light.py")
    training_proc = subprocess.Popen(
        [sys.executable, training_script],
        cwd=os.getcwd(),
        env=env
    )

    try:
        training_proc.wait()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
    finally:
        # Cleanup
        tensorboard_proc.terminate()
        training_proc.terminate()
        print("Training complete. Processes cleaned up.")

if __name__ == "__main__":
    main()
