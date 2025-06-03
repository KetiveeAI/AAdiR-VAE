import os
import sys
import subprocess
import time
import webbrowser
import io

def configure_unicode_console():
    # Configure console for UTF-8 output
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'UTF-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    configure_unicode_console()
    
    print("🚀 Starting lightweight training and TensorBoard...\n")
    
    # Create necessary directories
    os.makedirs("logs/tensorboard_lite", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Set environment variables for CUDA optimization
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use only the first GPU
    env["PYTHONIOENCODING"] = "utf-8"  # Ensure Python uses UTF-8 encoding

    print("🧠 Applied lightweight CUDA memory settings:")
    print("  - Max split size: 256MB")
    print("  - CUDA memory caching: Disabled")
    print("  - Mixed precision (AMP): Enabled (ensure it's set in your training script)")
    print("  - Active GPU: 0\n")

    # Start TensorBoard
    print("📊 Launching TensorBoard at http://localhost:6006 ...")
    try:
        tensorboard_proc = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/tensorboard_lite", "--port", "6006"],
            cwd=os.getcwd(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("⚠️ TensorBoard not found. Installing it...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"])
        tensorboard_proc = subprocess.Popen(
            ["tensorboard", "--logdir", "logs/tensorboard_lite", "--port", "6006"],
            cwd=os.getcwd(),
            env=env
        )

    # Give TensorBoard some time to launch
    time.sleep(3)
    webbrowser.open("http://localhost:6006")

    # Start lightweight training script
    training_script = os.path.join("scripts", "train_light.py").replace("\\", "/")
    log_file_path = os.path.join("logs", "train_output.log").replace("\\", "/")
    
    print(f"\n🛠️  Running training script: {training_script}")
    print(f"📁 Logging output to: {log_file_path}\n")

    with open(log_file_path, "w", encoding='utf-8') as log_file:  # Added encoding here
        training_proc = subprocess.Popen(
            [sys.executable, training_script],
            cwd=os.getcwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        try:
            training_proc.wait()
        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user. Cleaning up...")
        finally:
            training_proc.terminate()
            tensorboard_proc.terminate()
            print("✅ Training finished. All processes cleaned up.")

if __name__ == "__main__":
    main()