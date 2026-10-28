import os
import sys
import subprocess
import time

def run_step(description, command):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Command: {command}")
    print(f"{'='*60}\n")
    start_time = time.time()
    result = subprocess.run(command, shell=True)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {result.returncode} after {elapsed:.1f}s!")
        sys.exit(result.returncode)
    print(f"\n[DONE] Completed in {elapsed:.1f}s.")

def main():
    print("==================================================")
    print("       DUST Full Multi-Modal Training Pipeline     ")
    print("==================================================")
    
    device = 'cuda:0'
    
    # Stage 1: Train Video Swin Transformer 3D
    # Batch size 1 to stay safely within 6GB VRAM on RTX 3050
    run_step(
        "Stage 1/4: Train Video Model (Video Swin Transformer 3D)",
        f"python step_fusion_3d.py --num_epochs 10 --batch_size 1 --print_every 1 --lr 0.0005 --device {device}"
    )

    # Stage 2: Train Kinematics 1D Swin Transformer
    run_step(
        "Stage 2/4: Train Time-Series Model (Kinematics Swin 1D)",
        f"python step_fusion_1d.py --num_epochs 30 --batch_size 32 --print_every 1 --lr 0.0003 --device {device}"
    )

    # Stage 3: Extract Video Features
    run_step(
        "Stage 3/4: Extract Video Features from trained 3D model",
        f"python step_fusion_3d_get_feature.py --path_3d_model ./saved_models/step_3d/final_3d.pth --batch_size 1 --device {device}"
    )

    # Stage 4: Train Multimodal Fusion Model
    run_step(
        "Stage 4/4: Train Multimodal Fusion Model (1D Kinematics + 3D Video)",
        f"python step_fusion.py --num_epochs 30 --batch_size 16 --print_every 1 --lr 0.0003 --device {device}"
    )

    print("\n==================================================")
    print("  ALL 4 STAGES COMPLETED SUCCESSFULLY!")
    print("  Trained checkpoints are in ./saved_models/")
    print("==================================================")

if __name__ == '__main__':
    main()
