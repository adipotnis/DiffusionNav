"""generate spoof data to test the training pipeline."""
import os
import pickle
import argparse
import numpy as np
from PIL import Image


def generate_trajectory(num_frames: int = 30) -> dict:
    """generate random trajectory data."""
    # random walk for positions
    dt = 0.1
    velocities = np.random.randn(num_frames, 2) * 0.5
    positions = np.cumsum(velocities, axis=0) * dt
    
    # random yaw (heading direction)
    yaw = np.cumsum(np.random.randn(num_frames) * 0.1)
    
    return {
        "position": positions.astype(np.float32),
        "yaw": yaw.astype(np.float32),
    }


def generate_random_image(size: tuple = (256, 256)) -> Image.Image:
    """generate random rgb image."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def generate_spoof_dataset(
    output_dir: str,
    num_trajectories: int = 2,
    num_frames: int = 30,
    image_size: tuple = (256, 256),
):
    """generate complete spoof dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    traj_names = []
    
    for traj_idx in range(num_trajectories):
        traj_name = f"traj_{traj_idx:04d}"
        traj_dir = os.path.join(output_dir, traj_name)
        os.makedirs(traj_dir, exist_ok=True)
        
        # generate trajectory data
        traj_data = generate_trajectory(num_frames)
        with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)
        
        # generate images
        for frame_idx in range(num_frames):
            img = generate_random_image(image_size)
            img.save(os.path.join(traj_dir, f"{frame_idx}.jpg"))
        
        traj_names.append(traj_name)
        print(f"generated {traj_name} with {num_frames} frames")
    
    # create train/test split dirs
    train_dir = os.path.join(output_dir, "splits", "train")
    test_dir = os.path.join(output_dir, "splits", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # split: first half train, second half test (or all train if only 1)
    split_idx = max(1, num_trajectories // 2)
    train_names = traj_names[:split_idx]
    test_names = traj_names[split_idx:] if split_idx < num_trajectories else traj_names
    
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        f.write("\n".join(train_names))
    
    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        f.write("\n".join(test_names))
    
    print(f"\ngenerated {num_trajectories} trajectories in {output_dir}")
    print(f"train: {len(train_names)} trajectories")
    print(f"test: {len(test_names)} trajectories")
    print(f"\nto use, update config/vjepa.yaml:")
    print(f"  data_folder: {os.path.abspath(output_dir)}")
    print(f"  train: {os.path.abspath(train_dir)}/")
    print(f"  test: {os.path.abspath(test_dir)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spoof data for testing")
    parser.add_argument("--output", "-o", default="datasets/spoof", help="output directory")
    parser.add_argument("--num-traj", "-n", type=int, default=2, help="number of trajectories")
    parser.add_argument("--num-frames", "-f", type=int, default=30, help="frames per trajectory")
    parser.add_argument("--image-size", type=int, default=256, help="image size")
    args = parser.parse_args()
    
    generate_spoof_dataset(
        output_dir=args.output,
        num_trajectories=args.num_traj,
        num_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
    )

