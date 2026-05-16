
import os
import pickle
from PIL import Image
import io
import argparse
import tqdm
import yaml
import rosbag

from vint_train.process_data.process_data_utils import *


def main(args: argparse.Namespace):

    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # traj_name joins the last two path components, stripping the .bag suffix
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        bag_img_data, bag_traj_data = get_images_and_odom(
            b,
            config[args.dataset_name]["imtopics"],
            config[args.dataset_name]["odomtopics"],
            eval(config[args.dataset_name]["img_process_func"]),
            eval(config[args.dataset_name]["odom_process_func"]),
            rate=args.sample_rate,
            ang_offset=config[args.dataset_name]["ang_offset"],
        )

  
        if bag_img_data is None or bag_traj_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue
        # split out segments where the robot moves backwards
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)
            for i, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, f"{i}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="tartan_drive",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../datasets/tartan_drive/",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of bags to process (default: -1, all)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
