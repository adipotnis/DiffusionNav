"""convert a directory of image frames to an mp4 video."""
import argparse
import os
import glob
import re
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("error: opencv-python required. install with: pip install opencv-python")


def natural_sort_key(s: str) -> list:
    """sort strings with numbers naturally (e.g., frame_10.jpg comes after frame_2.jpg)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def get_image_files(directory: str, extensions: list = None) -> list:
    """get sorted list of image files from directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    
    # sort naturally (handle numbers correctly)
    files.sort(key=natural_sort_key)
    return files


def frames_to_video(
    input_dir: str,
    output_path: str,
    fps: float = 30.0,
    image_size: tuple = None,
    codec: str = 'mp4v',
):
    """convert directory of frames to mp4 video.
    
    args:
        input_dir: directory containing image frames
        output_path: output video file path
        fps: frames per second for output video
        image_size: (width, height) to resize frames. if None, uses first frame size
        codec: video codec (e.g., 'mp4v', 'avc1', 'x264')
    """
    if not HAS_CV2:
        raise ImportError("opencv-python required. install with: pip install opencv-python")
    
    # get sorted image files
    image_files = get_image_files(input_dir)
    if not image_files:
        raise ValueError(f"no image files found in {input_dir}")
    
    print(f"found {len(image_files)} image files")
    
    # read first frame to get dimensions
    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        raise ValueError(f"could not read first frame: {image_files[0]}")
    
    if image_size is None:
        height, width = first_frame.shape[:2]
    else:
        width, height = image_size
    
    print(f"video dimensions: {width}x{height}, fps: {fps}")
    
    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output_path}")
    
    # process frames
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"warning: could not read frame {i+1}/{len(image_files)}: {img_path}")
            continue
        
        # resize if needed
        if image_size is not None:
            frame = cv2.resize(frame, (width, height))
        elif frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f"processed {i+1}/{len(image_files)} frames")
    
    video_writer.release()
    print(f"video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="convert directory of frames to mp4 video")
    parser.add_argument("--input_dir", "-i", required=True, help="directory containing image frames")
    parser.add_argument("--output", "-o", required=True, help="output video file path (.mp4)")
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second (default: 30.0)")
    parser.add_argument("--width", type=int, help="output video width (default: use first frame width)")
    parser.add_argument("--height", type=int, help="output video height (default: use first frame height)")
    parser.add_argument("--codec", default="mp4v", help="video codec (default: mp4v, options: mp4v, avc1, x264)")
    
    args = parser.parse_args()
    
    # validate input directory
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"input directory does not exist: {args.input_dir}")
    
    # determine image size
    image_size = None
    if args.width and args.height:
        image_size = (args.width, args.height)
    elif args.width or args.height:
        raise ValueError("must specify both --width and --height, or neither")
    
    # ensure output path has .mp4 extension
    output_path = args.output
    if not output_path.endswith('.mp4'):
        output_path = output_path + '.mp4'
    
    frames_to_video(
        input_dir=args.input_dir,
        output_path=output_path,
        fps=args.fps,
        image_size=image_size,
        codec=args.codec,
    )


if __name__ == "__main__":
    main()
