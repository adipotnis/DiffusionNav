"""dataset for vjepa2 navigation training."""
import os
import pickle
import numpy as np
from typing import Dict, Tuple, List, Optional
import io
import threading
import time
import atexit

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import yaml
import tqdm
import lmdb


IMAGE_ASPECT_RATIO = 4 / 3

# process-local cache for trajectory data (shared across workers in same process)
_traj_cache_process_local = {}
_traj_cache_lock = threading.Lock()

# registry of lmdb connections for cleanup (per process)
_lmdb_connections = []
_lmdb_cleanup_registered = False

def _cleanup_lmdb_connections():
    """cleanup all lmdb connections in this process."""
    global _lmdb_connections
    for conn in _lmdb_connections:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
    _lmdb_connections.clear()

def _register_lmdb_cleanup():
    """register cleanup function (only once per process)."""
    global _lmdb_cleanup_registered
    if not _lmdb_cleanup_registered:
        atexit.register(_cleanup_lmdb_connections)
        _lmdb_cleanup_registered = True

def _load_traj_cached(traj_path: str) -> Dict:
    """load trajectory with process-local caching (thread-safe)."""
    with _traj_cache_lock:
        if traj_path not in _traj_cache_process_local:
            with open(traj_path, "rb") as f:
                _traj_cache_process_local[traj_path] = pickle.load(f)
        return _traj_cache_process_local[traj_path]


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


def load_camera_intrinsics(dataset_name: str, config_path: Optional[str] = None) -> Optional[Dict]:
    """load camera intrinsics from data_config.yaml for rectification."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "vint_train", "data", "data_config.yaml")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if dataset_name not in config:
            return None
        
        dataset_config = config[dataset_name]
        if "camera_metrics" not in dataset_config:
            return None
        
        cam_metrics = dataset_config["camera_metrics"]
        if "camera_matrix" not in cam_metrics or "dist_coeffs" not in cam_metrics:
            return None
        
        fx = cam_metrics["camera_matrix"]["fx"]
        fy = cam_metrics["camera_matrix"]["fy"]
        cx = cam_metrics["camera_matrix"]["cx"]
        cy = cam_metrics["camera_matrix"]["cy"]
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        k1 = cam_metrics["dist_coeffs"]["k1"]
        k2 = cam_metrics["dist_coeffs"]["k2"]
        p1 = cam_metrics["dist_coeffs"]["p1"]
        p2 = cam_metrics["dist_coeffs"]["p2"]
        k3 = cam_metrics["dist_coeffs"]["k3"]
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        
        return {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs}
    except Exception as e:
        print(f"warning: failed to load camera intrinsics: {e}")
        return None


def rectify_image(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """rectify image using camera intrinsics to remove distortion."""
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    rectified = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return rectified


def to_local_coords(positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float) -> np.ndarray:
    return (positions - curr_pos).dot(yaw_rotmat(curr_yaw))


def load_image(path: str, image_size: Tuple[int, int], camera_intrinsics: Optional[Dict] = None) -> torch.Tensor:
    """optimized image loading with reduced overhead. optionally rectifies image first."""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if camera_intrinsics is not None:
        img_array = np.array(img)
        rectified = rectify_image(
            img_array,
            camera_intrinsics["camera_matrix"],
            camera_intrinsics["dist_coeffs"]
        )
        img = Image.fromarray(rectified)
    
    w, h = img.size
    crop_h = h if w > h else int(w / IMAGE_ASPECT_RATIO)
    crop_w = int(h * IMAGE_ASPECT_RATIO) if w > h else w
    img = TF.center_crop(img, (crop_h, crop_w))
    img = img.resize(image_size, Image.BILINEAR)
    return TF.to_tensor(img)


class VJEPADataset(Dataset):
    """dataset that reads images and pkl files directly."""

    def __init__(
        self,
        data_folder: str,
        traj_names: List[str],
        image_size: Tuple[int, int],
        context_size: int,
        len_traj_pred: int,
        waypoint_spacing: int = 1,
        min_action_dist: int = 2,
        max_action_dist: int = 20,
        max_goal_dist: int = 20,
        end_slack: int = 0,
        normalize: bool = False,
        metric_waypoint_spacing: float = 0.25,
        action_normalization_factor: float = None,
        learn_angle: bool = False,
        dataset_name: Optional[str] = None,
        rectify_images: bool = False,
        use_lmdb_cache: bool = False,
        data_split_folder: Optional[str] = None,
    ):
        self.data_folder = data_folder
        self.image_size = image_size
        self.context_size = context_size
        self.len_traj_pred = len_traj_pred
        self.waypoint_spacing = waypoint_spacing
        self.min_action_dist = min_action_dist
        self.max_action_dist = max_action_dist
        self.max_goal_dist = max_goal_dist
        self.end_slack = end_slack
        self.normalize = normalize
        self.metric_waypoint_spacing = metric_waypoint_spacing
        if action_normalization_factor is None:
            action_normalization_factor = metric_waypoint_spacing * waypoint_spacing
        self.action_normalization_factor = action_normalization_factor
        self.learn_angle = learn_angle
        self.action_dim = 3 if learn_angle else 2
        self.dataset_name = dataset_name
        self.use_lmdb_cache = use_lmdb_cache
        self.data_split_folder = data_split_folder

        self.camera_intrinsics = None
        if rectify_images and dataset_name:
            self.camera_intrinsics = load_camera_intrinsics(dataset_name)
            if self.camera_intrinsics is None:
                print(f"warning: rectify_images=True but failed to load intrinsics for dataset '{dataset_name}'")

        self.traj_cache = {}
        self.index = []

        self._traj_load_count = 0
        self._traj_cache_hits = 0
        self._image_load_count = 0
        self._image_lmdb_hits = 0
        self._image_load_times = []

        # pre-load all trajectories so each worker has them cached immediately
        traj_load_start = time.time()
        print(f"[dataset init] pre-loading {len(traj_names)} trajectories into cache...")
        for traj_name in tqdm.tqdm(traj_names, desc="loading trajectories", disable=len(traj_names) < 10):
            self._load_traj(traj_name)
        traj_load_time = time.time() - traj_load_start
        print(f"[dataset init] loaded {len(traj_names)} trajectories in {traj_load_time:.2f}s (avg {traj_load_time/len(traj_names)*1000:.1f}ms per traj)")

        for traj_name in traj_names:
            traj_data = self.traj_cache[traj_name]
            traj_len = len(traj_data["position"])

            begin = (context_size - 1) * waypoint_spacing
            end = traj_len - end_slack - len_traj_pred * waypoint_spacing
            for t in range(begin, end):
                max_dist = min(max_goal_dist * waypoint_spacing, traj_len - t - 1)
                if max_dist > 0:
                    self.index.append((traj_name, t, max_dist))
        
        print(f"[dataset init] created {len(self.index)} samples from {len(traj_names)} trajectories")

        # keep traj_names reference for worker processes (shared via process-local cache)
        self._traj_names = traj_names

        self._image_cache = None
        if self.use_lmdb_cache:
            self._build_lmdb_cache()

        # register cleanup on exit to prevent semaphore leaks
        atexit.register(self.close)

    def _load_traj(self, traj_name: str) -> Dict:
        """load trajectory with caching and logging."""
        self._traj_load_count += 1

        # instance cache is lock-free (fastest path)
        if traj_name in self.traj_cache:
            self._traj_cache_hits += 1
            return self.traj_cache[traj_name]

        load_start = time.time()
        traj_path = os.path.join(self.data_folder, traj_name, "traj_data.pkl")
        # process-local cache shares across workers in same process
        traj_data = _load_traj_cached(traj_path)
        load_time = time.time() - load_start
        self.traj_cache[traj_name] = traj_data

        if load_time > 0.1 and self._traj_load_count <= 5:
            print(f"[traj load] slow load: {traj_name} took {load_time*1000:.1f}ms")
        
        return traj_data

    def _process_image_to_tensor(self, image_path: str) -> Optional[torch.Tensor]:
        """process image and return tensor - used for cache building."""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if self.camera_intrinsics is not None:
                img_array = np.array(img)
                rectified = rectify_image(
                    img_array,
                    self.camera_intrinsics["camera_matrix"],
                    self.camera_intrinsics["dist_coeffs"]
                )
                img = Image.fromarray(rectified)

            w, h = img.size
            crop_h = h if w > h else int(w / IMAGE_ASPECT_RATIO)
            crop_w = int(h * IMAGE_ASPECT_RATIO) if w > h else w
            img = TF.center_crop(img, (crop_h, crop_w))
            img = img.resize(self.image_size, Image.BILINEAR)
            return TF.to_tensor(img)  # returns [C, H, W] tensor in [0, 1]
        except Exception as e:
            print(f"[lmdb cache] error processing {image_path}: {e}")
            return None

    def _build_lmdb_cache(self):
        """build lmdb cache with pre-processed images (tensors)."""
        if not self.use_lmdb_cache or not self.data_split_folder:
            return

        base_cache_filename = f"vjepa_dataset_{self.dataset_name or 'default'}.lmdb"

        # prefer local copy (faster I/O on compute nodes)
        lmdb_local_dir = os.environ.get("LMDB_LOCAL_DIR")
        if lmdb_local_dir and os.path.exists(os.path.join(lmdb_local_dir, base_cache_filename)):
            cache_filename = os.path.join(lmdb_local_dir, base_cache_filename)
            print(f"[lmdb cache] using local copy: {cache_filename}")
        else:
            cache_filename = os.path.join(self.data_split_folder, base_cache_filename)

        if not os.path.exists(cache_filename):
            # lock file lets concurrent processes coordinate cache build
            lock_file = cache_filename + ".lock"
            if os.path.exists(lock_file):
                print(f"[lmdb cache] waiting for cache to be built by another process...")
                wait_start = time.time()
                while os.path.exists(lock_file) and not os.path.exists(cache_filename):
                    time.sleep(1)
                wait_time = time.time() - wait_start
                if wait_time > 0:
                    print(f"[lmdb cache] waited {wait_time:.1f}s for cache to be ready")
                if os.path.exists(cache_filename):
                    print(f"lmdb cache ready: {cache_filename}")
                else:
                    print(f"warning: lock file exists but cache not found, building cache...")
            
            if not os.path.exists(cache_filename):
                with open(lock_file, "w") as f:
                    f.write(str(os.getpid()))

                try:
                    build_start = time.time()
                    print(f"[lmdb cache] building cache: {cache_filename}")
                    print("[lmdb cache] this is a one-time operation and may take a while...")

                    collect_start = time.time()
                    image_paths = set()
                    for traj_name, t, _ in tqdm.tqdm(self.index, desc="collecting image paths"):
                        image_path = os.path.join(self.data_folder, traj_name, f"{t}.jpg")
                        if os.path.exists(image_path):
                            image_paths.add(image_path)
                    collect_time = time.time() - collect_start
                    print(f"[lmdb cache] collected {len(image_paths)} unique image paths in {collect_time:.1f}s")

                    write_start = time.time()
                    with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                        with image_cache.begin(write=True) as txn:
                            cached_count = 0
                            failed_count = 0
                            for image_path in tqdm.tqdm(image_paths, desc="processing and caching images"):
                                try:
                                    tensor = self._process_image_to_tensor(image_path)
                                    if tensor is not None:
                                        # tensor stored as float32 [C, H, W] bytes, keyed by image path
                                        tensor_bytes = tensor.numpy().tobytes()
                                        key = image_path.encode()
                                        txn.put(key, tensor_bytes)
                                        cached_count += 1
                                    else:
                                        failed_count += 1
                                except Exception as e:
                                    failed_count += 1
                                    if failed_count <= 5:
                                        print(f"[lmdb cache] warning: failed to cache {image_path}: {e}")
                    write_time = time.time() - write_start
                    build_time = time.time() - build_start
                    print(f"[lmdb cache] cached {cached_count} processed images, {failed_count} failed")
                    print(f"[lmdb cache] processing+write time: {write_time:.1f}s, total build time: {build_time:.1f}s")
                    print(f"[lmdb cache] cache built: {cache_filename}")
                finally:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)

        # read-only open is lock-free and supports many concurrent readers
        try:
            open_start = time.time()
            self._image_cache = lmdb.open(
                cache_filename,
                readonly=True,
                lock=False,
                max_readers=256,
                metasync=False,
            )
            _lmdb_connections.append(self._image_cache)
            _register_lmdb_cleanup()
            open_time = time.time() - open_start
            print(f"[lmdb cache] opened cache in {open_time*1000:.1f}ms")
        except Exception as e:
            print(f"[lmdb cache] warning: failed to open cache: {e}")
            self._image_cache = None
            self.use_lmdb_cache = False
    
    def __getstate__(self):
        """custom pickle state - don't pickle lmdb connection."""
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        """restore state and open lmdb cache with proper cleanup support."""
        restore_start = time.time()
        self.__dict__ = state

        self._traj_load_count = 0
        self._traj_cache_hits = 0
        self._image_load_count = 0
        self._image_lmdb_hits = 0
        self._image_load_times = []

        # clear instance cache; workers will lazy-load via process-local cache to save memory
        if hasattr(self, '_traj_names') and self._traj_names:
            print(f"[worker init] worker {os.getpid()} will load {len(self._traj_names)} trajectories on-demand")
            self.traj_cache = {}

        if self.use_lmdb_cache:
            base_cache_filename = f"vjepa_dataset_{self.dataset_name or 'default'}.lmdb"
            lmdb_local_dir = os.environ.get("LMDB_LOCAL_DIR")
            if lmdb_local_dir and os.path.exists(os.path.join(lmdb_local_dir, base_cache_filename)):
                cache_filename = os.path.join(lmdb_local_dir, base_cache_filename)
                print(f"[worker init] using local lmdb copy: {cache_filename}")
            else:
                cache_filename = os.path.join(self.data_split_folder, base_cache_filename)
            if os.path.exists(cache_filename):
                try:
                    cache_open_start = time.time()
                    self._image_cache = lmdb.open(
                        cache_filename,
                        readonly=True,
                        lock=False,
                        max_readers=256,
                        metasync=False,
                    )
                    _lmdb_connections.append(self._image_cache)
                    _register_lmdb_cleanup()
                    cache_open_time = time.time() - cache_open_start
                    print(f"[worker init] worker {os.getpid()} opened lmdb cache in {cache_open_time*1000:.1f}ms")
                except Exception as e:
                    print(f"[worker init] warning: worker {os.getpid()} failed to open lmdb cache: {e}")
                    self._image_cache = None
                    self.use_lmdb_cache = False
            else:
                print(f"[worker init] warning: worker {os.getpid()} lmdb cache not found: {cache_filename}")
                self._image_cache = None
                self.use_lmdb_cache = False
        else:
            self._image_cache = None
        
        restore_time = time.time() - restore_start
        if restore_time > 1.0:
            print(f"[worker init] worker {os.getpid()} initialization took {restore_time:.2f}s")
    
    def __del__(self):
        """cleanup lmdb connection on destruction to prevent leaks."""
        self.close()
    
    def close(self):
        """explicitly close lmdb connection to prevent semaphore leaks."""
        if hasattr(self, '_image_cache') and self._image_cache is not None:
            try:
                if self._image_cache in _lmdb_connections:
                    _lmdb_connections.remove(self._image_cache)
                self._image_cache.close()
                self._image_cache = None
            except Exception:
                pass
    
    def _load_image(self, traj_name: str, t: int) -> torch.Tensor:
        """load image with caching, logging, and performance tracking."""
        load_start = time.time()
        self._image_load_count += 1
        image_path = os.path.join(self.data_folder, traj_name, f"{t}.jpg")

        if self.use_lmdb_cache and self._image_cache is not None:
            try:
                lmdb_start = time.time()
                with self._image_cache.begin(write=False, buffers=True) as txn:
                    tensor_bytes = txn.get(image_path.encode(), default=None)
                lmdb_time = time.time() - lmdb_start

                if tensor_bytes is not None:
                    self._image_lmdb_hits += 1

                    decode_start = time.time()
                    # cached tensor is float32 numpy [C, H, W]
                    tensor_array = np.frombuffer(tensor_bytes, dtype=np.float32)
                    tensor_array = tensor_array.reshape(3, self.image_size[1], self.image_size[0])
                    # copy makes the buffer writable so torch doesn't warn
                    tensor_array = tensor_array.copy()
                    result = torch.from_numpy(tensor_array)
                    decode_time = time.time() - decode_start

                    load_time = time.time() - load_start
                    self._image_load_times.append(load_time)

                    if load_time > 0.05 and self._image_load_count <= 10:
                        print(f"[image load] slow lmdb load: {traj_name}/{t}.jpg took {load_time*1000:.1f}ms "
                              f"(lmdb={lmdb_time*1000:.1f}ms, decode={decode_time*1000:.1f}ms)")
                    return result
            except Exception as e:
                if self._image_load_count <= 5:
                    print(f"[image load] lmdb error for {traj_name}/{t}.jpg: {e}, falling back to disk")

        result = load_image(image_path, self.image_size, self.camera_intrinsics)
        load_time = time.time() - load_start
        self._image_load_times.append(load_time)
        if load_time > 0.1 and self._image_load_count <= 10:
            print(f"[image load] slow disk load: {traj_name}/{t}.jpg took {load_time*1000:.1f}ms")
        return result

    def _compute_actions(self, traj_data: Dict, curr_time: int) -> np.ndarray:
        end = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        positions = traj_data["position"][curr_time:end:self.waypoint_spacing]
        yaw = traj_data["yaw"][curr_time:end:self.waypoint_spacing]
        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if len(yaw) < self.len_traj_pred + 1:
            pad_len = self.len_traj_pred + 1 - len(yaw)
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], pad_len)])
            positions = np.concatenate([positions, np.tile(positions[-1:], (pad_len, 1))])

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        if self.learn_angle:
            actions = np.concatenate([waypoints[1:], (yaw[1:] - yaw[0])[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        # normalize from meters to unitless waypoint counts
        if self.normalize:
            actions[:, :2] /= self.action_normalization_factor

        return actions

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        """get item with performance tracking."""
        item_start = time.time()

        traj_name, curr_time, max_dist = self.index[idx]
        traj_data = self._load_traj(traj_name)
        traj_len = len(traj_data["position"])

        context_times = list(range(
            curr_time - (self.context_size - 1) * self.waypoint_spacing,
            curr_time + 1,
            self.waypoint_spacing,
        ))
        images = [self._load_image(traj_name, t) for t in context_times]
        obs_image = torch.cat(images, dim=0)
        actions = self._compute_actions(traj_data, curr_time)

        # randomized goal frame so goal_xy != actions[-1] in general
        min_offset = max(1, self.min_action_dist) * self.waypoint_spacing
        max_offset = max(min_offset, int(max_dist))
        goal_offset = int(np.random.randint(min_offset, max_offset + 1))
        goal_time = min(curr_time + goal_offset, traj_len - 1)
        goal_image = self._load_image(traj_name, goal_time)

        # goal_xy is the goal frame position expressed in the current local frame
        curr_pos = np.asarray(traj_data["position"][curr_time])
        curr_yaw = traj_data["yaw"][curr_time]
        if isinstance(curr_yaw, np.ndarray):
            curr_yaw = float(np.asarray(curr_yaw).squeeze())
        goal_pos_world = np.asarray(traj_data["position"][goal_time])
        goal_xy_local = to_local_coords(goal_pos_world[None], curr_pos, curr_yaw)[0]
        if self.normalize:
            goal_xy_local = goal_xy_local / self.action_normalization_factor
        goal_xy = torch.as_tensor(goal_xy_local, dtype=torch.float32)

        action_mask = torch.tensor(1.0, dtype=torch.float32)

        item_time = time.time() - item_start
        if item_time > 0.2 and idx < 10:
            print(f"[getitem] slow item {idx}: {item_time*1000:.1f}ms (traj: {traj_name}, time: {curr_time})")

        return (
            obs_image.float(),
            goal_image.float(),
            goal_xy,
            torch.as_tensor(actions, dtype=torch.float32),
            action_mask,
        )
    
    def print_stats(self):
        """print performance statistics."""
        import os
        import numpy as np
        worker_id = os.getpid()
        
        print(f"\n[dataset stats] worker {worker_id}:")
        print(f"  trajectory loads: {self._traj_load_count}")
        print(f"  trajectory cache hits: {self._traj_cache_hits}")
        if self._traj_load_count > 0:
            hit_rate = self._traj_cache_hits / self._traj_load_count * 100
            print(f"  trajectory cache hit rate: {hit_rate:.1f}%")
        
        print(f"  image loads: {self._image_load_count}")
        print(f"  lmdb cache hits: {self._image_lmdb_hits}")
        if self._image_load_count > 0:
            lmdb_hit_rate = self._image_lmdb_hits / self._image_load_count * 100
            print(f"  lmdb cache hit rate: {lmdb_hit_rate:.1f}%")
            
            if self._image_load_times:
                load_times = np.array(self._image_load_times)
                print(f"  image load times: avg {load_times.mean()*1000:.1f}ms, "
                      f"median {np.median(load_times)*1000:.1f}ms, "
                      f"max {load_times.max()*1000:.1f}ms")
        print()




