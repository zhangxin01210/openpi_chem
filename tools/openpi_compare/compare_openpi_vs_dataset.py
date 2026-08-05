#!/usr/bin/env python3
"""
Third-party offline evaluation script for OpenPI policy server.

Extracts samples from a LeRobot-format dataset, sends them to the OpenPI WebSocket
policy server for inference, and compares predicted actions against ground truth
with comprehensive metrics and visualizations.

Usage:
    python compare_openpi_vs_dataset.py \
        --dataset-root /path/to/stir20260316 \
        --openpi-root /path/to/openpi \
        --host localhost --port 8000 \
        --episode-index 0 \
        --prompt "scoop the chemical" \
        --save-dir ./results/episode0

Environment variables:
    OPENPI_API_KEY   Optional API key for the policy server.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Import from local package - add parent to path if needed
try:
    from openpi_compare import adapters
    from openpi_compare import plotting
except ModuleNotFoundError:
    # Fallback: add tools to path
    _tools_dir = Path(__file__).parent
    if str(_tools_dir.parent) not in sys.path:
        sys.path.insert(0, str(_tools_dir.parent))
    from openpi_compare import adapters
    from openpi_compare import plotting

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_IMAGE_SIZE = 224
DEFAULT_FPS = 25
DEFAULT_ENSEMBLE_ALPHA = 0.1


def _parse_key_value_items(items: list[str] | None) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI options."""
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Expected non-empty KEY=VALUE, got {item!r}")
        result[key] = value
    return result


def _parse_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [part.strip() for part in value.split(",") if part.strip()]


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenPI policy server predictions against LeRobot dataset ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset / environment
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to LeRobot dataset directory.",
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=None,
        help="Path to openpi source root (for optional imports).",
    )
    parser.add_argument(
        "--camera-map",
        action="append",
        default=None,
        metavar="ROLE=DATASET_KEY",
        help=(
            "Explicit camera mapping. Can be repeated. Roles are head, left_wrist, "
            "right_wrist, or camera_head/camera_left_wrist/camera_right_wrist."
        ),
    )
    parser.add_argument(
        "--required-server-cameras",
        type=str,
        default="camera_head,camera_left_wrist,camera_right_wrist",
        help=(
            "Comma-separated camera keys to always send to the server. Missing keys "
            "are zero-filled when --strict-observation is not set."
        ),
    )

    # Server connection
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Policy server host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Policy server port.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for the policy server.",
    )

    # Sample selection
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index to extract from dataset.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Start time in seconds (optional, for partial episode extraction).",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="End time in seconds (optional, for partial episode extraction).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (None = all).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for sampling (1 = every frame).",
    )
    parser.add_argument(
        "--use-original-image-size",
        action="store_true",
        default=False,
        help=(
            "Send images at their original resolution instead of resizing to 224x224. "
            "Useful for testing with different image sizes."
        ),
    )

    # Prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Language instruction prompt. If not provided, attempts to read from dataset.",
    )

    # Comparison settings
    parser.add_argument(
        "--ensemble",
        type=str,
        default="exp",
        choices=["exp", "mean", "median"],
        help="Ensemble aggregation method.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ENSEMBLE_ALPHA,
        help="Decay parameter for exponential ensemble weighting.",
    )
    parser.add_argument(
        "--action-semantics",
        type=str,
        default="absolute",
        choices=["auto", "absolute", "delta"],
        help=(
            "Action semantics. Use 'absolute' for normal policy outputs and 'delta' only "
            "when the server returns deltas. 'auto' is kept as a legacy alias for absolute."
        ),
    )
    parser.add_argument(
        "--action-labels",
        type=str,
        default=None,
        help="Comma-separated action dimension labels. Defaults to dataset feature names.",
    )
    parser.add_argument(
        "--strict-observation",
        action="store_true",
        help="Fail on missing observation fields instead of using placeholders where possible.",
    )

    # Output
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("./openpi_compare_results"),
        help="Directory to save results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: only load data and build observations, don't call server.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip generating preview images.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


class LeRobotDatasetLoader:
    """Loader for LeRobot-format datasets.

    Supports:
    - Video-based datasets (video files per camera)
    - Image-based datasets (frame images per camera)
    - Parquet data files (state, action, metadata)
    """

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)
        self._validate_dataset()
        self._fps = self._detect_fps()
        self._info = self._load_info()
        self._features = self._info.get("features", {})
        self._total_episodes = self._info.get("total_episodes", 0)
        self._total_frames = self._info.get("total_frames", 0)

        # Try to load tasks
        self._tasks: dict[int, str] = {}
        tasks_path = self.dataset_root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            try:
                import polars as pl
                df = pl.read_parquet(tasks_path)
                for row in df.iter_rows(named=True):
                    idx = row.get("index", row.get("task_index", 0))
                    task = row.get("task", row.get("instruction", row.get("__index_level_0__", "")))
                    if task:
                        self._tasks[int(idx)] = str(task)
            except ImportError:
                logger.warning("polars not available, skipping task loading")

        logger.info(
            "Dataset: %d episodes, %d total frames, fps=%.1f",
            self._total_episodes,
            self._total_frames,
            self._fps,
        )
        self._episodes_df = None

    def _validate_dataset(self) -> None:
        required = ["meta/info.json", "data"]
        for req in required:
            path = self.dataset_root / req
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset validation failed: {path} not found. "
                    f"Ensure this is a valid LeRobot dataset directory."
                )

    def _detect_fps(self) -> float:
        info = self._load_info()
        return float(info.get("fps", DEFAULT_FPS))

    def _load_info(self) -> dict[str, Any]:
        info_path = self.dataset_root / "meta" / "info.json"
        with open(info_path) as f:
            return json.load(f)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def features(self) -> dict[str, Any]:
        return self._features

    @property
    def action_labels(self) -> list[str] | None:
        names = self._features.get("action", {}).get("names")
        if isinstance(names, list) and names:
            return [str(name) for name in names]
        return None

    @property
    def action_dim(self) -> int | None:
        shape = self._features.get("action", {}).get("shape")
        if isinstance(shape, list) and shape:
            return int(shape[0])
        labels = self.action_labels
        return len(labels) if labels else None

    @property
    def episode_indices(self) -> list[int]:
        return list(range(self._total_episodes))

    def get_episode_frame_range(self, episode_index: int) -> tuple[int, int]:
        """Get the (start, end) frame indices for an episode.

        Returns:
            (start_frame, end_frame) inclusive.
        """
        import polars as pl

        df = self._load_episodes_df()
        episode_df = df.filter(pl.col("episode_index") == episode_index)

        if episode_df.is_empty():
            raise ValueError(f"Episode {episode_index} not found in episode index.")

        if "dataset_from_index" in episode_df.columns and "dataset_to_index" in episode_df.columns:
            return int(episode_df["dataset_from_index"][0]), int(episode_df["dataset_to_index"][0]) - 1

        # LeRobot stores 'length' as number of frames in episode.
        length = int(episode_df["length"][0])
        all_lengths = df.sort("episode_index")["length"].to_list()
        start = sum(all_lengths[:episode_index])
        end = start + length - 1
        return start, end

    def _load_episodes_df(self):
        """Load and cache episode metadata."""
        if self._episodes_df is not None:
            return self._episodes_df
        import polars as pl

        episode_path = self.dataset_root / "meta" / "episodes"
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode metadata not found at {episode_path}")

        parquet_files = sorted(episode_path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {episode_path}")

        self._episodes_df = pl.concat([pl.read_parquet(p) for p in parquet_files])
        return self._episodes_df

    def get_episode_video_path(self, episode_index: int, cam_key: str) -> Path | None:
        """Return the video file for a camera/episode when LeRobot metadata provides it."""
        import polars as pl

        df = self._load_episodes_df()
        episode_df = df.filter(pl.col("episode_index") == episode_index)
        if episode_df.is_empty():
            return None

        chunk_col = f"videos/{cam_key}/chunk_index"
        file_col = f"videos/{cam_key}/file_index"
        if chunk_col not in episode_df.columns or file_col not in episode_df.columns:
            return None

        chunk_idx = int(episode_df[chunk_col][0])
        file_idx = int(episode_df[file_col][0])
        candidate = self.dataset_root / "videos" / cam_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        return candidate if candidate.exists() else None

    def get_episode_data(
        self,
        episode_index: int,
        start_sec: float | None = None,
        end_sec: float | None = None,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load all frames from an episode.

        Args:
            episode_index: Episode to load
            start_sec: Optional start time (seconds)
            end_sec: Optional end time (seconds)
            stride: Frame stride
            max_frames: Maximum frames to load

        Returns:
            List of sample dicts, one per frame
        """
        import polars as pl

        # Load parquet data
        data_dir = self.dataset_root / "data"
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        logger.info("Loading parquet files from %s...", data_dir)
        df = pl.concat([pl.read_parquet(p) for p in parquet_files])

        # Filter to episode
        episode_df = df.filter(pl.col("episode_index") == episode_index)
        if episode_df.is_empty():
            raise ValueError(
                f"No data found for episode {episode_index}. "
                f"Available episodes: {df['episode_index'].unique().to_list()}"
            )

        episode_df = episode_df.sort("frame_index")

        # Apply time filtering
        if start_sec is not None:
            start_frame = int(start_sec * self._fps)
            episode_df = episode_df.filter(pl.col("frame_index") >= start_frame)

        if end_sec is not None:
            end_frame = int(end_sec * self._fps)
            episode_df = episode_df.filter(pl.col("frame_index") <= end_frame)

        # Apply stride
        if stride > 1:
            episode_df = episode_df.with_row_index(name="_orig_idx")
            episode_df = episode_df.filter(pl.col("_orig_idx") % stride == 0)

        # Apply max_frames
        if max_frames is not None:
            episode_df = episode_df.head(max_frames)

        # Get task for this episode
        task = self._tasks.get(episode_index, "")

        # Convert to list of dicts with proper numpy arrays
        samples = []
        for row in episode_df.iter_rows(named=True):
            sample: dict[str, Any] = {}

            # Extract scalar metadata
            for key in ["frame_index", "episode_index", "timestamp", "index", "task_index"]:
                if key in row:
                    val = row[key]
                    if val is not None:
                        if isinstance(val, (int, float)):
                            sample[key] = val
                        elif hasattr(val, "__float__"):
                            sample[key] = float(val)

            # Extract action
            for key in ["action", "actions", "observation.action"]:
                if key in row and row[key] is not None:
                    sample["action"] = np.asarray(row[key], dtype=np.float32)
                    break

            # Extract state
            state_keys = ["observation.state", "state", "observation_state", "observation/state"]
            for key in state_keys:
                if key in row:
                    val = row[key]
                    if val is not None:
                        if isinstance(val, list):
                            sample[key] = np.array(val, dtype=np.float32)
                        else:
                            sample[key] = np.asarray(val, dtype=np.float32)
                    break
            else:
                # Try to find state as observation.state from the row
                for k, v in row.items():
                    if "state" in k.lower() and "observation" in k.lower():
                        sample["observation.state"] = np.asarray(v, dtype=np.float32)
                        break

            # Extract task
            if "task" in row and isinstance(row["task"], str):
                sample["task"] = row["task"]
            elif task:
                sample["task"] = task

            samples.append(sample)

        logger.info(
            "Loaded %d frames from episode %d (stride=%d)",
            len(samples),
            episode_index,
            stride,
        )
        return samples


# --------------------------------------------------------------------------- #
# Dataset with video image loading
# --------------------------------------------------------------------------- #


class LeRobotDatasetWithImages(LeRobotDatasetLoader):
    """Extended loader that also provides access to raw images from parquet/video."""

    def __init__(self, dataset_root: Path) -> None:
        super().__init__(dataset_root)
        self._camera_keys = self._detect_camera_keys()
        self._video_capture_cache: dict[Path, Any] = {}
        logger.info("Detected camera keys: %s", list(self._camera_keys.keys()))

    def _detect_camera_keys(self) -> dict[str, str]:
        """Detect available camera keys from dataset features."""
        features = self._features
        cameras: dict[str, str] = {}

        for key in features:
            if "image" in key.lower() or "camera" in key.lower():
                dtype = features[key].get("dtype", "")
                if dtype == "video":
                    cameras[key] = "video"
                elif dtype == "image":
                    cameras[key] = "image"

        videos_dir = self.dataset_root / "videos"
        if videos_dir.exists():
            for camera_dir in sorted(path for path in videos_dir.iterdir() if path.is_dir()):
                if any(camera_dir.rglob("*.mp4")):
                    cameras.setdefault(camera_dir.name, "video")

        return cameras

    def load_samples_with_images(
        self,
        episode_index: int,
        start_sec: float | None = None,
        end_sec: float | None = None,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load samples with raw images embedded.

        For video-based datasets, this extracts individual frames from videos.
        For efficiency, only extracts frames that will actually be used.
        """
        import polars as pl

        data_dir = self.dataset_root / "data"
        video_dir = self.dataset_root / "videos"
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {data_dir}")

        df = pl.concat([pl.read_parquet(p) for p in parquet_files])
        episode_df = df.filter(pl.col("episode_index") == episode_index)
        episode_df = episode_df.sort("frame_index")

        if start_sec is not None:
            start_frame = int(start_sec * self._fps)
            episode_df = episode_df.filter(pl.col("frame_index") >= start_frame)

        if end_sec is not None:
            end_frame = int(end_sec * self._fps)
            episode_df = episode_df.filter(pl.col("frame_index") <= end_frame)

        if stride > 1:
            episode_df = episode_df.with_row_index(name="_orig_idx")
            episode_df = episode_df.filter(pl.col("_orig_idx") % stride == 0)

        if max_frames is not None:
            episode_df = episode_df.head(max_frames)

        task = self._tasks.get(episode_index, "")

        samples = []
        frame_indices = episode_df["frame_index"].to_list()

        # Try to load images from parquet directly (often stored as lists)
        for row in episode_df.iter_rows(named=True):
            sample: dict[str, Any] = {}

            for key in ["frame_index", "episode_index", "timestamp", "index", "task_index"]:
                if key in row and row[key] is not None:
                    sample[key] = row[key]

            # Load action
            for key in ["action", "actions", "observation.action"]:
                if key in row and row[key] is not None:
                    sample["action"] = np.asarray(row[key], dtype=np.float32)
                    break

            # Load state
            for key in ["observation.state", "state", "observation_state", "observation/state"]:
                if key in row and row[key] is not None:
                    sample["observation.state"] = np.asarray(row[key], dtype=np.float32)
                    break

            # Load images from parquet columns
            for cam_key in self._camera_keys:
                if cam_key in row and row[cam_key] is not None:
                    raw = row[cam_key]
                    if isinstance(raw, (list, np.ndarray)):
                        img = np.array(raw)
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            if img.min() >= 0 and img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                        if img.ndim == 3 and img.shape[0] == 3:
                            img = np.transpose(img, (1, 2, 0))
                        sample[cam_key] = img

            # Task
            if task:
                sample["task"] = task

            samples.append(sample)

        # If images weren't in parquet, try video loading
        if samples and not any(cam in samples[0] for cam in self._camera_keys) and video_dir.exists():
            try:
                for cam_key in self._camera_keys:
                    logger.info(
                        "Loading %d video frames for camera %s...",
                        len(frame_indices),
                        cam_key,
                    )
                    frames_by_index = self._load_frames_from_video(
                        video_dir, cam_key, frame_indices, episode_index
                    )
                    for i, frame_idx in enumerate(frame_indices):
                        if i >= len(samples) or cam_key in samples[i]:
                            continue
                        samples[i][cam_key] = frames_by_index.get(frame_idx)
            except ImportError:
                raise RuntimeError("cv2 not available, cannot load images from videos. Please install opencv-python.")

        logger.info("Loaded %d samples with images", len(samples))
        return samples

    def _get_video_candidates(self, video_dir: Path, cam_key: str, episode_index: int) -> tuple[list[Path], str]:
        video_path = video_dir / cam_key
        if not video_path.exists():
            logger.warning("Video path does not exist: %s", video_path)
            return [], ""

        metadata_video = self.get_episode_video_path(episode_index, cam_key)
        video_files = sorted(video_path.rglob("*.mp4"))
        if not video_files:
            logger.warning("No video files found in %s", video_path)
            return [], ""

        def _episode_number(path: Path) -> int | None:
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            return int(digits) if digits else None

        matching = [vf for vf in video_files if _episode_number(vf) == episode_index]
        candidate_files = []
        if metadata_video is not None:
            candidate_files.append(metadata_video)
        candidate_files.extend(vf for vf in matching if vf not in candidate_files)
        candidate_files.extend(vf for vf in video_files if vf not in candidate_files)
        codec = str(self._features.get(cam_key, {}).get("info", {}).get("video.codec", "")).lower()
        return candidate_files, codec

    def _load_frames_from_video(
        self,
        video_dir: Path,
        cam_key: str,
        frame_indices: list[int],
        episode_index: int,
    ) -> dict[int, np.ndarray]:
        """Load many frames from one camera video in a single sequential decode pass."""
        candidate_files, codec = self._get_video_candidates(video_dir, cam_key, episode_index)
        if not candidate_files:
            return {}

        wanted = sorted(set(int(idx) for idx in frame_indices))
        if not wanted:
            return {}

        for vf in candidate_files:
            try:
                frames = self._load_frames_with_pyav(vf, wanted)
                if len(frames) == len(wanted):
                    logger.info("Loaded %d/%d frames for %s via PyAV", len(frames), len(wanted), cam_key)
                    return frames
                if frames:
                    logger.warning(
                        "Loaded %d/%d frames for %s via PyAV; missing frames will use per-frame fallback",
                        len(frames),
                        len(wanted),
                        cam_key,
                    )
                    for missing in wanted:
                        if missing not in frames:
                            frames[missing] = self._load_frame_from_video(video_dir, cam_key, missing, episode_index)
                    return frames
            except ImportError:
                logger.debug("PyAV not available, falling back to per-frame video loading")
                break
            except Exception as e:
                logger.debug("PyAV failed for %s: %s", vf, e)

        logger.warning(
            "Falling back to per-frame video loading for %s. This can be slow for long AV1 episodes.",
            cam_key,
        )
        return {
            frame_idx: self._load_frame_from_video(video_dir, cam_key, frame_idx, episode_index)
            for frame_idx in wanted
        }

    def _load_frames_with_pyav(self, video_file: Path, wanted: list[int]) -> dict[int, np.ndarray]:
        """Decode selected frames sequentially with PyAV."""
        import av

        wanted_set = set(wanted)
        max_wanted = max(wanted)
        frames: dict[int, np.ndarray] = {}

        with av.open(str(video_file)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for decoded_idx, frame in enumerate(container.decode(stream)):
                frame_idx = decoded_idx
                if frame_idx in wanted_set:
                    frames[frame_idx] = frame.to_ndarray(format="rgb24")
                    if len(frames) == len(wanted_set):
                        break
                if frame_idx > max_wanted:
                    break

        return frames

    def _load_frame_from_video(
        self, video_dir: Path, cam_key: str, frame_idx: int, episode_index: int
    ) -> np.ndarray | None:
        """Load a specific frame from a video file using multiple backends."""
        import subprocess

        candidate_files, codec = self._get_video_candidates(video_dir, cam_key, episode_index)
        if not candidate_files:
            return None
        prefer_ffmpeg = codec == "av1"

        # Try the matching episode video first, then fall back to any file containing this frame.
        for vf in candidate_files:
            # Method 1: OpenCV is much faster than spawning ffmpeg for every frame.
            if not prefer_ffmpeg:
                try:
                    import cv2
                    cap = self._video_capture_cache.get(vf)
                    if cap is None:
                        cap = cv2.VideoCapture(str(vf))
                        self._video_capture_cache[vf] = cap
                    if cap.isOpened():
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_idx < total_frames:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                logger.debug(f"Loaded frame {frame_idx} from {vf.name} via cv2")
                                return frame
                except Exception as e:
                    logger.debug(f"cv2 failed for {vf}: {e}")

            # Method 2: ffmpeg fallback for codecs OpenCV cannot decode.
            try:
                import cv2
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
                if codec == "av1":
                    cmd.extend(["-c:v", "libdav1d"])
                cmd.extend([
                    "-i", str(vf),
                    "-vf", f"select=eq(n\\,{frame_idx})",
                    "-vframes", "1",
                    "-f", "image2pipe",
                    "-vcodec", "png",
                    "-",
                ])
                result = subprocess.run(cmd, capture_output=True, timeout=10, check=False)
                if result.returncode == 0 and result.stdout:
                    nparr = np.frombuffer(result.stdout, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        logger.debug(f"Loaded frame {frame_idx} from {vf.name} via ffmpeg")
                        return img
                if result.stderr:
                    logger.debug("ffmpeg failed for %s: %s", vf, result.stderr[:200])
            except FileNotFoundError:
                logger.debug("ffmpeg not available for video fallback")
            except Exception as e:
                logger.debug(f"ffmpeg failed for {vf}: {e}")

        # Video loading failed - raise an error with details
        raise RuntimeError(
            f"Failed to load frame {frame_idx} from camera {cam_key}. "
            f"Tried {len(candidate_files)} video file(s). "
            f"Video files found: {[str(vf) for vf in candidate_files]}"
        )

    def close(self) -> None:
        for cap in self._video_capture_cache.values():
            try:
                cap.release()
            except Exception:
                pass
        self._video_capture_cache.clear()


# --------------------------------------------------------------------------- #
# Server client (minimal reimplementation to avoid openpi-client dep)
# --------------------------------------------------------------------------- #


class PolicyServerClient:
    """Simple WebSocket client for the OpenPI policy server."""

    def __init__(
        self,
        host: str,
        port: int,
        api_key: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.api_key = api_key
        self._connect()

    def _connect(self) -> None:
        import websockets.sync.client
        import msgpack
        import numpy as np

        # Custom packer that handles numpy arrays properly (same as pi0_web_policy.py)
        def pack_array(obj):
            if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {obj.dtype}")

            if isinstance(obj, np.ndarray):
                return {
                    b"__ndarray__": True,
                    b"data": obj.tobytes(),
                    b"dtype": obj.dtype.str,
                    b"shape": obj.shape,
                }

            if isinstance(obj, np.generic):
                return {
                    b"__npgeneric__": True,
                    b"data": obj.item(),
                    b"dtype": obj.dtype.str,
                }

            return obj

        def unpack_array(obj):
            if b"__ndarray__" in obj:
                return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
            if b"__npgeneric__" in obj:
                return np.dtype(obj[b"dtype"]).type(obj[b"data"])
            return obj

        uri = f"ws://{self.host}:{self.port}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"

        logger.info("Connecting to policy server at %s...", uri)
        self._ws = websockets.sync.client.connect(
            uri, compression=None, max_size=None, additional_headers=headers
        )

        # Custom pack/unpack using same format as pi0_web_policy.py
        # Create Packer instance directly (not partial)
        self._packer = msgpack.Packer(default=pack_array)
        self._unpack_func = lambda data: msgpack.unpackb(data, object_hook=unpack_array)

        # Receive metadata (server uses same packer)
        metadata_bytes = self._ws.recv()
        if isinstance(metadata_bytes, str):
            raise RuntimeError(f"Server error: {metadata_bytes}")
        self._metadata = self._unpack_func(metadata_bytes)
        logger.info("Connected. Server metadata: %s", self._metadata)

    def get_metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Send observation and get action prediction."""
        import msgpack

        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server inference error:\n{response}")
        return self._unpack_func(response)

    def close(self) -> None:
        self._ws.close()


# --------------------------------------------------------------------------- #
# Sample integrity checks
# --------------------------------------------------------------------------- #


def check_sample_integrity(
    sample: dict[str, Any],
    expected_action_dim: int | None = None,
) -> list[str]:
    """Run integrity checks on a dataset sample.

    Returns list of warnings.
    """
    warnings: list[str] = []

    # Check action
    if "action" not in sample:
        warnings.append("Sample missing 'action' key")
    else:
        action = sample["action"]
        if not isinstance(action, np.ndarray):
            warnings.append(f"action is {type(action)}, expected numpy array")
        else:
            action_array = np.asarray(action).squeeze()
            if action_array.ndim == 0:
                warnings.append("action is scalar, expected vector")
            else:
                action_dim = int(action_array.shape[-1])
                if expected_action_dim is not None and action_dim != expected_action_dim:
                    warnings.append(f"action dim is {action_dim}, dataset metadata says {expected_action_dim}")

    # Check state
    state_keys = ["observation.state", "state", "observation_state"]
    found_state = any(k in sample for k in state_keys)
    if not found_state:
        warnings.append("Sample missing state keys")

    # Check cameras
    found_cam = any("image" in k.lower() or "camera" in k.lower() or "cam_" in k.lower() for k in sample)
    if not found_cam:
        warnings.append("Sample has no recognizable camera images")

    return warnings


# --------------------------------------------------------------------------- #
# Main comparison logic
# --------------------------------------------------------------------------- #


def run_comparison(
    args: argparse.Namespace,
    client: PolicyServerClient | None,
    adapter: adapters.ObservationAdapter,
    dataset: LeRobotDatasetWithImages,
    samples: list[dict[str, Any]],
    ep_start: int,
    ep_end: int,
    action_labels: list[str] | None,
    save_dir: Path,
) -> plotting.ComparisonResult | None:
    """Run the full comparison pipeline."""

    # Check episode index
    if args.episode_index < 0 or args.episode_index >= dataset.total_episodes:
        raise ValueError(
            f"Episode index {args.episode_index} out of range. "
            f"Dataset has {dataset.total_episodes} episodes (0 to {dataset.total_episodes - 1})."
        )

    logger.info(
        "Episode %d spans frames %d to %d (fps=%.1f)",
        args.episode_index,
        ep_start,
        ep_end,
        dataset.fps,
    )

    if len(samples) == 0:
        raise RuntimeError(f"No samples loaded for episode {args.episode_index}.")

    # Integrity check on first sample
    integrity_warnings = check_sample_integrity(samples[0], dataset.action_dim)
    if integrity_warnings:
        logger.warning("Integrity warnings for first sample: %s", integrity_warnings)

    # Determine frame indices
    frame_indices = np.array(
        [int(s.get("frame_index", i)) for i, s in enumerate(samples)]
    )
    n_frames = len(samples)

    # Extract ground truth actions
    gt_actions_list = []
    inferred_gt_dim = dataset.action_dim
    for s in samples:
        if "action" in s:
            action = s["action"]
            if action.ndim > 1:
                action = action.squeeze()
            inferred_gt_dim = int(action.shape[-1])
            gt_actions_list.append(action.astype(np.float32))
        else:
            if inferred_gt_dim is None:
                state = s.get("observation.state", s.get("state"))
                inferred_gt_dim = int(np.asarray(state).squeeze().shape[-1]) if state is not None else 1
            gt_actions_list.append(np.zeros(inferred_gt_dim, dtype=np.float32))

    gt_actions = np.stack(gt_actions_list)  # [T, D]

    logger.info(
        "GT actions shape: %s, range: [%.4f, %.4f]",
        gt_actions.shape,
        gt_actions.min(),
        gt_actions.max(),
    )

    # Dry run - just return sample info
    if args.dry_run:
        obs = adapter.adapt(samples[0])
        obs_warnings = adapters.validate_observation(
            obs,
            expected_camera_keys=adapter.required_server_cameras,
        )
        if obs_warnings:
            logger.warning("[DRY RUN] Observation warnings: %s", obs_warnings)

        obs_summary: dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_summary[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "valid": obs.get("_camera_valid", {}).get(key),
                }
            elif key == "_camera_valid":
                obs_summary[key] = value
            else:
                obs_summary[key] = value
        with open(save_dir / "dry_run_observation.json", "w") as f:
            json.dump(obs_summary, f, indent=2, default=str)
        logger.info(
            "[DRY RUN] Would process %d frames from episode %d. "
            "Sample keys: %s. Observation keys: %s",
            n_frames,
            args.episode_index,
            list(samples[0].keys()),
            list(obs.keys()),
        )
        return None

    # Connect to server
    if client is None:
        client = PolicyServerClient(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )

    # Infer per frame
    pred_chunks_list: list[np.ndarray] = []
    failed_frames: list[int] = []
    raw_responses: list[dict[str, Any]] = []

    logger.info("Starting inference on %d frames...", n_frames)

    for i, sample in enumerate(samples):
        try:
            obs = adapter.adapt(sample)
            response = client.infer(obs)
            raw_responses.append(response)

            action = response.get("action")
            if action is None:
                raise KeyError("Response missing 'action' key")

            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 1:
                action = action[np.newaxis, :]  # [1, D]

            pred_chunks_list.append(action)

            if (i + 1) % 20 == 0:
                logger.info("  Processed %d/%d frames", i + 1, n_frames)

        except Exception as e:
            logger.error("Frame %d inference failed: %s", i, e)
            failed_frames.append(i)
            raw_responses.append({"error": str(e)})
            # Use zeros as placeholder
            pred_chunks_list.append(np.zeros((1, gt_actions.shape[1]), dtype=np.float32))

    if failed_frames:
        logger.warning("%d frames failed: %s", len(failed_frames), failed_frames)

    # Stack predictions
    pred_chunks = np.stack(pred_chunks_list)  # [T, H, D]
    # Infer horizon from first prediction
    horizon = pred_chunks.shape[1] if pred_chunks.ndim == 3 else 1
    if pred_chunks.ndim == 2:
        pred_chunks = pred_chunks[:, np.newaxis, :]  # [T, 1, D]

    action_dim = pred_chunks.shape[2]
    if action_dim != gt_actions.shape[1]:
        logger.warning(
            "Action dim mismatch: pred=%d vs gt=%d. "
            "Using min=%d for comparison.",
            action_dim,
            gt_actions.shape[1],
            min(action_dim, gt_actions.shape[1]),
        )
        min_dim = min(action_dim, gt_actions.shape[1])
        pred_chunks = pred_chunks[:, :, :min_dim]
        gt_actions = gt_actions[:, :min_dim]
        action_dim = min_dim

    logger.info(
        "Pred chunks shape: %s, GT actions shape: %s",
        pred_chunks.shape,
        gt_actions.shape,
    )

    # Determine action semantics
    action_semantics = args.action_semantics
    if action_semantics == "auto":
        action_semantics = "absolute"
        logger.info("--action-semantics auto is a legacy alias for absolute")
    else:
        logger.info("Action semantics: %s", action_semantics)

    # Optionally convert delta to absolute. Because this diagnostic has access
    # to GT actions, reject a requested delta conversion if it clearly makes the
    # horizon-0 comparison worse; that usually means the server already returned
    # absolute actions.
    if action_semantics == "delta":
        logger.info("Converting delta actions to absolute using state trajectory...")
        raw_h0_mae = float(np.mean(np.abs(pred_chunks[:, 0, :] - gt_actions)))
        states_for_delta = np.zeros((n_frames, action_dim), dtype=np.float32)
        for i, s in enumerate(samples):
            state = s.get("observation.state", s.get("state"))
            if state is not None:
                state = np.asarray(state).squeeze().astype(np.float32)
                if len(state) >= action_dim:
                    states_for_delta[i] = state[:action_dim]
                else:
                    states_for_delta[i, :len(state)] = state

        # Accumulate deltas: first frame is absolute, rest are relative
        pred_abs = np.zeros_like(pred_chunks)
        pred_abs[:, 0, :] = pred_chunks[:, 0, :] + states_for_delta
        for k in range(1, horizon):
            pred_abs[:, k, :] = pred_abs[:, k - 1, :] + pred_chunks[:, k, :]

        converted_h0_mae = float(np.mean(np.abs(pred_abs[:, 0, :] - gt_actions)))
        if converted_h0_mae > max(raw_h0_mae * 2.0, raw_h0_mae + 1e-6):
            logger.warning(
                "Delta conversion made horizon-0 MAE worse (raw=%.4f, converted=%.4f). "
                "Keeping raw predictions as absolute. Use --verbose to inspect saved run_config.",
                raw_h0_mae,
                converted_h0_mae,
            )
            action_semantics = "absolute_delta_rejected_as_worse_than_raw"
        else:
            logger.info(
                "Delta conversion complete. Shape: %s, h0 MAE raw=%.4f -> converted=%.4f",
                pred_abs.shape,
                raw_h0_mae,
                converted_h0_mae,
            )
            pred_chunks = pred_abs

    # Run comparison
    logger.info("Running comparison...")
    labels = plotting.get_action_labels(gt_actions.shape[1], action_labels)

    result = plotting.run_comparison(
        gt_actions=gt_actions,
        pred_chunks=pred_chunks,
        frame_indices=frame_indices,
        ensemble_method=args.ensemble,
        alpha=args.alpha,
    )

    # Save data
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    np.savez_compressed(
        save_dir / "results.npz",
        frame_indices=frame_indices,
        gt_actions=gt_actions,
        pred_chunks=pred_chunks,
        pred_h0=result.pred_h0,
        pred_ensemble=result.pred_ensemble,
        valid_mask=result.valid_mask,
    )
    logger.info("Saved results.npz")

    # Save metrics
    plotting.save_metrics_json(result, labels, save_dir / "metrics.json")

    # Save raw responses
    raw_responses_clean: dict[str, Any] = {
        "n_frames": n_frames,
        "failed_frames": failed_frames,
        "metadata": {},
    }
    # Don't save full responses as JSON (too large), just record what we can
    raw_responses_clean["n_responses"] = len(raw_responses)
    raw_responses_clean["has_timing"] = any("server_timing" in response for response in raw_responses)
    with open(save_dir / "server_responses_meta.json", "w") as f:
        json.dump(raw_responses_clean, f, indent=2)

    # Save run config
    run_config = dict(vars(args))
    run_config.pop("api_key", None)
    run_config["save_dir"] = str(save_dir)
    run_config["episode_frame_range"] = [int(ep_start), int(ep_end)]
    run_config["dataset_fps"] = float(dataset.fps)
    run_config["action_semantics_detected"] = action_semantics
    run_config["action_labels"] = labels
    run_config["available_cameras"] = list(dataset._camera_keys.keys())
    run_config["server_metadata"] = client.get_metadata() if client else {}
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    return result


# --------------------------------------------------------------------------- #
# Generate plots
# --------------------------------------------------------------------------- #


def generate_plots(
    result: plotting.ComparisonResult,
    samples: list[dict[str, Any]],
    adapter: adapters.ObservationAdapter,
    args: argparse.Namespace,
    save_dir: Path,
    action_labels: list[str] | None = None,
) -> None:
    """Generate all visualization plots."""
    labels = plotting.get_action_labels(result.action_dim, action_labels)

    # Horizon-0 overlay per dimension
    for d in range(result.action_dim):
        label = labels[d] if d < len(labels) else f"dim_{d}"
        plotting.plot_action_overlay(
            gt=result.gt_actions[:, d : d + 1],
            pred=result.pred_h0[:, d : d + 1],
            title=f"Horizon-0 | {label}",
            labels=[label],
            filepath=save_dir / f"horizon0_overlay_dim{d:02d}.png",
            frame_indices=result.frame_indices,
            time_unit="frame",
        )

    # Ensemble overlay per dimension - use first T frames where we have GT
    T = result.pred_chunks.shape[0]
    valid_gt = result.gt_actions[:T]
    valid_ens = result.pred_ensemble[:T]
    valid_x = result.frame_indices[:T]

    for d in range(result.action_dim):
        label = labels[d] if d < len(labels) else f"dim_{d}"
        plotting.plot_action_overlay(
            gt=valid_gt[:, d : d + 1],
            pred=valid_ens[:, d : d + 1],
            title=f"Ensemble | {label}",
            labels=[label],
            filepath=save_dir / f"ensemble_overlay_dim{d:02d}.png",
            frame_indices=valid_x,
            time_unit="frame",
        )

    # Offset curves
    plotting.plot_offset_curves(
        offset_metrics=result.offset_metrics,
        horizon=result.horizon,
        title="Horizon Offset Analysis",
        filepath=save_dir / "offset_mae_curve.png",
    )

    # Error heatmap
    plotting.plot_error_heatmap(
        gt=result.gt_actions,
        pred=result.pred_h0,
        title=f"Error Heatmap (Horizon-0) | Episode {args.episode_index}",
        labels=labels,
        filepath=save_dir / "error_heatmap.png",
        frame_indices=result.frame_indices,
    )

    # Summary plot
    plotting.plot_summary(
        result=result,
        labels=labels,
        save_dir=save_dir,
        episode=args.episode_index,
        frame_range=(int(result.frame_indices[0]), int(result.frame_indices[-1])),
    )

    # Sample preview (first/middle/last frames)
    if not args.skip_preview and len(samples) > 0:
        plotting.generate_sample_preview(
            frames=samples,
            adapter=adapter,
            save_path=save_dir / "sample_preview.png",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add openpi to path if provided
    if args.openpi_root is not None:
        openpi_path = str(args.openpi_root)
        if openpi_path not in sys.path:
            sys.path.insert(0, openpi_path)
            logger.info("Added %s to Python path", openpi_path)

    # Check dataset
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_root}")

    # Check server connection info
    logger.info("=" * 60)
    logger.info("OpenPI Policy Comparison")
    logger.info("=" * 60)
    logger.info("Dataset: %s", args.dataset_root)
    logger.info("Episode: %d", args.episode_index)
    logger.info("Server: %s:%d", args.host, args.port)
    logger.info("Camera map: %s", args.camera_map or "(auto)")
    logger.info("Required server cameras: %s", args.required_server_cameras)
    logger.info("Image size: %s", "original" if args.use_original_image_size else f"{DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}")
    logger.info("Prompt: %s", args.prompt or "(from dataset)")
    logger.info("Ensemble: %s (alpha=%.2f)", args.ensemble, args.alpha)
    logger.info("Stride: %d, Max frames: %s", args.stride, args.max_frames)
    logger.info("Dry run: %s", args.dry_run)
    logger.info("Save dir: %s", args.save_dir)
    logger.info("=" * 60)

    # Create adapter
    camera_map = _parse_key_value_items(args.camera_map)
    required_server_cameras = tuple(_parse_csv(args.required_server_cameras) or [])
    adapter = adapters.create_adapter(
        image_size=DEFAULT_IMAGE_SIZE,
        use_original_image_size=args.use_original_image_size,
        prompt=args.prompt,
        allow_fallback_prompt=True,
        strict=args.strict_observation,
        camera_map=camera_map,
        required_server_cameras=required_server_cameras,
    )

    # Create save directory
    save_dir = args.save_dir / f"episode_{args.episode_index:04d}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Start client (only if not dry run)
    client: PolicyServerClient | None = None
    if not args.dry_run:
        api_key = args.api_key or os.environ.get("OPENPI_API_KEY")
        try:
            client = PolicyServerClient(
                host=args.host,
                port=args.port,
                api_key=api_key,
            )
        except Exception as e:
            logger.error("Failed to connect to server: %s", e)
            logger.error("Use --dry-run to skip server connection and just process data.")
            sys.exit(1)

    # Run comparison
    try:
        # Load samples for preview
        dataset = LeRobotDatasetWithImages(args.dataset_root)
        ep_start, ep_end = dataset.get_episode_frame_range(args.episode_index)
        samples = dataset.load_samples_with_images(
            episode_index=args.episode_index,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            stride=args.stride,
            max_frames=args.max_frames,
        )

        # Generate metadata
        action_labels = _parse_csv(args.action_labels) or dataset.action_labels
        metadata = {
            "episode_index": args.episode_index,
            "frame_range": [int(ep_start), int(ep_end)],
            "fps": float(dataset.fps),
            "n_samples": len(samples),
            "prompt": args.prompt or (samples[0].get("task") if samples else None),
            "use_original_image_size": args.use_original_image_size,
            "image_size": "original" if args.use_original_image_size else DEFAULT_IMAGE_SIZE,
            "stride": args.stride,
            "action_dim": dataset.action_dim,
            "action_labels": action_labels,
            "available_cameras": list(dataset._camera_keys.keys()) if hasattr(dataset, "_camera_keys") else [],
            "camera_map": camera_map,
            "required_server_cameras": list(required_server_cameras),
            "action_keys": ["action"],
            "state_keys": ["observation.state"],
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata.json")

        # Generate preview if not dry run
        if not args.dry_run and not args.skip_preview:
            plotting.generate_sample_preview(
                frames=samples[:: max(1, len(samples) // 10)][:9],
                adapter=adapter,
                save_path=save_dir / "sample_preview.png",
            )

        # Run the actual comparison
        result = run_comparison(
            args=args,
            client=client,
            adapter=adapter,
            dataset=dataset,
            samples=samples,
            ep_start=ep_start,
            ep_end=ep_end,
            action_labels=action_labels,
            save_dir=save_dir,
        )

        if result is not None:
            # Generate plots
            generate_plots(result, samples, adapter, args, save_dir, action_labels)

            # Print summary
            logger.info("=" * 60)
            logger.info("COMPARISON RESULTS")
            logger.info("=" * 60)
            logger.info("Episode %d | Frames %d-%d | %d frames",
                       args.episode_index, int(ep_start), int(ep_end), len(samples))
            logger.info("Horizon: %d | Action dim: %d", result.horizon, result.action_dim)
            logger.info("")
            logger.info("Horizon-0 Comparison:")
            logger.info("  Overall MAE:  %.4f", result.horizon0_overall_mae)
            logger.info("  Overall RMSE: %.4f", result.horizon0_overall_rmse)
            logger.info("  Per-dim MAE:  %s", np.array2string(result.horizon0_mae, precision=4))
            logger.info("")
            logger.info("Ensemble Comparison (method=%s, alpha=%.2f):",
                       result.ensemble_method, result.alpha)
            logger.info("  Overall MAE:  %.4f", result.ensemble_overall_mae)
            logger.info("  Overall RMSE: %.4f", result.ensemble_overall_rmse)
            logger.info("  Per-dim MAE:  %s", np.array2string(result.ensemble_mae, precision=4))
            logger.info("")
            logger.info("Results saved to: %s", save_dir)
            logger.info("=" * 60)

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if "dataset" in locals():
            try:
                dataset.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
