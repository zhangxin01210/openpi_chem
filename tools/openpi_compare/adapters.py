"""
Data adapters for converting LeRobot dataset samples to OpenPI policy server observations.

The adapter intentionally keeps dataset-specific assumptions small:
- camera keys are resolved from aliases or explicit user mappings
- missing wrist cameras can be zero-filled so server-side transforms still run
- images are normalized to uint8 HWC before being sent over WebSocket
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Image parsing utilities (mirrors openpi/src/openpi/policies/chem_policy.py)
# --------------------------------------------------------------------------- #


def parse_image(image) -> np.ndarray:
    """Parse a raw image from LeRobot dataset to uint8 HWC format.

    Handles:
    - Float32 [0,1] images (common in LeRobot) -> uint8 [0,255]
    - CHW layout -> HWC layout
    - Already uint8 HWC images (pass-through)
    - Invalid/non-array inputs -> raises descriptive error

    Args:
        image: Raw image array. May be float32 CHW, float32 HWC, uint8 CHW, or uint8 HWC.

    Returns:
        uint8 HWC numpy array ready for server.

    Raises:
        ValueError: If image has unexpected shape or dtype.
    """
    # Handle None or non-array inputs
    if image is None:
        raise ValueError("Image is None, cannot parse")
    
    # Convert to numpy array
    try:
        image = np.asarray(image)
    except Exception as e:
        raise ValueError(f"Failed to convert image to numpy array: {type(image).__name__}, error: {e}")
    
    # Handle empty arrays
    if image.size == 0:
        raise ValueError(f"Image array is empty, shape: {image.shape}")
    
    # Handle 0-dimensional arrays (scalars)
    if image.ndim == 0:
        raise ValueError(f"Image is a scalar (0-dim), not an image array: {image}")
    
    # Handle 1D arrays (flat)
    if image.ndim == 1:
        raise ValueError(f"Image is 1D (flat), shape: {image.shape}. Expected 2D or 3D image.")
    
    if np.issubdtype(image.dtype, np.floating):
        if image.min() < -0.5 or image.max() > 1.5:
            raise ValueError(
                f"Float image values appear to be outside [0,1] range: "
                f"min={image.min():.4f}, max={image.max():.4f}. "
                f"Expected [0,1] normalized float images."
            )
        image = (image * 255.0).clip(0, 255).astype(np.uint8)

    if image.dtype == np.uint8:
        pass
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")

    # CHW -> HWC
    if image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] != 3:
        raise ValueError(
            f"Image does not have 3 channels in last axis. Got shape: {image.shape}. "
            f"Expected HWC format."
        )

    return image


# --------------------------------------------------------------------------- #
# Image resizing utilities (wraps openpi_client.image_tools or PIL fallback)
# --------------------------------------------------------------------------- #

try:
    from openpi_client import image_tools as _image_tools

    def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize image to target height/width with padding, using openpi_client."""
        return _image_tools.resize_with_pad(image, height, width)

    def convert_to_uint8_image(image: np.ndarray) -> np.ndarray:
        """Convert image to uint8, using openpi_client."""
        return _image_tools.convert_to_uint8(image)

    _HAS_OPENPI_CLIENT = True

except ImportError:
    _HAS_OPENPI_CLIENT = False
    import PIL.Image

    def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
        """Fallback resize_with_pad using PIL."""
        pil_img = PIL.Image.fromarray(image)
        orig_w, orig_h = pil_img.size
        if orig_w == width and orig_h == height:
            return np.array(pil_img)
        ratio = max(orig_w / width, orig_h / height)
        new_w = int(orig_w / ratio)
        new_h = int(orig_h / ratio)
        resized = pil_img.resize((new_w, new_h), PIL.Image.BILINEAR)
        result = PIL.Image.new("RGB", (width, height), 0)
        pad_x = max(0, (width - new_w) // 2)
        pad_y = max(0, (height - new_h) // 2)
        result.paste(resized, (pad_x, pad_y))
        return np.array(result)

    def convert_to_uint8_image(image: np.ndarray) -> np.ndarray:
        """Fallback convert_to_uint8 using numpy."""
        if np.issubdtype(image.dtype, np.floating):
            return (image * 255).astype(np.uint8)
        return image.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Camera role definitions
# --------------------------------------------------------------------------- #


DEFAULT_SERVER_CAMERA_KEYS = ("camera_head", "camera_left_wrist", "camera_right_wrist")

CAMERA_ROLE_ALIASES: dict[str, list[str]] = {
    "head": [
        "observation.images.head",
        "observation/images/head",
        "observation.images.cam_front",
        "observation/images/cam_front",
        "images.head",
        "images.cam_front",
        "camera_head",
        "camera_front",
        "cam_front",
        "cam_high",
        "cam_top",
        "front",
        "head",
    ],
    "left_wrist": [
        "observation.images.left_wrist",
        "observation/images/left_wrist",
        "observation.images.cam_left",
        "observation/images/cam_left",
        "images.left_wrist",
        "images.cam_left",
        "camera_left_wrist",
        "camera_left",
        "cam_left_wrist",
        "cam_left",
        "left_wrist",
    ],
    "right_wrist": [
        "observation.images.right_wrist",
        "observation/images/right_wrist",
        "observation.images.cam_right",
        "observation/images/cam_right",
        "images.right_wrist",
        "images.cam_right",
        "camera_right_wrist",
        "camera_right",
        "cam_right_wrist",
        "cam_right",
        "right_wrist",
    ],
}

# --------------------------------------------------------------------------- #
# Base adapter class
# --------------------------------------------------------------------------- #


class ObservationAdapter:
    """Convert LeRobot samples to OpenPI server observations."""

    def __init__(
        self,
        image_size: int = 224,
        use_original_image_size: bool = False,
        prompt: str | None = None,
        allow_fallback_prompt: bool = True,
        strict: bool = True,
        camera_map: dict[str, str] | None = None,
        required_server_cameras: tuple[str, ...] = DEFAULT_SERVER_CAMERA_KEYS,
    ) -> None:
        """Initialize the adapter.

        Args:
            image_size: Target image size (square). Images are resized with padding.
                Only used when use_original_image_size is False.
            use_original_image_size: If True, send images at their original resolution
                without resizing. If False, resize to image_size x image_size.
            prompt: Explicit prompt to use. If None, attempts to read from data.
            allow_fallback_prompt: If True and prompt is missing, use dataset task.
            strict: If True, raise error on missing expected keys. If False, try to
                recover or use zeros.
            camera_map: Optional explicit mapping from role (head/left_wrist/right_wrist)
                or server camera key to a dataset camera key.
            required_server_cameras: Server camera keys that should always be present
                in the outgoing observation. Missing cameras are zero-filled when
                strict is False.
        """
        self.image_size = image_size
        self.use_original_image_size = use_original_image_size
        self.prompt = prompt
        self.allow_fallback_prompt = allow_fallback_prompt
        self.strict = strict
        self.camera_map = camera_map or {}
        self.required_server_cameras = required_server_cameras

    def _find_camera_key(self, sample: dict[str, Any], candidates: list[str]) -> str | None:
        """Find the first matching camera key in a sample."""
        for key in candidates:
            if key in sample:
                return key
        return None

    def _infer_camera_key_by_name(self, sample: dict[str, Any], role: str) -> str | None:
        """Infer a camera key from loose naming conventions."""
        role_tokens = {
            "head": ("head", "front", "base", "top", "high", "third"),
            "left_wrist": ("left_wrist", "wrist_left", "cam_left", "left"),
            "right_wrist": ("right_wrist", "wrist_right", "cam_right", "right"),
        }[role]
        image_like = [
            key for key, value in sample.items()
            if ("image" in key.lower() or "camera" in key.lower() or "cam_" in key.lower())
            and isinstance(value, np.ndarray)
        ]
        for key in image_like:
            normalized = key.lower().replace("/", ".")
            if any(token in normalized for token in role_tokens):
                return key
        return None

    def _detect_available_cameras(self, sample: dict[str, Any]) -> dict[str, str]:
        """Detect which cameras are available in the sample."""
        detected: dict[str, str] = {}

        for role, aliases in CAMERA_ROLE_ALIASES.items():
            explicit = self.camera_map.get(role) or self.camera_map.get(f"camera_{role}")
            if explicit:
                detected[role] = explicit
                continue
            key = self._find_camera_key(sample, aliases)
            if key is None:
                key = self._infer_camera_key_by_name(sample, role)
            if key:
                detected[role] = key

        return detected

    def _load_and_process_image(
        self, sample: dict[str, Any], key: str | None, mask_label: str, template: np.ndarray | None = None
    ) -> tuple[np.ndarray, bool]:
        """Load and process an image from the sample.

        Returns:
            Tuple of (processed_image, is_valid). is_valid is True if image was
            successfully loaded, False if it was zero-filled (masked).
        """
        if key is None or key not in sample:
            if self.strict:
                raise KeyError(
                    f"Required camera key {key!r} not found in sample. "
                    f"Available keys: {list(sample.keys())}"
                )
            else:
                # Return a zero-filled placeholder
                logger.debug("Camera %s not found, using zero placeholder", key)
                if template is not None:
                    return np.zeros_like(template, dtype=np.uint8), False
                return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8), False

        raw = sample[key]

        # Handle None or invalid values
        if raw is None:
            raise ValueError(
                f"Camera {key!r} is None in sample. "
                f"Available keys: {list(sample.keys())}"
            )

        try:
            # Video types from LeRobot may need frame extraction
            if hasattr(raw, "__iter__") and not isinstance(raw, np.ndarray):
                # Handle iterable of frames (e.g., from video)
                if hasattr(raw, "__len__"):
                    try:
                        raw = np.array(raw)
                    except Exception:
                        pass

            if isinstance(raw, np.ndarray):
                if raw.ndim == 4:
                    # Video: take first frame [T, H, W, C] or [T, C, H, W]
                    frame_idx = 0
                    if raw.shape[0] < 10:
                        frame_idx = 0
                    raw = raw[frame_idx]
                    if raw.shape[0] == 3:
                        raw = np.transpose(raw, (1, 2, 0))

                image = parse_image(raw)
            else:
                raise TypeError(
                    f"Unexpected image type for key {key!r}: {type(raw).__name__}. "
                    f"Expected numpy array."
                )

            # Resize to target size (unless use_original_image_size is True)
            if self.use_original_image_size:
                # Keep original resolution, just convert to uint8
                image = convert_to_uint8_image(image)
            else:
                # Resize to target size
                image = resize_with_pad(image, self.image_size, self.image_size)
                image = convert_to_uint8_image(image)

            return image, True

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse image for {key}: {e}. Using zero placeholder.")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8), False

    def _extract_state(self, sample: dict[str, Any]) -> np.ndarray:
        """Extract robot state from sample."""
        state_key = "observation.state"
        state_keys = [
            "observation.state",
            "state",
            "observation/state",
            "observation.joint_position",
            "joint_position",
        ]
        found_key = None
        for k in state_keys:
            if k in sample:
                found_key = k
                break

        if found_key is None:
            raise KeyError(
                f"State key not found in sample. Searched: {state_keys}. "
                f"Available: {list(sample.keys())}"
            )

        state = np.asarray(sample[found_key])
        if state.ndim > 1:
            state = state.squeeze()
        return state.astype(np.float32)

    def _extract_action(self, sample: dict[str, Any]) -> np.ndarray | None:
        """Extract ground-truth action from sample."""
        action_keys = ["action", "actions", "observation.action"]
        for k in action_keys:
            if k in sample:
                action = np.asarray(sample[k])
                if action.ndim > 1:
                    action = action.squeeze()
                return action.astype(np.float32)
        return None

    def _extract_prompt(self, sample: dict[str, Any]) -> str | None:
        """Extract prompt from sample or use the instance-level prompt."""
        if self.prompt is not None:
            return self.prompt

        prompt_keys = ["task", "prompt", "observation.task", "language_instruction"]
        for k in prompt_keys:
            if k in sample:
                val = sample[k]
                if isinstance(val, str):
                    return val
                if isinstance(val, (np.ndarray, tuple, list)) and len(val) > 0:
                    return str(val[0]) if hasattr(val, "__len__") else str(val)

        if self.allow_fallback_prompt:
            logger.warning(
                "No prompt found in sample and none provided. "
                "Consider passing --prompt explicitly. Using empty string."
            )
            return ""
        return None

    def _extract_timestamp(self, sample: dict[str, Any]) -> float | None:
        """Extract timestamp from sample."""
        for key in ["timestamp", "observation.timestamp", "time"]:
            if key in sample:
                val = np.asarray(sample[key]).flatten()
                return float(val[0]) if len(val) > 0 else None
        return None

    def _extract_frame_index(self, sample: dict[str, Any]) -> int | None:
        """Extract frame index from sample."""
        for key in ["frame_index", "index"]:
            if key in sample:
                val = np.asarray(sample[key]).flatten()
                return int(val[0]) if len(val) > 0 else None
        return None

    def adapt(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert a LeRobot dataset sample to OpenPI server observation format.

        Args:
            sample: A single frame from LeRobot dataset. Expected keys include:
                - observation.images.head / camera_head (camera images)
                - observation.images.right_wrist / camera_right_wrist
                - observation.state
                - action (optional, for comparison)
                - task / prompt (optional, for language instruction)

        Returns:
            Observation dict matching the common ChemPolicy inference schema.

        Raises:
            KeyError: If required keys are missing in strict mode.
        """
        detected_cameras = self._detect_available_cameras(sample)
        image_keys = [
            key for key, value in sample.items()
            if ("image" in key.lower() or "camera" in key.lower() or "cam_" in key.lower())
            and isinstance(value, np.ndarray)
        ]
        unused_image_keys = [key for key in image_keys if key not in set(detected_cameras.values())]
        for role in ("head", "left_wrist", "right_wrist"):
            if role not in detected_cameras and unused_image_keys:
                detected_cameras[role] = unused_image_keys.pop(0)

        processed_by_source: dict[str, tuple[np.ndarray, bool]] = {}
        for key in image_keys:
            processed_by_source[key] = self._load_and_process_image(sample, key, key)

        def get_processed(key: str | None, label: str, template: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
            if key is not None and key in processed_by_source:
                return processed_by_source[key]
            return self._load_and_process_image(sample, key, label, template=template)

        camera_head, head_valid = get_processed(detected_cameras.get("head"), "base")
        camera_left_wrist, left_valid = get_processed(
            detected_cameras.get("left_wrist"), "left_wrist", template=camera_head
        )
        camera_right_wrist, right_valid = get_processed(
            detected_cameras.get("right_wrist"), "right_wrist", template=camera_head
        )

        # Extract state
        state = self._extract_state(sample)

        obs: dict[str, Any] = {"state": state}
        for source_key, (image, _valid) in processed_by_source.items():
            obs[source_key] = image
        validity = {
            "camera_head": head_valid,
            "camera_left_wrist": left_valid,
            "camera_right_wrist": right_valid,
        }
        for source_key, (_image, valid) in processed_by_source.items():
            validity[source_key] = valid

        standard_alias_images = {
            "camera_head": camera_head,
            "camera_left_wrist": camera_left_wrist,
            "camera_right_wrist": camera_right_wrist,
        }
        for key, image in standard_alias_images.items():
            if key in self.required_server_cameras:
                obs[key] = image
        for key in self.required_server_cameras:
            if key not in obs:
                obs[key] = np.zeros_like(camera_head, dtype=np.uint8)
                validity[key] = False
        obs["_camera_valid"] = validity

        # Extract prompt
        prompt = self._extract_prompt(sample)
        if prompt is not None:
            obs["prompt"] = prompt

        return obs

# --------------------------------------------------------------------------- #
# Adapter factory
# --------------------------------------------------------------------------- #


def create_adapter(
    image_size: int = 224,
    use_original_image_size: bool = False,
    prompt: str | None = None,
    allow_fallback_prompt: bool = True,
    strict: bool = True,
    camera_map: dict[str, str] | None = None,
    required_server_cameras: tuple[str, ...] = DEFAULT_SERVER_CAMERA_KEYS,
) -> ObservationAdapter:
    """Create the default observation adapter.

    Args:
        image_size: Target image size.
        use_original_image_size: If True, send images at original resolution.
        prompt: Explicit prompt.
        allow_fallback_prompt: Allow using dataset task as prompt.
        strict: Strict key checking.

    Returns:
        An ObservationAdapter instance.
    """
    return ObservationAdapter(
        image_size=image_size,
        use_original_image_size=use_original_image_size,
        prompt=prompt,
        allow_fallback_prompt=allow_fallback_prompt,
        strict=strict,
        camera_map=camera_map,
        required_server_cameras=required_server_cameras,
    )


# --------------------------------------------------------------------------- #
# Observation structure validation
# --------------------------------------------------------------------------- #


def validate_observation(
    obs: dict[str, Any],
    expected_camera_keys: tuple[str, ...] = DEFAULT_SERVER_CAMERA_KEYS,
) -> list[str]:
    """Validate an observation dict and return list of warnings.

    Args:
        obs: Observation dict from ObservationAdapter.adapt().
        expected_camera_keys: Camera keys expected by the target policy server.

    Returns:
        List of warning messages (empty if all checks pass).
    """
    warnings: list[str] = []

    for key in expected_camera_keys:
        if key not in obs:
            warnings.append(f"Missing expected camera key: {key}")
        elif not isinstance(obs[key], np.ndarray):
            warnings.append(f"{key} is not a numpy array: {type(obs[key])}")
        elif obs[key].dtype != np.uint8:
            warnings.append(f"{key} dtype is {obs[key].dtype}, expected uint8")
        elif obs[key].shape[-1] != 3:
            warnings.append(f"{key} shape is {obs[key].shape}, expected HWC with 3 channels")

    if "state" not in obs:
        warnings.append("Missing state")
    elif not isinstance(obs["state"], np.ndarray):
        warnings.append(f"state is not numpy array: {type(obs['state'])}")
    elif obs["state"].ndim != 1:
        warnings.append(f"state shape is {obs['state'].shape}, expected 1D")

    # Check prompt
    if "prompt" not in obs:
        warnings.append("Missing prompt")

    return warnings
