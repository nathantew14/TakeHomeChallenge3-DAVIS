"""Classical CV operations for the segmentation pipeline.

Each operation is a pure function (ndarray -> ndarray) plus a parameter
spec that the web UI turns into sliders.  bridge.py runs it in the browser;
the same functions can be called directly to reproduce a result.

Every stage image is uint8, either (h, w) or (h, w, 3) BGR -- OpenCV's
convention.  scikit-image wants RGB, so conversions happen at that boundary.
"""

import cv2
import numpy as np
from skimage.segmentation import find_boundaries, slic

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _odd(n):
    """Kernel sizes must be odd and >= 1."""
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# --------------------------------------------------------------------------
# pre-processing
# --------------------------------------------------------------------------

def downscale(image, width=854):
    """INTER_AREA resize.  Fewer pixels means a faster SLIC and less texture
    for it to chase.  Note any mask built downstream must be resized back to
    the source size (nearest-neighbour) before it can be scored."""
    width = int(width)
    if width >= image.shape[1]:
        return image
    h = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)


def grayscale(image):
    return _gray(image)


def gaussian_kernel_for(sigma):
    """Kernel wide enough to hold the useful part of a Gaussian of `sigma`.

    The curve is effectively zero past ~3 sigma, so 6 sigma covers both tails;
    forced odd because the kernel needs a centre pixel.
    """
    return _odd(int(round(6.0 * float(sigma))))


def gaussian_blur(image, kernel_mode=0, kernel_size=5, sigma=1.2):
    """kernel_mode 1 sizes the kernel from sigma instead of the slider.

    Worth preferring: a kernel narrower than the sigma asks for truncates the
    Gaussian, and OpenCV renormalises what is left, so you silently get less
    blur than the sigma implies.
    """
    k = gaussian_kernel_for(sigma) if int(kernel_mode) == 1 else _odd(kernel_size)
    return cv2.GaussianBlur(image, (k, k), float(sigma))


def box_blur(image, kernel_size=5):
    k = _odd(kernel_size)
    return cv2.blur(image, (k, k))


def median_blur(image, kernel_size=5):
    """Replaces each pixel with the median of its neighbourhood.  Removes
    speckle and fine texture without softening the edges around it."""
    return cv2.medianBlur(image, _odd(kernel_size))


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """Edge-preserving smoothing: averages only over neighbours that are both
    near in space and similar in colour.  Flattens fur/grass texture while
    keeping object outlines sharp -- a good fit ahead of SLIC."""
    return cv2.bilateralFilter(image, max(1, int(diameter)),
                               float(sigma_color), float(sigma_space))


def histogram_equalise(image, mode=1, clip_limit=2.0, tile_size=8):
    """Redistribute intensities to use the full range.

    mode 0 = global equalizeHist, mode 1 = CLAHE (local, contrast-limited).
    Colour input is equalised on the L channel in Lab so hues are unchanged;
    equalising B, G and R separately would shift the colours.
    """
    def eq(channel):
        if int(mode) == 0:
            return cv2.equalizeHist(channel)
        clahe = cv2.createCLAHE(
            clipLimit=max(0.01, float(clip_limit)),
            tileGridSize=(max(1, int(tile_size)), max(1, int(tile_size))),
        )
        return clahe.apply(channel)

    if image.ndim == 2:
        return eq(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = eq(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen(image, amount=1.0, kernel_size=3):
    """Unsharp mask.  Weights are (1 + amount) and (-amount) so they sum to 1
    and overall brightness is preserved."""
    k = _odd(kernel_size)
    amount = float(amount)
    return cv2.addWeighted(image, 1.0 + amount, cv2.blur(image, (k, k)), -amount, 0)


# --------------------------------------------------------------------------
# SLIC superpixels
# --------------------------------------------------------------------------

# One-entry memo of the last label map.  The describer needs the superpixel
# count, and flipping the display mode should not re-segment -- both hit this
# instead of running SLIC again.  It also leaves the labels reachable for a
# later superpixel-selection step.
_slic_memo = {"image": None, "key": None, "labels": None}


def slic_labels(image, n_segments=200, compactness=10.0, smoothing=0, sigma=1.0,
                colour_space=0, max_iter=10, enforce_connectivity=1):
    """Run SLIC and return the int label map (h, w), labels starting at 0.

    Colour input is converted BGR -> RGB here; skimage then converts to Lab
    itself when colour_space is 0.  skimage min-max rescales every input to
    0..1 *before* that conversion, so Lab distances run on a 0..100 scale but
    RGB and grayscale ones on 0..1.  Compactness is divided by 100 on those
    paths so one slider value means roughly the same balance in every mode.

    Deterministic: seeds are placed on a regular grid, so there is no random
    seed to record.
    """
    key = (image.shape, int(n_segments), float(compactness), int(smoothing),
           float(sigma), int(colour_space), int(max_iter), int(enforce_connectivity))
    if _slic_memo["image"] is image and _slic_memo["key"] == key:
        return _slic_memo["labels"]

    if image.ndim == 3:
        data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        channel_axis, lab = -1, int(colour_space) == 0
    else:
        data, channel_axis, lab = image, None, False
    scale = 1.0 if lab else 0.01

    labels = slic(
        data,
        n_segments=max(1, int(n_segments)),
        compactness=max(1e-5, float(compactness) * scale),
        sigma=float(sigma) if int(smoothing) == 1 else 0.0,
        max_num_iter=max(1, int(max_iter)),
        convert2lab=lab,
        enforce_connectivity=bool(int(enforce_connectivity)),
        channel_axis=channel_axis,
        start_label=0,
    )
    _slic_memo.update(image=image, key=key, labels=labels)
    return labels


def mean_colour(image, labels):
    """Paint every superpixel with its average colour."""
    n = int(labels.max()) + 1
    flat = labels.ravel()
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    chans = image.reshape(-1, 1) if image.ndim == 2 else image.reshape(-1, image.shape[2])
    means = np.stack([np.bincount(flat, weights=chans[:, c], minlength=n) / counts
                      for c in range(chans.shape[1])], axis=1)
    out = np.round(means[flat]).astype(np.uint8)
    return out.reshape(image.shape)


_BOUNDARY_BGR = (0, 230, 255)   # yellow: readable on fur, rock, water and grass


def draw_boundaries(image, labels, colour=_BOUNDARY_BGR):
    """Overlay one-pixel superpixel borders."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    out[find_boundaries(labels, mode="inner")] = colour
    return out


def slic_superpixels(image, n_segments=200, compactness=10.0, smoothing=0, sigma=1.0,
                     colour_space=0, max_iter=10, enforce_connectivity=1, display=0):
    """SLIC superpixels, rendered for viewing.

    display 0 = borders over the input, 1 = mean colour per superpixel,
    2 = mean colour with borders.
    """
    labels = slic_labels(image, n_segments, compactness, smoothing, sigma,
                         colour_space, max_iter, enforce_connectivity)
    display = int(display)
    if display == 0:
        return draw_boundaries(image, labels)
    flat = mean_colour(image, labels)
    return flat if display == 1 else draw_boundaries(flat, labels)


def _describe_slic(p, image=None):
    """Report what SLIC actually produced: the count rarely matches the ask,
    since seeds sit on a grid and connectivity enforcement merges orphans."""
    if image is None:
        return None
    labels = slic_labels(image, p["n_segments"], p["compactness"], p["smoothing"],
                         p["sigma"], p["colour_space"], p["max_iter"],
                         p["enforce_connectivity"])
    n = int(labels.max()) + 1
    h, w = image.shape[:2]
    step = (h * w / max(1, int(p["n_segments"]))) ** 0.5
    return "%d superpixels (asked %d)  ·  grid step ≈ %.0f px" % (n, int(p["n_segments"]), step)


def _describe_gaussian(p, image=None):
    """Report the resolved kernel in auto mode, and flag a truncating one in
    manual mode -- the failure there is silent otherwise."""
    sigma = float(p["sigma"])
    if int(p["kernel_mode"]) == 1:
        k = gaussian_kernel_for(sigma)
        return "auto  ·  %d×%d px" % (k, k)
    k, want = _odd(p["kernel_size"]), gaussian_kernel_for(sigma)
    if sigma > 0 and k < want:
        return "truncated  ·  6 sigma wants %d×%d px" % (want, want)
    return None


# --------------------------------------------------------------------------
# registry -- drives the UI
# --------------------------------------------------------------------------
# param spec: (key, label, min, max, step, default)
# a "choices" entry turns the control into a segmented button group instead.

OPERATIONS = {
    "slic": {
        "label": "SLIC Superpixels",
        "hint": "k-means in (colour, x, y). Compactness trades colour adherence (low) for regular, grid-like cells (high).",
        "fn": slic_superpixels,
        "describe": _describe_slic,
        "params": [
            {"key": "n_segments", "label": "Superpixels", "min": 10, "max": 2000, "step": 10, "default": 200},
            {"key": "compactness", "label": "Compactness", "min": 0.5, "max": 60, "step": 0.5, "default": 10.0},
            {"key": "smoothing", "label": "Smoothing", "choices": ["Off", "Gaussian"], "default": 0},
            {"key": "sigma", "label": "Sigma", "min": 0.1, "max": 10, "step": 0.1, "default": 1.0, "when": {"smoothing": 1}},
            {"key": "colour_space", "label": "Feature space", "choices": ["Lab", "RGB"], "default": 0},
            {"key": "max_iter", "label": "Iterations", "min": 1, "max": 30, "step": 1, "default": 10},
            {"key": "enforce_connectivity", "label": "Connectivity", "choices": ["Off", "Enforce"], "default": 1},
            {"key": "display", "label": "Show", "choices": ["Borders", "Mean colour", "Both"], "default": 0},
        ],
    },
    "downscale": {
        "label": "Downscale",
        "hint": "INTER_AREA. Faster SLIC and less texture; masks must be resized back to 854×480.",
        "fn": downscale,
        "params": [
            {"key": "width", "label": "Target width", "min": 100, "max": 854, "step": 2, "default": 427, "unit": "px"},
        ],
    },
    "grayscale": {
        "label": "Grayscale",
        "hint": "BGR to single channel. Throws away the colour SLIC mostly relies on.",
        "fn": grayscale,
        "params": [],
    },
    "gaussian_blur": {
        "label": "Gaussian Blur",
        "hint": "Isotropic low-pass. Suppresses fine texture, but also softens outlines.",
        "fn": gaussian_blur,
        "describe": _describe_gaussian,
        "params": [
            {"key": "kernel_mode", "label": "Kernel", "choices": ["Manual", "6 × sigma"], "default": 0},
            {"key": "kernel_size", "label": "Kernel size", "min": 1, "max": 61, "step": 2, "default": 5, "unit": "px", "when": {"kernel_mode": 0}},
            {"key": "sigma", "label": "Sigma", "min": 0, "max": 10, "step": 0.1, "default": 1.2},
        ],
    },
    "box_blur": {
        "label": "Box Blur",
        "hint": "Uniform average. Cheaper than Gaussian, blockier result.",
        "fn": box_blur,
        "params": [
            {"key": "kernel_size", "label": "Kernel size", "min": 1, "max": 21, "step": 2, "default": 5, "unit": "px"},
        ],
    },
    "median_blur": {
        "label": "Median Filter",
        "hint": "Kills speckle without softening edges.",
        "fn": median_blur,
        "params": [
            {"key": "kernel_size", "label": "Kernel size", "min": 1, "max": 31, "step": 2, "default": 5, "unit": "px"},
        ],
    },
    "bilateral_filter": {
        "label": "Bilateral Filter",
        "hint": "Edge-preserving smoothing. Flattens texture inside regions, keeps outlines.",
        "fn": bilateral_filter,
        "params": [
            {"key": "diameter", "label": "Diameter", "min": 1, "max": 25, "step": 1, "default": 9, "unit": "px"},
            {"key": "sigma_color", "label": "Sigma colour", "min": 1, "max": 200, "step": 1, "default": 75},
            {"key": "sigma_space", "label": "Sigma space", "min": 1, "max": 200, "step": 1, "default": 75},
        ],
    },
    "histogram_equalise": {
        "label": "Histogram Equalisation",
        "hint": "Stretches contrast on L so similar-looking regions separate further.",
        "fn": histogram_equalise,
        "describe": lambda p, image=None: (
            "global" if int(p["mode"]) == 0
            else "CLAHE  ·  %dx%d tiles" % (int(p["tile_size"]), int(p["tile_size"]))),
        "params": [
            {"key": "mode", "label": "Mode", "choices": ["Global", "CLAHE"], "default": 1},
            {"key": "clip_limit", "label": "Clip limit", "min": 0.5, "max": 12, "step": 0.1, "default": 2.0, "when": {"mode": 1}},
            {"key": "tile_size", "label": "Tile grid", "min": 1, "max": 32, "step": 1, "default": 8, "when": {"mode": 1}},
        ],
    },
    "sharpen": {
        "label": "Sharpen",
        "hint": "Unsharp mask. Boosts edges but also amplifies texture.",
        "fn": sharpen,
        "params": [
            {"key": "amount", "label": "Amount", "min": 0, "max": 4, "step": 0.05, "default": 1.0},
            {"key": "kernel_size", "label": "Kernel size", "min": 1, "max": 31, "step": 2, "default": 3, "unit": "px"},
        ],
    },
}


DEFAULT_PIPELINE = [
    {"type": "gaussian_blur", "params": {"kernel_mode": 1, "kernel_size": 5, "sigma": 1.0}, "enabled": False},
    {"type": "slic", "params": {"n_segments": 200, "compactness": 10.0, "smoothing": 0, "sigma": 1.0,
                                "colour_space": 0, "max_iter": 10, "enforce_connectivity": 1,
                                "display": 0}, "enabled": True},
]


def defaults_for(op_type):
    """Default param dict for a newly added block."""
    spec = OPERATIONS[op_type]
    return {p["key"]: p["default"] for p in spec["params"]}


def merged_params(op_type, params):
    """Defaults overlaid with whatever the client sent, unknown keys dropped."""
    spec = OPERATIONS[op_type]
    allowed = {p["key"] for p in spec["params"]}
    merged = defaults_for(op_type)
    merged.update({k: v for k, v in (params or {}).items() if k in allowed})
    return merged


def describe_op(op_type, params, image=None):
    """Short readout of values the sliders imply but do not show directly.

    `image` is the op's *input*, needed by describers whose readout depends on
    the data -- SLIC's actual superpixel count cannot be known from the sliders.
    """
    fn = OPERATIONS[op_type].get("describe")
    if fn is None:
        return None
    try:
        return fn(merged_params(op_type, params), image)
    except Exception:
        return None


def apply_op(image, op_type, params):
    return OPERATIONS[op_type]["fn"](image, **merged_params(op_type, params))


def run_pipeline(image, ops):
    """Run the pipeline, returning [{label, image, ...}, ...] including the source.

    A failing block is skipped rather than killing the whole run, so a bad
    slider value degrades to 'no effect' instead of a blank screen.
    """
    stages = [{"label": "Source", "type": "source", "image": image, "error": None, "detail": None}]
    current = image
    for op in ops:
        if not op.get("enabled", True):
            continue
        op_type = op.get("type")
        if op_type not in OPERATIONS:
            continue
        error = None
        before = current                      # the describer sees the input
        try:
            current = apply_op(current, op_type, op.get("params"))
        except Exception as exc:  # keep the UI responsive on bad params
            error = str(exc)
        stages.append({
            "label": OPERATIONS[op_type]["label"],
            "type": op_type,
            "image": current,
            "error": error,
            "detail": describe_op(op_type, op.get("params"), before),
        })
    return stages
