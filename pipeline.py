"""Classical CV operations for the segmentation pipeline.

Each operation is a function plus a parameter spec that the web UI turns
into sliders.  bridge.py runs it in the browser; run_pipeline() reproduces a
result anywhere else.

What flows between blocks is a *state* dict, not a bare image:

    image     what this stage shows -- uint8, (h, w) or (h, w, 3) BGR
    labels    SLIC label map (h, w), or None before SLIC
    base      the image SLIC segmented; region features are measured on
              this, never on the rendered overlay
    features  (n_regions, d) weighted feature matrix, or None
    feature_names, feature_groups   one entry per column

Most blocks are plain image filters (ndarray -> ndarray); apply_op lifts them
onto the state.  Blocks marked "state": True take and return the whole dict.
BGR is OpenCV's convention; scikit-image wants RGB, so conversions happen at
that boundary.
"""

import cv2
import numpy as np
from scipy.ndimage import find_objects
from skimage.feature import local_binary_pattern
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

# One-entry memo of the last label map, so flipping the display mode does
# not re-segment.
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


def _region_sums(flat, values, n):
    """Per-region column sums of `values` (n_pixels, k) via bincount."""
    return np.stack([np.bincount(flat, weights=values[:, c], minlength=n)
                     for c in range(values.shape[1])], axis=1)


def _region_means(labels, values):
    """Per-region means of a (h, w) or (h, w, k) array -> (n_regions, k)."""
    flat = labels.ravel()
    n = int(labels.max()) + 1
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    v = values.reshape(flat.size, -1).astype(np.float64)
    return _region_sums(flat, v, n) / counts[:, None]


def mean_colour(image, labels):
    """Paint every superpixel with its average colour."""
    means = _region_means(labels, image)
    return np.round(means[labels]).astype(np.uint8).reshape(image.shape)


_BOUNDARY_BGR = (0, 230, 255)   # yellow: readable on fur, rock, water and grass


def draw_boundaries(image, labels, colour=_BOUNDARY_BGR):
    """Overlay one-pixel superpixel borders."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    out[find_boundaries(labels, mode="inner")] = colour
    return out


def slic_superpixels(state, n_segments=200, compactness=10.0, smoothing=0, sigma=1.0,
                     colour_space=0, max_iter=10, enforce_connectivity=1, display=0):
    """SLIC superpixels.  Stores the label map and the image it was computed
    on, and renders for viewing: display 0 = borders over the input,
    1 = mean colour per superpixel, 2 = mean colour with borders."""
    image = state["image"]
    labels = slic_labels(image, n_segments, compactness, smoothing, sigma,
                         colour_space, max_iter, enforce_connectivity)
    display = int(display)
    if display == 0:
        shown = draw_boundaries(image, labels)
    else:
        flat = mean_colour(image, labels)
        shown = flat if display == 1 else draw_boundaries(flat, labels)
    return new_state(shown, labels=labels, base=image)


def _describe_slic(p, before, after):
    """Report what SLIC actually produced: the count rarely matches the ask,
    since seeds sit on a grid and connectivity enforcement merges orphans."""
    if after.get("labels") is None:
        return None
    n = int(after["labels"].max()) + 1
    h, w = after["labels"].shape
    step = (h * w / max(1, int(p["n_segments"]))) ** 0.5
    return "%d superpixels (asked %d)  ·  grid step ≈ %.0f px" % (n, int(p["n_segments"]), step)


# --------------------------------------------------------------------------
# region features
# --------------------------------------------------------------------------
# Feature menu after Hoiem, Efros & Hebert, "Recovering Surface Layout from
# an Image" (IJCV 2007) -- colour, texture and location/shape cues per
# superpixel -- with LBP (Ojala et al., 2002) as the cheap texture default
# and a small Gabor bank as the filter-response alternative Hoiem uses.

def _gray_of(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _bgr_of(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def _f_lab_mean(image, labels):
    lab = cv2.cvtColor(_bgr_of(image), cv2.COLOR_BGR2LAB)
    return _region_means(labels, lab), ["L mean", "a mean", "b mean"]


def _f_lab_std(image, labels):
    lab = cv2.cvtColor(_bgr_of(image), cv2.COLOR_BGR2LAB).astype(np.float64)
    m = _region_means(labels, lab)
    m2 = _region_means(labels, lab * lab)
    return np.sqrt(np.maximum(m2 - m * m, 0)), ["L std", "a std", "b std"]


def _f_hsv_mean(image, labels):
    """Hue is an angle, so it is averaged as a unit vector: 359° and 1° are
    neighbours, and a plain mean would put them at 180°.  The vector is
    scaled by saturation, since hue means little on a grey pixel."""
    hsv = cv2.cvtColor(_bgr_of(image), cv2.COLOR_BGR2HSV).astype(np.float64)
    ang = hsv[..., 0] * (np.pi / 90.0)          # OpenCV hue is 0..179
    sat = hsv[..., 1] / 255.0
    stack = np.dstack([sat * np.cos(ang), sat * np.sin(ang), sat, hsv[..., 2] / 255.0])
    return _region_means(labels, stack), ["hue cos", "hue sin", "saturation", "value"]


def _f_centroid(image, labels):
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xy = np.dstack([xx / max(1, w - 1), yy / max(1, h - 1)])
    return _region_means(labels, xy), ["x centroid", "y centroid"]


def _f_extent(image, labels):
    """10th and 90th percentiles of x and y: where the region spans, not just
    where its centre sits."""
    h, w = labels.shape
    flat = labels.ravel()
    n = int(labels.max()) + 1
    order = np.argsort(flat, kind="stable")
    xs = (order % w) / max(1, w - 1)
    ys = (order // w) / max(1, h - 1)
    ends = np.cumsum(np.bincount(flat, minlength=n))
    out = np.zeros((n, 4))
    start = 0
    for r in range(n):
        stop = ends[r]
        if stop > start:
            out[r, 0:2] = np.percentile(xs[start:stop], [10, 90])
            out[r, 2:4] = np.percentile(ys[start:stop], [10, 90])
        start = stop
    return out, ["x p10", "x p90", "y p10", "y p90"]


_LBP_P, _LBP_R = 8, 1


def _f_lbp(image, labels):
    """Uniform LBP histogram: P + 2 = 10 bins, each the fraction of the
    region's pixels with that micro-pattern (edge, corner, flat, ...).
    Invariant to monotonic brightness changes, so shading does not read as
    texture."""
    codes = local_binary_pattern(_gray_of(image), _LBP_P, _LBP_R, method="uniform")
    bins = _LBP_P + 2
    flat = labels.ravel()
    n = int(labels.max()) + 1
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    hist = np.bincount(flat * bins + codes.astype(np.int64).ravel(),
                       minlength=n * bins).reshape(n, bins)
    return hist / counts[:, None], ["LBP %d" % b for b in range(bins)]


_GABOR_WAVELENGTHS = (4.0, 8.0)                 # px
_GABOR_THETAS = (0.0, 45.0, 90.0, 135.0)        # degrees


def _f_gabor(image, labels):
    """Mean response magnitude of a 4-orientation x 2-scale Gabor bank.
    Bandwidth is fixed at ~1 octave (sigma = 0.56 wavelength)."""
    g = _gray_of(image).astype(np.float32) / 255.0
    maps, names = [], []
    for lam in _GABOR_WAVELENGTHS:
        sig = 0.56 * lam
        k = _odd(int(round(6 * sig)))
        for th in _GABOR_THETAS:
            kern = cv2.getGaborKernel((k, k), sig, np.deg2rad(th), lam, 1.0, 0, ktype=cv2.CV_32F)
            kern -= kern.mean()                   # zero DC: flat regions respond ~0
            maps.append(np.abs(cv2.filter2D(g, cv2.CV_32F, kern)))
            names.append("Gabor λ%d %d°" % (lam, th))
    return _region_means(labels, np.dstack(maps)), names


def _f_gradient(image, labels):
    g = _gray_of(image).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 255.0
    m = _region_means(labels, mag)
    m2 = _region_means(labels, mag * mag)
    return np.hstack([m, np.sqrt(np.maximum(m2 - m * m, 0))]), ["grad mean", "grad std"]


def _f_shape(image, labels):
    """log area (region sizes are heavy-tailed) and bounding-box aspect."""
    h, w = labels.shape
    n = int(labels.max()) + 1
    area = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    aspect = np.zeros(n)
    for r, sl in enumerate(find_objects(labels + 1, max_label=n)):
        if sl is not None:
            aspect[r] = np.log((sl[1].stop - sl[1].start) / (sl[0].stop - sl[0].start))
    return np.stack([np.log(np.maximum(area, 1) / (h * w)), aspect], axis=1), ["log area", "log aspect"]


# (key, label, group, fn) -- order is the bit order of the "features" mask
FEATURES = [
    ("lab_mean", "Lab mean", "colour", _f_lab_mean),
    ("lab_std", "Lab std", "colour", _f_lab_std),
    ("hsv_mean", "HSV mean", "colour", _f_hsv_mean),
    ("centroid", "Centroid", "position", _f_centroid),
    ("extent", "Extent", "position", _f_extent),
    ("lbp", "LBP", "texture", _f_lbp),
    ("gabor", "Gabor", "texture", _f_gabor),
    ("gradient", "Gradient", "texture", _f_gradient),
    ("shape", "Area / aspect", "shape", _f_shape),
]
FEATURE_GROUPS = ["colour", "position", "texture", "shape"]


def _group_bits(group):
    return sum(1 << i for i, f in enumerate(FEATURES) if f[2] == group)


def region_features(image, labels, features=0b101001, w_colour=1.0, w_position=1.0,
                    w_texture=1.0, w_shape=1.0):
    """Per-region feature matrix, standardised and weighted.

    `features` is a bitmask over FEATURES.  Each column is z-scored across
    regions, then every selected feature's block is divided by sqrt(its
    width), so each feature contributes the same total variance whether it is
    2 numbers or 10 -- otherwise the 10 LBP bins would outvote 3 Lab channels.
    Finally each block is multiplied by its group weight, so in a squared
    Euclidean distance a feature counts weight² times.

    Returns (X, names, groups) with X of shape (n_regions, d).
    """
    weights = {"colour": w_colour, "position": w_position,
               "texture": w_texture, "shape": w_shape}
    blocks, names, groups = [], [], []
    for i, (key, label, group, fn) in enumerate(FEATURES):
        if not int(features) >> i & 1:
            continue
        raw, cols = fn(image, labels)
        raw = np.asarray(raw, dtype=np.float64)
        std = raw.std(axis=0)
        z = (raw - raw.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)
        blocks.append(z * (float(weights[group]) / np.sqrt(raw.shape[1])))
        names += cols
        groups += [group] * len(cols)
    n = int(labels.max()) + 1
    X = np.hstack(blocks) if blocks else np.zeros((n, 0))
    return X, names, groups


def _stretch(v):
    """Map values to 0..255 with a 2-98 percentile stretch."""
    lo, hi = np.percentile(v, [2, 98]) if v.size else (0.0, 1.0)
    return np.clip((v - lo) / (hi - lo if hi > lo else 1.0) * 255.0, 0, 255)


def feature_pca(X, k=3):
    """Top-k principal components of X (n, d), zero-padded if d < k.  Signs
    are fixed so each component's largest loading is positive -- SVD's sign
    is arbitrary, and without this the colours could flip between runs."""
    n, d = X.shape
    out = np.zeros((n, k))
    if n < 2 or d == 0:
        return out, np.zeros(k)
    Xc = X - X.mean(axis=0)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    m = min(k, vt.shape[0])
    for c in range(m):
        if vt[c, np.argmax(np.abs(vt[c]))] < 0:
            vt[c] = -vt[c]
    out[:, :m] = Xc @ vt[:m].T
    var = s ** 2
    ratio = np.zeros(k)
    ratio[:m] = var[:m] / var.sum() if var.sum() > 0 else 0
    return out, ratio


_EDGE_BGR = (30, 30, 30)


def extract_features(state, features=0b101001, w_colour=1.0, w_position=1.0,
                     w_texture=1.0, w_shape=1.0, display=0, feature_index=0):
    """Region features, measured on the image SLIC segmented.

    display 0 paints each region by its first three principal components as
    RGB -- regions that look alike under the chosen features get similar
    colours.  display 1 paints one feature column as a heatmap.
    """
    labels = state.get("labels")
    if labels is None:
        raise ValueError("needs a SLIC block upstream")
    X, names, groups = region_features(state["base"], labels, features,
                                       w_colour, w_position, w_texture, w_shape)
    if X.shape[1] == 0:
        raise ValueError("select at least one feature")
    if int(display) == 0:
        pcs, _ = feature_pca(X, 3)
        rgb = np.stack([_stretch(pcs[:, c]) for c in range(3)], axis=1)
        region_bgr = rgb[:, ::-1].astype(np.uint8)
    else:
        col = X[:, int(feature_index) % X.shape[1]]
        grey = _stretch(col).astype(np.uint8).reshape(-1, 1)
        region_bgr = cv2.applyColorMap(grey, cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    shown = draw_boundaries(region_bgr[labels], labels, _EDGE_BGR)
    return new_state(shown, labels=labels, base=state["base"], features=X,
                     feature_names=names, feature_groups=groups)


def _describe_features(p, before, after):
    X = after.get("features")
    if X is None:
        return None
    text = "%d regions × %d features" % X.shape
    if int(p["display"]) == 0:
        _, ratio = feature_pca(X, 3)
        text += "  ·  PC1-3 explain %.0f%%" % (100 * ratio.sum())
    elif X.shape[1]:
        i = int(p["feature_index"]) % X.shape[1]
        text += "  ·  showing #%d %s" % (i, after["feature_names"][i])
    return text


def _describe_gaussian(p, before, after):
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
# a "choices" entry turns the control into a segmented button group instead;
# a "multi" entry into toggle chips whose value is a bitmask.
# "when": {key: value} shows a param only while another has that value;
# {key: {"bits": mask}} while a bitmask param has any of those bits set.

def _weight(group, default=1.0):
    return {"key": "w_" + group, "label": group.capitalize() + " weight", "min": 0,
            "max": 3, "step": 0.05, "default": default,
            "when": {"features": {"bits": _group_bits(group)}}}


OPERATIONS = {
    "slic": {
        "label": "SLIC Superpixels",
        "hint": "k-means in (colour, x, y). Compactness trades colour adherence (low) for regular, grid-like cells (high).",
        "fn": slic_superpixels,
        "state": True,
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
    "features": {
        "label": "Region Features",
        "hint": "Per-superpixel features, z-scored and balanced so each counts equally; weights scale whole groups. Measured on SLIC's input.",
        "fn": extract_features,
        "state": True,
        "describe": _describe_features,
        "params": [
            {"key": "features", "label": "Features", "multi": [f[1] for f in FEATURES],
             "groups": [f[2] for f in FEATURES], "default": (1 << 0) | (1 << 3) | (1 << 5)},
            _weight("colour"),
            _weight("position"),
            _weight("texture"),
            _weight("shape"),
            {"key": "display", "label": "Show", "choices": ["PCA → RGB", "One feature"], "default": 0},
            {"key": "feature_index", "label": "Feature #", "min": 0, "max": 40, "step": 1, "default": 0, "when": {"display": 1}},
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
        "describe": lambda p, before, after: (
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
    {"type": "features", "params": {}, "enabled": True},
]


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def new_state(image, labels=None, base=None, features=None, feature_names=None,
              feature_groups=None):
    return {"image": image, "labels": labels, "base": base, "features": features,
            "feature_names": feature_names, "feature_groups": feature_groups}


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


def describe_op(op_type, params, before, after):
    """Short readout of values the sliders imply but do not show directly.

    Gets the op's input and output states: SLIC's real superpixel count, for
    one, only exists once it has run.
    """
    fn = OPERATIONS[op_type].get("describe")
    if fn is None:
        return None
    try:
        return fn(merged_params(op_type, params), before, after)
    except Exception:
        return None


def apply_op(state, op_type, params):
    """Run one block on a state and return the next state.

    Image filters see only state["image"].  One that changes the image size
    invalidates the label map, since it no longer lines up with the pixels.
    """
    spec = OPERATIONS[op_type]
    p = merged_params(op_type, params)
    if spec.get("state"):
        return spec["fn"](state, **p)
    image = spec["fn"](state["image"], **p)
    if image.shape[:2] != state["image"].shape[:2]:
        return new_state(image)
    return {**state, "image": image}


def run_pipeline(image, ops):
    """Run the pipeline from a source image, returning one stage per enabled
    block plus the source: [{label, type, image, state, error, detail}, ...].

    A failing block is skipped rather than killing the whole run, so a bad
    slider value degrades to 'no effect' instead of a blank screen.
    """
    state = new_state(image)
    stages = [{"label": "Source", "type": "source", "image": image, "state": state,
               "error": None, "detail": None}]
    for op in ops:
        if not op.get("enabled", True):
            continue
        op_type = op.get("type")
        if op_type not in OPERATIONS:
            continue
        error = None
        before = state
        try:
            state = apply_op(state, op_type, op.get("params"))
        except Exception as exc:  # keep the UI responsive on bad params
            error = str(exc)
        stages.append({
            "label": OPERATIONS[op_type]["label"],
            "type": op_type,
            "image": state["image"],
            "state": state,
            "error": error,
            "detail": describe_op(op_type, op.get("params"), before, state),
        })
    return stages
