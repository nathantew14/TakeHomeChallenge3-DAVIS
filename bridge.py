"""Browser-side glue between pipeline.py and the Pyodide worker.

Owns three jobs:

  * decode the source image once and keep it resident,
  * run the pipeline through a per-stage cache, so dragging the slider on
    block 5 only recomputes blocks 5..n,
  * hand pixels back as raw RGBA for putImageData -- no PNG encode, which
    was most of the old per-keystroke cost.

Metadata crosses into JS as JSON strings (Pyodide converts str implicitly);
pixels cross as numpy arrays the worker reads with getBuffer('u8'), which is
a view straight onto the wasm heap rather than a copy through a proxy.
"""

import json
import time

import cv2
import numpy as np

import pipeline

PREVIEW_MAX = 1400  # cap on the long side of the stage we actually display
THUMB_MAX = 220

_source = None      # full-resolution BGR, decoded once
_cache = []         # one entry per *enabled* op, parallel to the active list
_last = None        # buffers for the most recent run(), fetched by the worker


# --------------------------------------------------------------------------
# source
# --------------------------------------------------------------------------

def set_source(path, name=None):
    """Decode the source image off Pyodide's virtual FS, where the worker has
    already written it.  Returns JSON metadata; invalidates the cache."""
    global _source, _cache, _last
    name = name or path.rsplit("/", 1)[-1]
    with open(path, "rb") as fh:
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode {name}")
    _source, _cache, _last = img, [], None
    return json.dumps({"path": name, "width": int(img.shape[1]), "height": int(img.shape[0])})


def schema():
    """Operation registry + default pipeline, minus the function objects."""
    ops = {
        # fn_name lets the UI emit a runnable transcript without assuming the
        # registry key matches the function it points at
        key: {"label": spec["label"], "hint": spec["hint"], "params": spec["params"],
              "fn_name": spec["fn"].__name__}
        for key, spec in pipeline.OPERATIONS.items()
    }
    return json.dumps({
        "operations": ops,
        "order": list(pipeline.OPERATIONS.keys()),
        "default_pipeline": pipeline.DEFAULT_PIPELINE,
    })


# --------------------------------------------------------------------------
# pixel helpers
# --------------------------------------------------------------------------

def _fit(image, longest, coarse=False):
    """Resize so the long side is at most `longest`.

    INTER_AREA straight from 4032px to a 220px thumbnail costs ~49ms, which
    dwarfs the pipeline itself.  With coarse=True we decimate by an integer
    stride first and let INTER_AREA finish the job -- ~30x faster.  The stride
    is capped so at least two source pixels per output pixel survive into the
    averaging pass, which is what keeps it from aliasing; the previews the
    user actually judges still take the exact path.
    """
    h, w = image.shape[:2]
    if max(h, w) <= longest:
        return image
    if coarse:
        step = max(1, int(max(h, w) // (longest * 2)))
        if step > 1:
            image = image[::step, ::step]
            h, w = image.shape[:2]
    scale = longest / max(h, w)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _rgba(image, longest, coarse=False):
    """Downsize for display, then widen to the RGBA that ImageData wants."""
    im = _fit(image, longest, coarse)
    code = cv2.COLOR_GRAY2RGBA if im.ndim == 2 else cv2.COLOR_BGR2RGBA
    return np.ascontiguousarray(cv2.cvtColor(im, code))


# --------------------------------------------------------------------------
# cached run
# --------------------------------------------------------------------------

def _key(op):
    """Cache identity of one block: type plus params, order-independent."""
    return json.dumps([op.get("type"), op.get("params") or {}], sort_keys=True)


def _recompute(active):
    """Extend/trim the stage cache to match `active`, returning the index of
    the first stage that had to be recomputed.

    Mirrors pipeline.run_pipeline's error handling: a block that raises is
    reported but leaves the image untouched, so one bad slider value degrades
    to 'no effect' rather than blanking the run.
    """
    global _cache
    keys = [_key(op) for op in active]

    reuse = 0
    while reuse < len(keys) and reuse < len(_cache) and _cache[reuse]["key"] == keys[reuse]:
        reuse += 1

    del _cache[reuse:]
    current = _cache[-1]["image"] if _cache else _source

    for i in range(reuse, len(active)):
        op = active[i]
        error = None
        before = current                      # the describer sees the input
        started = time.perf_counter()
        try:
            current = pipeline.apply_op(current, op["type"], op.get("params"))
        except Exception as exc:
            error = str(exc)
        _cache.append({
            "key": keys[i],
            "image": current,
            "type": op["type"],
            "error": error,
            "detail": pipeline.describe_op(op["type"], op.get("params"), before),
            "ms": (time.perf_counter() - started) * 1000,
        })
    return reuse


def _describe(image, label, op_type, error, detail, ms):
    return {
        "label": label,
        "type": op_type,
        "error": error,
        "detail": detail,
        "ms": round(ms, 2),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": 1 if image.ndim == 2 else int(image.shape[2]),
    }


def run(ops_json, selected, want_all_thumbs=False):
    """Run the pipeline and stage the buffers the worker will collect.

    Returns JSON only.  Pixels come back through preview_bytes()/thumb_bytes(),
    so a stage the UI already has on screen costs nothing to skip.
    """
    global _last
    if _source is None:
        raise RuntimeError("source image not loaded")

    ops = json.loads(ops_json)
    active = [o for o in ops
              if o.get("enabled", True) and o.get("type") in pipeline.OPERATIONS]

    started = time.perf_counter()
    reuse = _recompute(active)
    compute_ms = (time.perf_counter() - started) * 1000

    images = [_source] + [c["image"] for c in _cache]
    stages = [_describe(_source, "Source", "source", None, None, 0.0)]
    for c in _cache:
        stages.append(_describe(c["image"], pipeline.OPERATIONS[c["type"]]["label"],
                                c["type"], c["error"], c["detail"], c["ms"]))

    # Stage 0 is the source and never changes; enabled block i is stage i + 1.
    # Everything from the first recomputed block onward needs a fresh thumb.
    fresh = list(range(len(stages))) if want_all_thumbs else list(range(reuse + 1, len(stages)))

    selected = max(0, min(int(selected), len(stages) - 1))
    started = time.perf_counter()
    _last = {
        "preview": _rgba(images[selected], PREVIEW_MAX),
        "thumbs": {i: _rgba(images[i], THUMB_MAX, coarse=True) for i in fresh},
    }
    encode_ms = (time.perf_counter() - started) * 1000

    return json.dumps({
        "stages": stages,
        "selected": selected,
        "reused": reuse,
        "fresh_thumbs": fresh,
        "compute_ms": round(compute_ms, 1),
        "encode_ms": round(encode_ms, 1),
    })


def preview_array():
    """Contiguous (h, w, 4) uint8. The worker reads it via getBuffer('u8')."""
    return _last["preview"]


def preview_shape():
    a = _last["preview"]
    return json.dumps([int(a.shape[1]), int(a.shape[0])])


def thumb_array(index):
    return _last["thumbs"][int(index)]


def thumb_shape(index):
    a = _last["thumbs"][int(index)]
    return json.dumps([int(a.shape[1]), int(a.shape[0])])


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export_png(ops_json):
    """Re-run from the full-resolution source and encode a PNG.

    Deliberately bypasses the cache: the interactive cache holds whatever the
    pipeline produced, which is already full-res here, but keeping export on
    its own path means a future preview-scaling change can't silently ship a
    downscaled download.
    """
    ops = json.loads(ops_json)
    stages = pipeline.run_pipeline(_source, ops)
    final = stages[-1]["image"]
    ok, buf = cv2.imencode(".png", final)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return np.ascontiguousarray(buf.reshape(-1))
