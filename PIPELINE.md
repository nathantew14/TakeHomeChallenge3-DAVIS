# Segmentation pipeline

Interactive, block-based editor for classical segmentation of the DAVIS frames in `images/`.
Python (`pipeline.py`) runs in the browser via Pyodide in a Web Worker, so the UI stays responsive.

## Run

```bash
uv sync                       # local deps, only needed to reproduce results outside the browser
python -m http.server 8000    # from this folder -- the page fetches files relative to itself
```

Open http://localhost:8000. The first load downloads numpy, OpenCV and scikit-image (with scipy) into the browser, which takes a while; later loads come from the cache.

## Use

- Pick a sequence from the dropdown at the top of the left panel.
- Add, reorder (drag the grip), toggle or remove blocks. Every stage shows in the filmstrip.
- **SLIC Superpixels**: superpixel count, compactness, optional Gaussian smoothing (sigma), Lab/RGB features, iterations, connectivity. The block reports how many superpixels SLIC actually produced.
- **Export TXT** records every block's settings, plus a Python transcript that reproduces the result with `pipeline.py`. SLIC places its seeds on a grid, so there is no random seed.

The tool only ever reads `images/`. It never opens `masks/`, `previews/` or `original_annotations/`, because those are ground truth.
