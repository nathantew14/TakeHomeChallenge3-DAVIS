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
- **Region Features** (after SLIC): one weight slider (0-3) per feature, grouped by colour (lightness L, chroma a/b, Lab std, HSV mean), position (x, y centroid, extent), texture (LBP, Gabor, gradient) and shape (area, aspect). Weight 0 leaves a feature out. Features are measured on the image SLIC segmented, not the overlay. Each column is z-scored across regions and each feature is scaled by 1/sqrt(its width), so at weight 1 every feature carries one unit of variance, and weight w counts w² in k-means distances. Defaults: L, a/b, x, y, LBP at 1, the rest 0. *Show*: PCA → RGB of the weighted matrix (similar regions, similar colours), or *Feature map* -- touching any weight slider switches to that feature's heatmap (multi-column features show their first principal component), even at weight 0.
- **K-means Clustering** (after Region Features): k, random seed, restarts. Implemented in numpy (k-means++ seeding) so the seed reproduces identically in the browser and locally; clusters are numbered largest-first and the readouts name them by colour (c0 red, c1 blue, ...).
- **Foreground Rule** (after K-means): border contact. A cluster -- or, with *Score per: Connected blob*, each connected piece of a cluster -- is foreground if it owns at most *Max border share* of the image's outer pixel ring and covers at least *Min cluster area*. If nothing qualifies, the least-contact one is used and the readout says so. Then optionally keep the largest blob and fill holes. The mask is resized to the source size with nearest-neighbour.
- **Export mask** writes the last mask as a single-channel 0/255 PNG at 854×480, ready for `evaluate.py`. Rename it (e.g. `initial.png`, `revised.png`) before evaluating.
- **Export TXT** records every block's settings, plus a Python transcript that reproduces the result with `pipeline.run_pipeline` (the last stage's `state` holds the labels and feature matrix). SLIC places its seeds on a grid, so there is no random seed.

The tool only ever reads `images/`. It never opens `masks/`, `previews/` or `original_annotations/`, because those are ground truth.
