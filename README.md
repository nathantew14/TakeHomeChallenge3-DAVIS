# Take-Home Challenge 3: Can You Cut It Out?

Choose ONE of the ten supplied images. This is a still-image task: video processing and network training are not required.

## Your experiment
1. Segment the specified target using k-means OR SLIC. You may use colour, position, texture features or filtering. Select clusters/superpixels manually if needed, but do not trace, paint or erase individual pixels. Record your settings and random seed.
2. Identify a failure. Write a prediction about ONE change to your features or parameters, then run the revised method while holding other settings fixed (including random seed where applicable).
3. Apply a promptable model, such as SAM, to the same image and target. Start with one positive click. You may refine the prompt. Record the model/tool, all final prompts, and the number of interactions. Extract original image pixels using the predicted mask. Do not generate replacement content.
4. Finalize and save the initial classical, revised classical and final promptable masks BEFORE opening the ground-truth mask or computing scores. Use ground truth only for evaluation, never as an input, to select clusters, or to tune parameters/prompts. This is a classroom comparison, not an unseen benchmark.
5. Report foreground IoU for each of the three masks. Explain whether the score change supports your prediction and identify one boundary/detail error that the overall IoU score does not fully convey.

## Submission
ONE editable PowerPoint slide showing:
- Original image, sequence/frame ID, and intended target.
- Initial classical, revised classical, and promptable cutouts, each with IoU.
- Features/settings, the one change, and the promptable model/prompts.
- Prediction versus observation and one meaningful difference between the methods.
Keep your three binary prediction PNGs and code so results can be reproduced if requested. Libraries and AI assistance are allowed if you can explain the method.

## Evaluation
Foreground IoU = intersection / union = TP / (TP + FP + FN).
A correct background pixel does not contribute to this score. IoU ranges from 0 to 1. A score of 1 indicates identical masks. Do not average foreground and background IoU for this task.

Use the supplied 854 x 480 image and corresponding mask. Export each prediction at exactly that size, with foreground 255 and background 0 (0/1 is also accepted). Probability outputs must be thresholded consistently before evaluation. For a model that internally resizes the image, map the prediction back to the original image coordinates. Use nearest-neighbour interpolation if resizing an already binary mask. Do not score screenshots, coloured overlays or the RGB cutouts.

Install dependencies if necessary: `pip install pillow numpy`.
Run from this folder, for example:

```sh
python evaluate.py --gt masks/bear__00000.png --pred initial.png revised.png promptable.png
```

The script reports IoU, TP, FP and FN, and rejects incompatible sizes or nonbinary inputs. The edge convention for two empty masks is IoU = 1; none of the supplied reference masks is empty.

## Image choices
All files are DAVIS 2016, 480p, frame 00000. "Medium" is an instructor-oriented visual selection, not an official DAVIS difficulty category. These images are not equally difficult. Selection is based on visual inspection, not measured k-means/SLIC/SAM performance. Grade the controlled comparison and reasoning rather than ranking students by IoU across different images.

| Sequence | Target | Main challenge |
|---|---|---|
| bear | The bear | Brown fur against rocks and soil, shadow and irregular outline |
| blackswan | The swan, excluding its reflection | Thin neck, feather edges and textured water |
| bus | The visible bus, excluding the foreground signpost | Multiple colours, dark windows, partial occlusion |
| camel | The camel, excluding cast shadow | Similar sand/body colours, thin legs and tail |
| car-roundabout | The large dark car in the foreground | Reflections, windows and similar background colours |
| cows | The foreground cow | White/brown patches, thin legs and nearby fence |
| dance-twirl | The foreground dancer including clothing | Thin limbs, clothing detail, people in the background |
| dog | The foreground dog | Fur/grass colour similarity and soft motion boundaries |
| goat | The goat | Camouflage against rock and irregular fur outline |
| tennis | The player, excluding the racket and ball | Small foreground, thin limbs and varied clothing colours |

The binary reference determines the exact annotated boundary. Segment visible target pixels, not inferred hidden parts. The bus, dancer and tennis player illustrate why target scope and occlusion conventions matter. Goat and dance-twirl sit toward the harder end of this selection.

## Files and provenance
- `images/`: 10 original JPEG frames, byte-for-byte copies from the official archive.
- `masks/`: 10 grayscale binary masks (white foreground / black background).
- `original_annotations/`: unchanged official annotation PNGs for provenance.
- `previews/`: turquoise overlays for instructor review of annotation alignment.
- `image-mask-contact-sheet.jpg`: paired image/mask overview for instructor review.
- `manifest.json`: sequence/frame IDs, dimensions, paths, foreground fraction and SHA-256 checksums.
- `DAVIS-original-README.md`: original dataset attribution and terms, preserved verbatim.

Give students `images/`, `masks/`, this README and `evaluate.py`. The mask sheet and overlays are intended for instructor review; students should complete their predictions before viewing the reference masks. If desired, release the masks separately after their predictions are fixed.

Source: https://davischallenge.org/davis2016/code.html
Official archive: https://cgl.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
Citation: Perazzi et al., A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation, CVPR 2016.
Retain the original README and dataset attribution when sharing this teaching subset. Consult the original dataset terms included in that file.
