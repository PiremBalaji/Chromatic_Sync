# Chromatic Sync

Chromatic Sync analyzes garment images, extracts their dominant colors, and
compares clothing palettes with an explainable compatibility score.

## Run the application

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The current app supports two image sources:

- Upload a JPEG/PNG image for validation and RGB preparation.
- Upload two JPEG/PNG images to compare their palettes. Since uploaded images
  have no annotation mask, this mode provides a manual crop for each image and
  warns that remaining backgrounds can affect the result.
- Browse the DeepFashion2 training images from the external dataset root,
  load the matching JSON annotation, rasterize its segmentation polygons, and
  display the original image, mask, and isolated garment.

The segmentation view uses the dataset's existing annotations, so it does not
download or run a heavy segmentation model. The app runs masked K-means
clustering on garment pixels and displays up to five dominant colors with RGB,
HEX, an approximate color name, and pixel proportion. Background pixels are
not included in dataset-mode clustering.

The dataset view can compare the selected garment with a second training
image. It reports a deterministic heuristic score from 0 to 100, a harmony
category, and an explanation based on hue separation, neutral colors, palette
proportions, brightness, saturation, and CIELAB Delta E. It also suggests
derived complementary, analogous, or neutral colors. This is an explainable
baseline, not a trained or universally valid fashion judgment.

For development, the app reads the first **2,000** matched training image and
annotation pairs. This is a browser/processing limit only; it does not delete
or move anything from the full dataset. Two thousand pairs are sufficient for
testing the color pipeline and creating an initial evaluation set. A larger
sample is recommended later if training a machine-learning model.

Run the tests with:

```powershell
python -m pytest
```

## Dataset recommendation

Download **DeepFashion2** for the next phase. It includes clothing images,
garment categories, bounding boxes, and segmentation annotations, making it
appropriate for testing garment isolation before dominant-color extraction.
Use the official dataset source or its official hosting instructions rather
than an unverified re-upload. It is large, so start with a small subset.

DeepFashion2 is useful for image-level garment segmentation; it does not
provide human-validated clothing-color compatibility labels. For a later
compatibility-learning experiment, use a separately licensed outfit dataset
such as Polyvore Outfits, and document its access and license terms.

For the DeepFashion2 training split, your extracted folder is already in the
correct layout:

```text
E:\DeepFashion2\deepfashion2_original_images\
└── train\
    ├── image\
    │   ├── 000001.jpg
    │   └── ...
    └── annos\
        ├── 000001.json
        └── ...
```

The image and annotation must have the same numeric filename, for example
`image/000001.jpg` and `annos/000001.json`. Do not put only the images in
the project; the annotation files contain the garment categories, bounding
boxes, and segmentation polygons needed for the segmentation phase. You do
not need to move or copy this extracted folder. The application uses this path
as its configured dataset root. You may keep extracting a small matching
subset while developing.

## Why no CNN is required

The DeepFashion2 JSON files already provide ground-truth garment polygons, so
the application can perform reliable local segmentation without downloading
large model weights. A CNN would add significant installation and runtime
requirements but would not provide compatibility labels. The current system is
therefore a complete deterministic baseline: annotated segmentation,
masked K-means, CIELAB/HSV features, explainable scoring, and recommendations.
A learned compatibility model should only be added after collecting
human-reviewed compatible/incompatible garment pairs and a held-out test split.
