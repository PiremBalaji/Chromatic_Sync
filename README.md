# Chromatic Sync

Chromatic Sync analyzes garment images, extracts their dominant colors, and
compares clothing palettes with an explainable compatibility score.

## Run Phase 1

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Phase 1 currently validates JPEG/PNG uploads, converts them to RGB, resizes a
copy for analysis, and displays both versions. Segmentation and color
extraction will be added without coupling them to the Streamlit UI.

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
not need to move or copy this extracted folder. The current Phase 1 upload
screen does not read the dataset yet; the segmentation phase will use this
path as its configured dataset root. You may keep extracting a small matching
subset while developing.

Do not train a model or claim accuracy until a labeled evaluation split has
been created.
