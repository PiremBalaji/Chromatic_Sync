"""Phase 1 Streamlit application for Chromatic Sync."""

import streamlit as st

from modules.preprocessing import resize_for_analysis, load_image

st.set_page_config(page_title="Chromatic Sync", page_icon="🎨", layout="wide")
st.title("Chromatic Sync")
st.write("Upload a garment image to validate it and prepare it for color analysis.")

uploaded_file = st.file_uploader(
    "Choose a garment image",
    type=["jpg", "jpeg", "png"],
    help="JPEG and PNG files up to 10 MB are supported.",
)

if uploaded_file is not None:
    try:
        image = load_image(uploaded_file.getvalue())
        analysis_image = resize_for_analysis(image)
    except (TypeError, ValueError) as error:
        st.error(str(error))
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("Analysis preview")
            st.image(analysis_image, use_container_width=True)
            st.caption(
                f"RGB image: {analysis_image.width} × {analysis_image.height} pixels"
            )
        st.success("Image loaded successfully. Segmentation and color extraction are next.")
