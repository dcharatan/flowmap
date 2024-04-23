# Disable Ruff's rules on star imports for common.
# ruff: noqa: F405, F403

import os
import uuid
from pathlib import Path

from dominate import document
from dominate.tags import div, img, link
from PIL import Image

from .common import *

OUTPUT_PATH = Path("figures/splat_comparison")
GT_PATH = Path("/mnt/sn850x/flowmap_converted/flowmap")
FALLBACK = Path.home() / "Downloads/flowcameron.jpeg"
ASPECT_RATIO = 3 / 2

SCENES = (
    # name, image name, callout size (%), callout x offset (%), callout y offset (%)
    (SCENE_CATERPILLAR, "001.png", 25, 42, 28),
    (SCENE_BENCH, "011.png", 25, 20, 2),
    (SCENE_HYDRANT, "021.png", 25, 28, 15),
    (SCENE_KITCHEN, "041.png", 25, 45, 25),
    # (SCENE_ROOM0, "001.png", 20, 60, 10),
)

METHODS = (
    METHOD_FLOWMAP,
    METHOD_MVSCOLMAP,
    METHOD_DROIDSLAM,
    METHOD_NOPENERF,
)

# It seems like there's some kind of rounding that happens during the export process.
# This makes it less noticeable.
M = 10
GAP_SM = f"{M}pt"
GAP_LG = f"{4 * M}pt"

# Procedure for HTML to PDF conversion:
# 1. Open the HTML file in Chrome.
# 2. Inspect the total height of the HTML tag. Update @page to have the correct height,
#    which is the second element of size.
# 3. Using the print dialog, save to PDF. Make sure that in "more settings," the
#    "background graphics" box is checked.

CSS = f"""
@page {{
    size: {M * 347.12361}pt {5249.5 * 1.005}px;
    margin: 0;
}}

html, body {{
    margin: 0;
    padding: 0;
    width: {M * 347.12361}pt;
    font-size: {M * 7.5}pt;
}}

.grid {{
    display: grid;
    grid-template-columns: 1fr {GAP_LG} 1fr {GAP_SM} 1fr {GAP_LG} 1fr {GAP_SM} 1fr;
    grid-gap: 0 0;
    width: 100%;
}}

.grid-item {{
    overflow: hidden;
    position: relative;
}}

.grid-image {{
    width: 100%;
    display: block;
    aspect-ratio: {ASPECT_RATIO};
    object-fit: cover;
}}

.gt-callout {{
    position: absolute;
    border: {2 * M}pt solid #E6194B;
}}

.estimates-intrinsics {{
    grid-column: 3 / span 3;
    font-weight: bold;
    border-bottom: {0.5 * M}pt solid black;
}}

.requires-known-intrinsics {{
    grid-column: 7 / span 3;
    font-weight: bold;
    border-bottom: {0.5 * M}pt solid black;
}}

"""


def crop(image: Image.Image, ratio: float) -> Image.Image:
    """Crop images so they have the same aspect ratio."""
    w, h = image.size

    if w / h < ratio:
        # Chop off the top and bottom.
        h_new = int(w / ratio)
        image = image.crop((0, (h - h_new) // 2, w, h - (h - h_new) // 2))
    else:
        # Chop off the sides.
        w_new = int(h * ratio)
        image = image.crop(((w - w_new) // 2, 0, w - (w - w_new) // 2, h))
    return image


def crop_and_save(
    image: Image.Image,
    path: Path,
    ratio: float = ASPECT_RATIO,
) -> None:
    image = crop(image, ratio)
    image.save(path)


def row_gap(height: str):
    div(style=f"height: {height}; grid-column: 1 / span {2 * len(METHODS) + 1}")


def get_image(path: Path) -> Path:
    if not path.exists():
        path = FALLBACK
    copied_name = f"{uuid.uuid4().hex}{path.suffix}"
    os.system(f'cp "{path}" "{OUTPUT_PATH / copied_name}"')
    return copied_name


if __name__ == "__main__":
    # Define the figure layout.
    with (doc := document("fig_splat_comparison_pdf")):
        with doc.head:
            link(rel="stylesheet", href="style.css")
        with div(cls="grid"):
            # Top Row
            div("Estimates Intrinsics", cls="estimates-intrinsics")
            div("Requires Known Intrinsics", cls="requires-known-intrinsics")

            # Label Row
            div("Ground Truth")
            div()
            for col, method in enumerate(METHODS):
                # Add the label.
                div(method.full_name)

                # Add a variable-width column gap.
                if col < len(METHODS) - 1:
                    div()

            row_gap(GAP_SM)

            for row, (scene, image, cl_size, cl_x, cl_y) in enumerate(SCENES):
                # Collect the images used in the column.
                gt_image = get_image(GT_PATH / scene.tag / "images" / image)
                method_images = [
                    get_image(
                        METRICS_PATH
                        / f"{METRICS_PREFIX}{scene.tag}_{method.tag}/{image}"
                    )
                    for method in METHODS
                ]

                # Add the ground-truth image.
                with div(cls="grid-item"):
                    # Add the ground-truth image if it exists.
                    img(cls="grid-image", src=gt_image)
                    div(
                        cls="gt-callout",
                        style=(
                            f"width: {cl_size}%; "
                            f"height: {cl_size}%; "
                            f"left: {cl_x}%; "
                            f"top: {cl_y}%; "
                        ),
                    )

                # Add the method results.
                for col, _ in enumerate(METHODS):
                    # Add a div that's used as a variable-width column gap.
                    div()

                    # Add the grid item.
                    with div(cls="grid-item"):
                        img(cls="grid-image", src=method_images[col])

                        # This would add a little PSNR callout. It covers too much of
                        # the available space though.
                        # div(
                        #     "12.34 dB",
                        #     style=(
                        #         f"position: absolute; right: {2 * M}pt; "
                        #         f"bottom: {2 * M}pt; background-color: #000a; "
                        #         f"color: #fff; padding: {M}pt {2 * M}pt; "
                        #         f"border-radius: {M}pt; "
                        #     ),
                        # )

                row_gap(GAP_SM)

                # Add the callouts.
                scale = 100 / cl_size
                x = (100 - cl_size) / 2 - cl_x
                y = (100 - cl_size) / 2 - cl_y

                # Add the grid item.
                with div(cls="grid-item"):
                    img(
                        cls="grid-image",
                        src=gt_image,
                        style=(
                            f"transform: scale({scale}, {scale}) "
                            f"translate({x}%, {y}%); "
                            "z-index: 1; "
                        ),
                    )

                    # Add the red border.
                    div(
                        style=(
                            "position: absolute;  width: 100%; height: 100%; "
                            "z-index: 2; box-sizing: border-box; "
                            f"border: {2 * M}pt solid #E6194B; top: 0; left: 0; "
                        )
                    )

                # Add the spacer.
                div()

                for col, _ in enumerate(METHODS):
                    # Add the grid item.
                    with div(cls="grid-item"):
                        img(
                            cls="grid-image",
                            src=method_images[col],
                            style=(
                                f"transform: scale({scale}, {scale}) "
                                f"translate({x}%, {y}%); "
                                "z-index: 1; "
                            ),
                        )

                    # Add a div that's used as a variable-width column gap.
                    if col < len(METHODS) - 1:
                        div()

                if row != len(SCENES) - 1:
                    row_gap(GAP_LG)

    # Write the HTML and CSS documents.
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    with (OUTPUT_PATH / "index.html").open("w") as f:
        f.write(str(doc))
    with (OUTPUT_PATH / "style.css").open("w") as f:
        f.write(CSS)
