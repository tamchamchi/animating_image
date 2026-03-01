import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from io import BytesIO
from itertools import product
from typing import List

import vtracer
from PIL import Image

from src.utils.svg_file import compare_pil_images, optimize_svg_with_scour, svg_to_png

from .interface import IConverter


class VtracerGribSearch(IConverter):
    def __init__(self):
        super().__init__()

        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256">
            <circle cx="50" cy="50" r="40" fill="red" />
        </svg>"""

        # Parameter search space
        self.speckle_values = [10, 20, 40]
        self.layer_diff_values = [64, 128]
        self.color_precision_values = [4, 5, 6]

    # ----------------------------------------------------------------------
    # Resize while preserving aspect ratio
    # ----------------------------------------------------------------------
    def _resize_keep_aspect_ratio(self, image: Image.Image, target_max_size=384):
        """
        Resize an image so its longest edge equals `target_max_size`,
        preserving its original aspect ratio.

        Returns:
            resized_image, (orig_w, orig_h), (new_w, new_h)
        """
        orig_w, orig_h = image.size
        aspect = orig_w / orig_h

        if orig_w >= orig_h:
            new_w = target_max_size
            new_h = int(target_max_size / aspect)
        else:
            new_h = target_max_size
            new_w = int(target_max_size * aspect)

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized, (orig_w, orig_h), (new_w, new_h)

    # ----------------------------------------------------------------------
    # Helpers for cleaning SVG metadata
    # ----------------------------------------------------------------------
    def _remove_version_attribute(self, svg_str: str) -> str:
        """
        Remove the "version" attribute from <svg> if it exists.
        """
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.ElementTree(ET.fromstring(svg_str))
        root = tree.getroot()

        if "version" in root.attrib:
            del root.attrib["version"]

        output = BytesIO()
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return output.getvalue().decode("utf-8")

    def _remove_xml_tag(self, svg_str: str) -> str:
        """Remove the <?xml ...?> declaration if present."""
        return re.sub(r"<\?xml[^>]+\?>\s*", "", svg_str)

    # ----------------------------------------------------------------------
    # Main convert method using vtracer + parameter grid search
    # ----------------------------------------------------------------------
    def _convert_by_vtracer(self, image, limit=10000) -> str:
        """
        Try many combinations of vtracer params and choose the best SVG
        under the given size limit based on SSIM score.
        """
        best_svg = ""
        best_ssim = 0.0
        best_size = 0

        # Convert to RGBA
        rgba_image = image.convert("RGBA")

        # Resize *with aspect ratio preserved*
        resized_image, original_size, resized_size = self._resize_keep_aspect_ratio(
            rgba_image, target_max_size=384
        )

        pixels = list(resized_image.getdata())

        for filter_speckle, layer_difference, color_precision in product(
            self.speckle_values,
            self.layer_diff_values,
            self.color_precision_values,
        ):

            svg_code = vtracer.convert_pixels_to_svg(
                rgba_pixels=pixels,
                size=resized_image.size,
                colormode="color",
                hierarchical="stacked",
                mode="polygon",
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                layer_difference=layer_difference,
                corner_threshold=60,
                length_threshold=4.0,
                max_iterations=10,
                splice_threshold=45,
                path_precision=8,
            )

            # Compress SVG
            compressed_svg = optimize_svg_with_scour(svg_code)
            byte_len = len(compressed_svg.encode("utf-8"))

            # Compare similarity with the original
            ssim = compare_pil_images(image, svg_to_png(compressed_svg))

            # Pick the best candidate under size limit
            if byte_len <= limit and byte_len > best_size:
                if ssim >= best_ssim:
                    best_ssim = ssim
                    best_size = byte_len

                    cleaned = self._remove_version_attribute(compressed_svg)
                    cleaned = self._remove_xml_tag(cleaned)
                    best_svg = cleaned

        return best_svg if best_svg else self.default_svg

    # ----------------------------------------------------------------------
    # Process a list of images in parallel
    # ----------------------------------------------------------------------
    def convert_all(self, images: List[Image.Image], max_workers=4, limit=10000) -> List[str]:
        svgs = []
        convert_func = partial(self._convert_by_vtracer, limit=limit)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for svg in executor.map(convert_func, images):
                svgs.append(svg)

        return svgs

    def convert(self, images, limit=20000):
        return self.convert_all(images, limit=limit)
