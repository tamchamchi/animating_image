from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np
from PIL import Image

from .interface import IConverter


class ContourBasedConvertor(IConverter):
    def __init__(self):
        super().__init__()

    # ======================
    #  KEEP ASPECT RATIO
    # ======================
    def _resize_keep_aspect(self, image, max_side=384):
        """
        Resize while preserving aspect ratio.
        max_side = largest side after resize.
        """
        orig_w, orig_h = image.size

        # If already smaller, return original
        if max(orig_w, orig_h) <= max_side:
            return image, (orig_w, orig_h), (orig_w, orig_h)

        if orig_w >= orig_h:
            new_w = max_side
            new_h = int(orig_h * (max_side / orig_w))
        else:
            new_h = max_side
            new_w = int(orig_w * (max_side / orig_h))

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return resized, (orig_w, orig_h), (new_w, new_h)

    def _compress_hex_color(self, hex_color):
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
            return f"#{r // 17:x}{g // 17:x}{b // 17:x}"
        return hex_color

    # ==========================================================
    #           CONTOUR FEATURE EXTRACTION
    # ==========================================================
    def _extract_features_by_scale(self, img_np, num_colors=16):
        if len(img_np.shape) == 3 and img_np.shape[2] > 1:
            img_rgb = img_np
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        # Color quantization
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        palette = centers.astype(np.uint8)
        quantized = palette[labels.flatten()].reshape(img_rgb.shape)

        hierarchical_features = []

        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_colors = [palette[i] for i in sorted_indices]

        center_x, center_y = width / 2, height / 2

        for color in sorted_colors:
            color_mask = cv2.inRange(quantized, color, color)

            contours, _ = cv2.findContours(
                color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            hex_color = self._compress_hex_color(
                f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            )

            color_features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:
                    continue

                m = cv2.moments(contour)
                if m["m00"] == 0:
                    continue

                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])

                dist_from_center = np.sqrt(
                    ((cx - center_x) / width) ** 2 +
                    ((cy - center_y) / height) ** 2
                )

                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                points = " ".join(
                    [f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx]
                )

                importance = (
                    0.5 * area +
                    0.3 * (1 - dist_from_center) +
                    0.2 * len(approx)
                )

                color_features.append(
                    {
                        "points": points,
                        "color": hex_color,
                        "area": area,
                        "importance": importance,
                        "point_count": len(approx),
                    }
                )

            color_features.sort(key=lambda x: x["importance"], reverse=True)
            hierarchical_features.extend(color_features)

        hierarchical_features.sort(key=lambda x: x["importance"], reverse=True)
        return hierarchical_features

    # ==========================================================
    #        POLYGON SIMPLIFICATION
    # ==========================================================
    def _simplify_polygon(self, points_str, simplification_level):
        if simplification_level == 0:
            return points_str

        points = points_str.split()

        # Level 1: 1 decimal
        if simplification_level == 1:
            return " ".join([
                f"{float(p.split(',')[0]):.1f},{float(p.split(',')[1]):.1f}"
                for p in points
            ])

        # Level 2: integer
        if simplification_level == 2:
            return " ".join([
                f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                for p in points
            ])

        # Level 3: reduce #points
        if simplification_level == 3:
            if len(points) <= 4:
                return " ".join([
                    f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                    for p in points
                ])

            step = min(2, len(points) // 3)
            reduced = [points[i] for i in range(0, len(points), step)]
            if len(reduced) < 3:
                reduced = points[:3]
            if points[-1] not in reduced:
                reduced.append(points[-1])

            return " ".join([
                f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                for p in reduced
            ])

        return points_str

    # ==========================================================
    #        MAIN BITMAP → SVG FUNCTION (WITH ASPECT RATIO)
    # ==========================================================
    def bitmap_to_svg_layered(
        self,
        image,
        limit=10000,
        resize=True,
        max_side=384,
        adaptive_fill=True,
        num_colors=None,
    ):
        """
        Convert bitmap → SVG while preserving 
        - original SVG width/height
        - correct aspect ratio in viewBox
        """
        # Resize but keep aspect ratio
        if resize:
            resized, (orig_w, orig_h), (new_w, new_h) = self._resize_keep_aspect(image, max_side=max_side)
            image = resized
        else:
            orig_w, orig_h = image.size
            new_w, new_h = orig_w, orig_h

        img_np = np.array(image)

        # Adaptive color count
        if num_colors is None:
            pixel_count = new_w * new_h
            if pixel_count < 65536:
                num_colors = 8
            elif pixel_count < 262144:
                num_colors = 12
            else:
                num_colors = 16

        height, width = img_np.shape[:2]

        # Background color
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            c = np.mean(img_np, axis=(0, 1)).astype(int)
            bg_hex = self._compress_hex_color(f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}")
        else:
            bg_hex = "#fff"

        # SVG header uses original size
        svg_header = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{orig_w}" height="{orig_h}" '
            f'viewBox="0 0 {width} {height}">\n'
        )
        svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex}"/>\n'

        svg = svg_header + svg_bg
        svg_footer = "</svg>"
        base_size = len((svg + svg_footer).encode("utf-8"))

        features = self._extract_features_by_scale(img_np, num_colors=num_colors)

        # Not adaptive: just add highest importance polygons
        if not adaptive_fill:
            for feat in features:
                poly = f'<polygon points="{feat["points"]}" fill="{feat["color"]}" />\n'
                if len((svg + poly + svg_footer).encode("utf-8")) > limit:
                    break
                svg += poly
            return svg + svg_footer

        # Adaptive mode
        feature_sizes = []
        for feat in features:
            feature_sizes.append(
                {
                    "original": len(f'<polygon points="{feat["points"]}" fill="{feat["color"]}" />\n'.encode("utf-8")),
                    "level1": len(f'<polygon points="{self._simplify_polygon(feat["points"],1)}" fill="{feat["color"]}" />\n'.encode("utf-8")),
                    "level2": len(f'<polygon points="{self._simplify_polygon(feat["points"],2)}" fill="{feat["color"]}" />\n'.encode("utf-8")),
                    "level3": len(f'<polygon points="{self._simplify_polygon(feat["points"],3)}" fill="{feat["color"]}" />\n'.encode("utf-8")),
                }
            )

        bytes_used = base_size
        added = set()

        # Pass 1: highest quality
        for i, feat in enumerate(features):
            size = feature_sizes[i]["original"]
            if bytes_used + size <= limit:
                svg += f'<polygon points="{feat["points"]}" fill="{feat["color"]}" />\n'
                bytes_used += size
                added.add(i)

        # Pass 2: simplified levels
        for level in range(1, 4):
            for i, feat in enumerate(features):
                if i in added:
                    continue
                size = feature_sizes[i][f"level{level}"]
                if bytes_used + size <= limit:
                    simplified = self._simplify_polygon(feat["points"], level)
                    svg += f'<polygon points="{simplified}" fill="{feat["color"]}" />\n'
                    bytes_used += size
                    added.add(i)

        # Safety check
        final_svg = svg + svg_footer
        if len(final_svg.encode("utf-8")) > limit:
            return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex}"/></svg>'

        return final_svg

    def convert_all(self, images, max_workers=4, limit=10000):
        svgs = []
        convert_func = partial(self.bitmap_to_svg_layered, limit=limit)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for svg in executor.map(convert_func, images):
                svgs.append(svg)

        return svgs

    def convert(self, images, limit=20000):
        return self.convert_all(images, limit=limit)
