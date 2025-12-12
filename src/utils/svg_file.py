
import io

import cairosvg
import numpy as np
from PIL import Image
from scour import scour
from skimage.metrics import structural_similarity as ssim
from svgpathtools import svg2paths
import xml.etree.ElementTree as ET

def optimize_svg_with_scour(svg):
    options = scour.parse_args([
        '--enable-viewboxing',
        '--enable-id-stripping',
        '--enable-comment-stripping',
        '--shorten-ids',
        '--indent=none',
        '--strip-xml-prolog',
        '--remove-metadata',
        '--remove-descriptive-elements',
        '--disable-embed-rasters',
        '--enable-viewboxing',
        '--create-groups',
        '--renderer-workaround',
        '--set-precision=2',
    ])

    svg = scour.scourString(svg, options)

    svg = svg.replace('id=""', '')
    svg = svg.replace('version="1.0"', '')
    svg = svg.replace('version="1.1"', '')
    svg = svg.replace('version="2.0"', '')
    svg = svg.replace('  ', ' ')
    svg = svg.replace('>\n', '>')

    return svg


def compare_pil_images(img1: Image.Image, img2: Image.Image, size=(384, 384)):
    """
    Compare two PIL images after resizing to a fixed size using SSIM metric.

    Parameters:
        img1 (PIL.Image.Image): The first image to compare.
        img2 (PIL.Image.Image): The second image to compare.
        size (tuple): Target resize dimensions (default: 384x384).

    Returns:
        float: SSIM score between the two images (higher is more similar)
    """
    # Resize both images to the same dimensions using high-quality LANCZOS resampling
    # and convert to grayscale for consistent comparison
    img1_gray = img1.resize(size, Image.Resampling.LANCZOS).convert('L')
    img2_gray = img2.resize(size, Image.Resampling.LANCZOS).convert('L')

    # Convert PIL Images to numpy arrays for numerical processing
    arr1 = np.array(img1_gray)
    arr2 = np.array(img2_gray)

    # Calculate Structural Similarity Index (SSIM) between the two arrays
    # SSIM considers luminance, contrast, and structure for perceptual similarity
    # Returns score (0-1, where 1 means identical) and full comparison map
    score_ssim, _ = ssim(arr1, arr2, full=True)

    # Return only the SSIM score (ignore the full comparison map)
    return score_ssim


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
         The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
         The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
         The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if "viewBox" not in svg_code:
        svg_code = svg_code.replace(
            "<svg", f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_data)).convert("RGB").resize(size)


def extract_topk_polygons(svg_path, top_k=10, samples_per_curve=20):
    # --- Convert segment (line/curve) -> list points ---
    def segment_to_points(seg, n=samples_per_curve):
        return [(seg.point(t).real, seg.point(t).imag)
                for t in np.linspace(0, 1, n)]

    # --- Convert one SVG path -> polygon ---
    def path_to_polygon(path):
        pts = []
        for seg in path:
            pts.extend(segment_to_points(seg))
        return pts

    # --- Shoelace polygon area ---
    def polygon_area(points):
        if len(points) < 3:
            return 0.0
        area = 0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2

    # ---- Parse SVG paths ----
    paths, attributes = svg2paths(svg_path)

    # Convert all to polygons + compute area
    poly_area_pairs = []
    for p in paths:
        poly = path_to_polygon(p)
        area = polygon_area(poly)
        poly_area_pairs.append((area, poly))

    poly_area_pairs.sort(key=lambda x: x[0], reverse=True)

    return [poly for (area, poly) in poly_area_pairs[:top_k]]


def polygons_to_json_object(polygons, default_name="unknown"):
    json_list = []

    for poly in polygons:
        obj = {
            "name": default_name,
            "polygon": [[float(x), float(y)] for (x, y) in poly]
        }
        json_list.append(obj)

    return json_list


def restore_polygon_to_image_coords(polygon, svg_w, svg_h, img_w, img_h):
    scale_x = img_w / svg_w
    scale_y = img_h / svg_h

    restored = [
        [x * scale_x, y * scale_y]
        for (x, y) in polygon
    ]
    return restored

def get_svg_size(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    svg_w = int(float(root.get("width")))
    svg_h = int(float(root.get("height")))

    return svg_w, svg_h
