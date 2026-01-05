from typing import Dict, List


def calculate_polygon_area(polygon_coords: List[List[float]]) -> float:
    """
    Calculates the area of a polygon using the Shoelace formula.

    The Shoelace formula (also known as Gauss's area formula) is a method
    for finding the area of a simple polygon given the coordinates of its vertices.
    It takes a list of (x, y) coordinates of the vertices in order (clockwise or counter-clockwise).

    Args:
        polygon_coords: A list of [x, y] coordinates representing the vertices
                        of the polygon. The vertices should be in order.

    Returns:
        The calculated area of the polygon as a float. Returns 0.0 if the
        polygon has fewer than 3 vertices.

    Examples:
        >>> calculate_polygon_area([[0,0], [0,1], [1,1], [1,0]])
        1.0
        >>> calculate_polygon_area([[0,0], [1,0], [0.5,1]])
        0.5
    """
    n = len(polygon_coords)
    if n < 3:
        # A polygon must have at least 3 vertices to have a meaningful area.
        return 0.0

    area = 0.0
    # Iterate through the vertices, connecting the last vertex to the first.
    for i in range(n):
        x1, y1 = polygon_coords[i]
        x2, y2 = polygon_coords[
            (i + 1) % n
        ]  # (i + 1) % n wraps around to the first vertex
        area += x1 * y2 - x2 * y1

    # The formula yields twice the area, so we divide by 2 and take the absolute value
    # to handle polygons defined in either clockwise or counter-clockwise order.
    return abs(area / 2.0)


def assign_polygon_ids_by_area(input_list: List[Dict]) -> List[Dict]:
    """
    Assigns a unique 'id_polygon' to each polygon object in a list,
    based on their calculated area. The polygons are sorted by area in ascending order.

    The function first calculates the area for each polygon, sorts the entire
    list by this area, and then assigns sequential IDs starting from 1.

    Args:
        input_list: A list of dictionaries. Each dictionary is expected to have
                    at least a 'polygon' key, whose value is a list of [x, y]
                    coordinates representing the polygon's vertices.
                    Example: [{'name': 'PolyA', 'polygon': [[0,0], [0,1], [1,1]]}, ...]

    Returns:
        A new list of dictionaries, where each dictionary now includes an
        'id_polygon' field. The list is sorted by polygon area in ascending order,
        and the temporary 'area' field is removed from the final output.

    Raises:
        KeyError: If any dictionary in the input_list does not contain a 'polygon' key.
        TypeError: If the 'polygon' value is not a list of lists (coordinates).

    Examples:
        >>> polygons = [
        ...     {'name': 'Square', 'polygon': [[0,0], [0,2], [2,2], [2,0]]},
        ...     {'name': 'Triangle', 'polygon': [[0,0], [1,0], [0.5,1]]},
        ...     {'name': 'Rectangle', 'polygon': [[0,0], [0,3], [1,3], [1,0]]}
        ... ]
        >>> sorted_polygons = assign_polygon_ids_by_area(polygons)
        >>> for p in sorted_polygons: print(p)
        {'name': 'Triangle', 'polygon': [[0, 0], [1, 0], [0.5, 1]], 'id_polygon': 1}
        {'name': 'Square', 'polygon': [[0, 0], [0, 2], [2, 2], [2, 0]], 'id_polygon': 2}
        {'name': 'Rectangle', 'polygon': [[0, 0], [0, 3], [1, 3], [1, 0]], 'id_polygon': 3}
    """
    if not input_list:
        return []

    # Step 1: Calculate the area for each polygon and temporarily store it.
    # We create a copy of each item to avoid modifying the original input_list
    # directly before sorting and final ID assignment.
    polygons_with_areas = []
    for item in input_list:
        # Ensure the 'polygon' key exists
        if "polygon" not in item:
            raise KeyError(
                "Each dictionary in input_list must contain a 'polygon' key."
            )

        item_copy = item.copy()
        # Calculate area using the helper function
        item_copy["area"] = calculate_polygon_area(item_copy["polygon"])
        polygons_with_areas.append(item_copy)

    # Step 2: Sort the list of polygons based on their calculated area in ascending order.
    polygons_with_areas.sort(key=lambda x: x["area"])

    # Step 3: Assign sequential 'id_polygon' values to the sorted polygons.
    output_list = []
    for i, item in enumerate(polygons_with_areas):
        # Create a final copy to add the 'id_polygon' field
        final_item = item.copy()
        final_item["id_polygon"] = i + 1  # IDs typically start from 1

        # Remove the temporary 'area' field if it's not needed in the final output.
        del final_item["area"]
        output_list.append(final_item)

    return output_list
