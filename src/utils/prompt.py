PROMPT_SUBJECT_STYLE_TRANSFER = """
Use the uploaded image strictly as a reference for drawing style.
Create an image in a hand-drawn style of {subject}.
KEEP IMAGE STRUCTURE.
Both arms extended to the sides. Both legs standing straight.
"""

PROMPT_IMAGE_STYLE_TRANSFER = """
Use the uploaded image strictly as a reference for drawing style.
Recreate the input image in a hand-drawn style.
KEEP THE ORIGINAL POSE, STRUCTURE, AND COMPOSITION.
Ensure both arms remain extended to the sides as in the style reference imageinput.
"""

PROMPT_BG_STYLE_TRANSFER = """
Create a 2D side-scrolling game background designed for horizontal tiling.
The scene should be a stylized cartoon beach environment with clear, separate layers: sky, distant islands, ocean, and foreground sand.
Each layer must be seamless so they can repeat horizontally.
Use vibrant colors and soft gradients similar to a platformer game.
Do not include characters, obstacles, or UI.
Produce a wide panoramic image suitable for parallax scrolling in a 2D game."""

PROMPT_SUBJECT_GENERATION = """
{subject} with both feet pointing to the right, with both arms extended to the sides,
drawn in the style of a colored pencil sketch on white paper. 
Use a naive, childlike art style with visible pencil stroke textures. 
Simple facial features (dot eyes), cute proportions, and a hand-drawn, 
grainy aesthetic. Isolated on white background."""
