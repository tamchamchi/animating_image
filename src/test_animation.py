from .animator.meta_animator import MetaAnimator
animator = MetaAnimator()

char_folder = "/home/anhndt/animating_image/src/configs/characters/char13"
char_name = "char13"
actions=["standing", "jumping", "running", "jesse_dancing", "waving", "speaking"]
# actions = ["standing", "waving"]
# actions = ["running"]
# actions = ["speaking"]

for action in actions:
    animator.animate(action=action, char_path=char_folder, char_name=char_name)