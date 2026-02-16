"""Dataset Distillation RL package root.

Exports convenience helpers from submodules so `import adra` works out-of-the-box.
"""

from importlib import import_module as _imp

# Optionally re-export commonly-used symbols
for _name in [
    "utils_rl",
    "utils",
    "utils_rl.checkpoint_manager",
    "logit_difference",  # new – exposes logit_difference_rank at package level
]:
    try:
        _imp(_name)
    except ModuleNotFoundError:
        pass

del _imp, _name 