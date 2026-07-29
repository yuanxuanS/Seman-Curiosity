import os


# Standalone tools (for example detector evaluation) should not have to import
# Habitat, Orient Anything, and every policy registration as a side effect of
# importing ``src.detectors``. The default preserves the legacy eager imports.
if os.environ.get("SEMANTIC_CURIOSITY_LIGHT_IMPORT") != "1":
    from .policy_rl.agents import *
    from .policy_rl.envs import *
    from .policy_rl.algo import *
    from .finetune import *

project_name = "Semantic_Curiosity"
