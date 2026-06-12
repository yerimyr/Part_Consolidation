"""Part Consolidation (PC) prototype environment.

This environment is a *prototype* for researching automated part consolidation
under additive manufacturing constraints.

Key ideas
---------
- A set of parts must be partitioned into groups (consolidated parts).
- The agent constructs groups sequentially using an encoder-decoder policy.
- A special action 0 (SEP) closes the current group and starts a new one.
- A random *compatibility matrix* encodes hard spatial constraints:
  if compat[i,j] == False then parts i and j cannot be in the same group.

Other constraints (material, motion, size/build limit) are handled in the reward.
"""

from .env import PartConsolidationEnv

__all__ = ["PartConsolidationEnv"]
