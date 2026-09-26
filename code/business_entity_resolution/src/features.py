# Stub for features.py so that train.py can be tested locally before Krrish finishes his track.
FEATURE_VERSION = 1

# Format: (feature_name, monotonic_direction)
# 1 = strictly increasing (higher similarity -> higher match prob)
# -1 = strictly decreasing
# 0 = unconstrained
FEATURE_NAMES = [
    "prior_score",
    "n_channels",
    "best_rank", 
    "is_source3",
]

def featurise(s1_rows, cand_rows, context):
    """Stub implementation to be written by Krrish"""
    pass
