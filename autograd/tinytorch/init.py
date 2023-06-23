import os
compute_lib = os.getenv('NANO-AUTOGRAD_COMPUTE', 'numpy')
if compute_lib == 'cupy':
    try:
        import cupy as np
    except ImportError:
        raise ImportError("CuPy library is not installed.")
else:
    import numpy as np