"""
plugin_imports.py
=================
Explicit imports for packages used by common plugins.

Nuitka traces imports statically at compile time. Since built-in plugins
are loaded dynamically via importlib (not as static Python imports), Nuitka
would not discover their dependencies automatically.

Importing this module from main.py ensures Nuitka bundles all packages
that the built-in plugins rely on.

Do NOT remove imports here even if they appear unused — they exist solely
to inform the Nuitka dependency tracer.
"""

# ── image_process_nodes ────────────────────────────────────────────────────
import numpy                             # noqa: F401
import PIL                               # noqa: F401
import PIL.Image                         # noqa: F401
import PIL.ImageDraw                     # noqa: F401
import PIL.ImageFilter                   # noqa: F401
import PIL.ImageEnhance                  # noqa: F401
import PIL.ImageOps                      # noqa: F401
import skimage                           # noqa: F401
import skimage.exposure                  # noqa: F401
import skimage.filters                   # noqa: F401
import skimage.morphology                # noqa: F401
import skimage.color                     # noqa: F401
import skimage.transform                 # noqa: F401
import skimage.restoration              # noqa: F401
import scipy.ndimage                     # noqa: F401
try:
    import pyqtgraph                     # noqa: F401
except ImportError:
    pass  # optional — only needed by volume_nodes plugin

# ── roi_nodes ──────────────────────────────────────────────────────────────
# (uses PIL, numpy, PySide6 — already covered above / bundled by Qt plugin)

# ── mask_nodes ────────────────────────────────────────────────────────────
import skimage.measure                   # noqa: F401
import skimage.segmentation              # noqa: F401
import skimage.feature                   # noqa: F401
import skimage.graph                     # noqa: F401
import scipy.spatial                     # noqa: F401

# ── vision_nodes ──────────────────────────────────────────────────────────
import skimage.feature                   # noqa: F401
import skimage.draw                      # noqa: F401

# ── analysis_nodes ────────────────────────────────────────────────────────
import pandas                            # noqa: F401
import scipy.stats                       # noqa: F401

# ── stats_nodes ───────────────────────────────────────────────────────────
import scipy.optimize                    # noqa: F401
import statsmodels                       # noqa: F401

# ── plot_nodes ────────────────────────────────────────────────────────────
import matplotlib                        # noqa: F401
import matplotlib.pyplot                 # noqa: F401
import matplotlib.figure                 # noqa: F401
import matplotlib.backends.backend_agg  # noqa: F401
import seaborn                       # noqa: F401

# ── filopodia_nodes ───────────────────────────────────────────────────────
# (uses skimage, numpy, PIL, scipy — all covered above)

# ── tifffile (used by io_nodes but also image plugins) ────────────────────
import tifffile                      # noqa: F401

# ── Rust extension modules ─────────────────────────────────────────────────
# These are compiled .so/.pyd files. Nuitka copies binary extensions it can
# trace statically — importing them here ensures they're included in the build.
try:
    import oir_reader_rs             # noqa: F401
    import image_process_rs          # noqa: F401
except ImportError:
    pass  # optional Rust extensions