"""
skimage_nuitka_compat.py
========================
Nuitka frozen-build compatibility patches. Import at the very top of main.py.

Fix 1 — skimage.measure._regionprops KeyError
  _install_properties_docs() iterates over PROPS to add docstrings. In Nuitka
  the PROPS dict may not be populated when that call runs → KeyError: 'area'.
  Silently suppressed via a sys.meta_path loader wrapper.

Fix 2 — Pillow cannot identify TIFF files
  PIL.TiffImagePlugin (and its dependency PIL.TiffTags) are never imported
  anywhere in the codebase, so Nuitka's static analyser does not include them
  in the frozen build. Explicitly importing them here forces inclusion and
  triggers plugin self-registration so Image.open() can identify .tif/.tiff.
"""
import sys

# ── Fix 2: force PIL plugin registration ─────────────────────────────────────
# Nuitka's static analyser only includes modules that are explicitly imported.
# PIL discovers format plugins lazily at runtime, so we must import each plugin
# we need. Calling PIL.Image.init() afterwards forces full registration of all
# included plugins into the PIL.Image.OPEN / SAVE / MIME dicts — this is
# especially important for ThreadPoolExecutor workers where lazy imports may not
# fire correctly in the frozen context.
import PIL.TiffTags         # noqa: F401  – required by TiffImagePlugin
import PIL.TiffImagePlugin  # noqa: F401  – registers TIFF / BigTIFF
import PIL.JpegImagePlugin  # noqa: F401  – registers JPEG
import PIL.PngImagePlugin   # noqa: F401  – registers PNG
import PIL.BmpImagePlugin   # noqa: F401  – registers BMP
import PIL.Image
PIL.Image.init()             # force all imported plugins into OPEN/SAVE dicts
import tifffile               # noqa: F401  – needed for confocal TIFF reading in frozen build


class _RegionpropsNuitkaPatcher:
    """sys.meta_path finder that wraps the _regionprops loader to suppress the KeyError."""

    _TARGET = "skimage.measure._regionprops"

    def find_spec(self, fullname, path, target=None):
        if fullname != self._TARGET:
            return None

        # Find the real spec from the remaining finders
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is not None:
                spec.loader = _PatchedLoader(spec.loader)
                return spec

        return None


class _PatchedLoader:
    """Loader wrapper that catches KeyError raised by _install_properties_docs."""

    def __init__(self, original):
        self._original = original

    def create_module(self, spec):
        if hasattr(self._original, "create_module"):
            return self._original.create_module(spec)
        return None

    def exec_module(self, module):
        try:
            self._original.exec_module(module)
        except KeyError:
            # _install_properties_docs() failed because PROPS is empty in the
            # Nuitka frozen build. All actual regionprops functions (regionprops,
            # label, RegionProperties, …) are defined before that call and are
            # fully available in the module object. Only docstrings are missing.
            pass


sys.meta_path.insert(0, _RegionpropsNuitkaPatcher())