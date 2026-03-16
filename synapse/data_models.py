from pydantic import BaseModel, ConfigDict
from typing import Any, Optional, Union, Dict, List
import pandas as pd
from PIL import Image
import matplotlib.figure
# from rdkit.Chem.rdchem import Mol as RDMol

class NodeData(BaseModel):
    """Base class for all data passed between nodes."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metadata: Dict[str, Any] = {}
    source_path: Optional[str] = None

    @classmethod
    def merge(cls, items: list) -> "NodeData":
        """Merge a list of same-typed NodeData objects into one. Override in subclasses."""
        raise NotImplementedError(f"{cls.__name__} does not support merging.")

class TableData(NodeData):
    """Wraps a pandas DataFrame or Series."""
    payload: Union[pd.DataFrame, pd.Series]

    @property
    def df(self) -> pd.DataFrame:
        if isinstance(self.payload, pd.Series):
            return self.payload.to_frame()
        return self.payload

    @classmethod
    def merge(cls, items: list) -> "TableData":
        """Concatenate all DataFrames, injecting frame/file from batch metadata."""
        dfs = []
        for item in items:
            df = item.df.copy()
            meta = getattr(item, 'metadata', {}) or {}
            # Inject frame/file columns from batch context (don't overwrite existing)
            if 'frame' in meta and 'frame' not in df.columns:
                df.insert(0, 'frame', meta['frame'])
            if 'file' in meta and 'file' not in df.columns:
                col_pos = 1 if 'frame' in df.columns else 0
                df.insert(col_pos, 'file', meta['file'])
            dfs.append(df)
        return cls(payload=pd.concat(dfs, ignore_index=True))

class StatData(TableData):
    """Wraps a pandas DataFrame with statistics."""
    pass

class ImageData(NodeData):
    """Wraps a PIL Image."""
    payload: Image.Image
    
    @property
    def image(self) -> Image.Image:
        return self.payload

    @classmethod
    def merge(cls, items: list):
        """Merge images into a TableData with frame, file, and object columns."""
        rows = []
        for item in items:
            meta = getattr(item, 'metadata', {}) or {}
            rows.append({
                'frame': meta.get('frame', ''),
                'file': meta.get('file', ''),
                'object': item,
            })
        return TableData(payload=pd.DataFrame(rows))

class MaskData(ImageData):
    """Wraps a boolean/binary PIL Image representing a mask."""
    pass

class SkeletonData(MaskData):
    """
    A thinned 1-pixel-wide skeleton mask produced by SkeletonizeNode.
    Subclass of MaskData so it can feed any node that accepts a mask,
    but typed distinctly so SkeletonAnalysisNode can require skeleton input.
    """
    pass

class LabelData(NodeData):
    """
    Integer label array where each connected region has a unique positive integer
    value (0 = background).  payload is a numpy int32 ndarray of shape (H, W).
    image (optional) is a pre-generated RGB PIL Image for display purposes,
    produced by the source node so downstream nodes do not need to recompute it.
    """
    payload: Any   # np.ndarray dtype int32, shape (H, W)
    image:   Any = None  # PIL.Image RGB colored visualization

    @classmethod
    def merge(cls, items: list):
        """Merge label arrays into a TableData with frame, file, and object columns."""
        rows = []
        for item in items:
            meta = getattr(item, 'metadata', {}) or {}
            rows.append({
                'frame': meta.get('frame', ''),
                'file': meta.get('file', ''),
                'object': item,
            })
        return TableData(payload=pd.DataFrame(rows))

class FigureData(NodeData):
    """Wraps a matplotlib Figure."""
    payload: Optional[matplotlib.figure.Figure]
    svg_override: Optional[bytes] = None  # edited SVG from SvgEditorNode

    @property
    def fig(self) -> Optional[matplotlib.figure.Figure]:
        return self.payload

    @classmethod
    def merge(cls, items: list):
        """Merge figures into a TableData with frame, file, and object columns."""
        rows = []
        for item in items:
            if item.payload is None:
                continue
            meta = getattr(item, 'metadata', {}) or {}
            rows.append({
                'frame': meta.get('frame', ''),
                'file': meta.get('file', ''),
                'object': item,
            })
        return TableData(payload=pd.DataFrame(rows))

class ConfocalDatasetData(NodeData):
    """Wraps the specialized confocal dataset dictionary."""
    payload: Dict[str, Any]

    @classmethod
    def merge(cls, items: list):
        """Merge confocal datasets into a TableData with frame, file, and object columns."""
        rows = []
        for item in items:
            meta = getattr(item, 'metadata', {}) or {}
            rows.append({
                'frame': meta.get('frame', ''),
                'file': meta.get('file', ''),
                'object': item,
            })
        return TableData(payload=pd.DataFrame(rows))

# class RDMolData(NodeData):
#     """Wraps the specialized RDMol dictionary."""
#     payload: RDMol

#     @classmethod
#     def merge(cls, items: list) -> list:
#         """Collect RDMol into a list."""
#         return [i.payload for i in items]
