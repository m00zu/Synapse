"""
nodes/__init__.py
=================
Public API of the core nodes package.

Core nodes are always loaded at startup. Domain-specific nodes
(image processing, statistics, plotting, etc.) live in plugins/
and are loaded dynamically by plugin_loader.py.
"""
from .base import (
    PORT_COLORS,
    NODE_SIGNALS,
    NodeSignals,
    ColorPickerButtonWidget,
    NodeColorPickerWidget,
    NodeFileSelector,
    NodeFileSaver,
    NodeDirSelector,
    NodeProgressBar,
    NodeTableWidget,
    NodeImageWidget,
    NodeToolBoxWidget,
    BaseExecutionNode,
    BaseImageProcessNode,
    _arr_to_pil,
)

from .io_nodes import (
    FileReadNode,
    FolderIteratorNode,
    VideoIteratorNode,
    BatchAccumulatorNode,
    BatchGateNode,
    ImageReadNode,
    SaveNode,
    parse_channels,
)

from .display_nodes import (
    DisplayNode,
    DataTableCellNode,
    DataFigureCellNode,
    ImageCellNode,
)

from .utility_nodes import (
    UniversalDataNode,
    PathModifierNode,
    CollectNode,
    SelectCollectionNode,
    PopCollectionNode,
    SplitCollectionNode,
    SaveCollectionNode,
    RenameCollectionNode,
    CollectionInfoNode,
    FilterCollectionNode,
    MapNamesNode,
)

__all__ = [
    # Shared infrastructure
    'PORT_COLORS', 'NODE_SIGNALS', 'NodeSignals',
    'ColorPickerButtonWidget', 'NodeColorPickerWidget',
    'NodeFileSelector', 'NodeFileSaver', 'NodeDirSelector',
    'NodeProgressBar', 'NodeTableWidget', 'NodeImageWidget',
    'NodeToolBoxWidget', 'BaseExecutionNode',
    # IO
    'FileReadNode', 'FolderIteratorNode', 'VideoIteratorNode',
    'BatchAccumulatorNode', 'BatchGateNode', 'ImageReadNode', 'SaveNode',
    'parse_channels',
    # Display
    'DisplayNode', 'DataTableCellNode', 'DataFigureCellNode',
    'ImageCellNode',
    # Utility
    'UniversalDataNode', 'PathModifierNode',
    'CollectNode', 'SelectCollectionNode', 'PopCollectionNode',
    'SplitCollectionNode', 'SaveCollectionNode',
    'RenameCollectionNode', 'CollectionInfoNode', 'FilterCollectionNode', 'MapNamesNode',
    # DataFrame Operations — moved to data_processing plugin
]
