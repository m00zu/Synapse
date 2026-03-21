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
    SvgEditorNode,
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

from .dataframe_nodes import (
    EditableTableNode,
    FilterTableNode,
    MathColumnNode,
    AggregateTableNode,
    RenameGroupNode,
    ReshapeTableNode,
    SortTableNode,
    TopNNode,
    ColumnValueSplitNode,
    TwoTableMathNode,
    SelectColumnsNode,
    RandomSampleNode,
    ExtractObjectNode,
    ConcatTablesNode,
    JoinTablesNode,
    DropFillNaNNode,
    NormalizeColumnNode,
    ValueCountsNode,
    DropDuplicatesNode,
    TypeCastColumnNode,
    StringColumnOpsNode,
    GroupNormalizationNode,
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
    'ImageCellNode', 'SvgEditorNode',
    # Utility
    'UniversalDataNode', 'PathModifierNode',
    'CollectNode', 'SelectCollectionNode', 'PopCollectionNode',
    'SplitCollectionNode', 'SaveCollectionNode',
    'RenameCollectionNode', 'CollectionInfoNode', 'FilterCollectionNode', 'MapNamesNode',
    # DataFrame Operations
    'EditableTableNode', 'FilterTableNode', 'MathColumnNode',
    'AggregateTableNode', 'RenameGroupNode', 'ReshapeTableNode',
    'SortTableNode', 'TopNNode', 'ColumnValueSplitNode',
    'TwoTableMathNode', 'SelectColumnsNode', 'RandomSampleNode',
    'ExtractObjectNode',
    'ConcatTablesNode', 'JoinTablesNode', 'DropFillNaNNode', 'NormalizeColumnNode',
    'ValueCountsNode', 'DropDuplicatesNode', 'TypeCastColumnNode', 'StringColumnOpsNode',
    'GroupNormalizationNode',
]
