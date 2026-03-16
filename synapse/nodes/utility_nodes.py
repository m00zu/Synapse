"""
nodes/utility_nodes.py
======================
General-purpose utility nodes.
"""
import NodeGraphQt
from ..data_models import TableData, ImageData
from PIL import Image
import pandas as pd
import numpy as np
from .base import (
    BaseExecutionNode, PORT_COLORS,
    NodeDirSelector,
)


class UniversalDataNode(BaseExecutionNode):
    """
    Executes arbitrary Python code to process multiple inputs and push results to outputs.

    Available variables in user code:
    - `inputs` — list of upstream data values
    - `output` — assign the result here (auto-wrapped into `TableData`, `ImageData`, or `FigureData`)
    - `pd`, `np`, `plt`, `sns` — pre-imported libraries

    Keywords: script, python, custom logic, arbitrary code, transform, 腳本, 工具, 自訂邏輯, 通用, 轉換
    """
    __identifier__ = 'nodes.data'
    NODE_NAME = 'Universal Node'
    PORT_SPEC = {'inputs': ['any'], 'outputs': ['any']}

    def __init__(self):
        super(UniversalDataNode, self).__init__()

        self.add_input('in', multi_input=True, color=PORT_COLORS['any'])
        self.add_output('out', multi_output=True, color=PORT_COLORS['any'])

        self.add_text_input('code', 'Python Code', tab='Execution')
        self.set_property('code', 'output = inputs[0]')
        
        self.output_values = {}

    def evaluate(self):
        """
        Executes the python code with 'inputs' representing a list of upstream data.
        'outputs' should be filled by the user's python code as a list.
        """
        self.reset_progress()
        in_values = []
        in_port = self.inputs().get('in')
        if in_port and in_port.connected_ports():
            for connected in in_port.connected_ports():
                upstream_node = connected.node()
                up_val = upstream_node.output_values.get(connected.name(), None)
                if isinstance(up_val, TableData):
                    up_val = up_val.df
                elif hasattr(up_val, 'payload'):
                    up_val = up_val.payload
                in_values.append(up_val)
                
        local_scope = {
            "inputs": in_values, 
            "output": None
        }
        
        func_str = self.get_property("code")
        
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            local_scope['pd'] = pd
            local_scope['np'] = np
            local_scope['plt'] = plt
            local_scope['sns'] = sns
            
            exec(func_str, globals(), local_scope)
            
            if "output" in local_scope:
                res = local_scope["output"]
                if isinstance(res, (pd.DataFrame, pd.Series)):
                    from data_models import TableData
                    res = TableData(payload=res)
                elif isinstance(res, Image.Image):
                    res = ImageData(payload=res)
                elif hasattr(res, '__class__') and 'Figure' in type(res).__name__:
                    from data_models import FigureData
                    res = FigureData(payload=res)
                
                self.output_values['out'] = res
            
            self.mark_clean()
            self.set_progress(100)
            return True, None
            
        except Exception as e:
            self.mark_error()
            return False, str(e)


class PathModifierNode(BaseExecutionNode):
    """
    Takes a file path and modifies it by adding a suffix, changing the extension, or overriding the folder.

    **suffix** — string appended to the file stem (default: `_analyzed`).
    **ext** — replacement file extension (leave empty to keep original).
    **folder** — optional folder override for the output path.

    Keywords: path, filename, suffix, extension, rename, 路徑, 檔名, 副檔名, 工具, 重新命名
    """
    __identifier__ = 'nodes.utility'
    NODE_NAME = 'Path Modifier'
    PORT_SPEC = {'inputs': ['path'], 'outputs': ['path']}

    def __init__(self):
        super(PathModifierNode, self).__init__(use_progress=False)
        self.add_input('path', color=PORT_COLORS['path'])
        self.add_output('path', color=PORT_COLORS['path'])
        
        self.add_text_input('suffix', 'Suffix', text='_analyzed')
        self.add_text_input('ext', 'New Extension', text='')
        folder_selector = NodeDirSelector(self.view, name='folder', label='Folder Override')
        self.add_custom_widget(
            folder_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties'
        )

    def evaluate(self):
        from pathlib import Path
        
        in_port = self.inputs().get('path')
        if not in_port or not in_port.connected_ports():
            self.mark_error()
            return False, "No input path"
            
        connected = in_port.connected_ports()[0]
        upstream_node = connected.node()
        orig_path_val = upstream_node.output_values.get(connected.name(), None)
        
        if not orig_path_val:
            self.mark_error()
            return False, "Input path is empty"
        
        if isinstance(orig_path_val, str):
            orig_path_str = orig_path_val
        else:
            self.mark_error()
            return False, f"Unsupported path input type: {type(orig_path_val).__name__}"

        orig_path = Path(orig_path_str)
        suffix = self.get_property('suffix')
        new_ext = self.get_property('ext')
        folder_override = self.get_property('folder')
        
        stem = orig_path.stem
        ext = new_ext if new_ext else orig_path.suffix
        if ext and not ext.startswith('.'):
            ext = '.' + ext
            
        new_filename = f"{stem}{suffix}{ext}"
        
        if folder_override:
            new_path = Path(folder_override) / new_filename
        else:
            new_path = orig_path.parent / new_filename
            
        self.output_values['path'] = str(new_path)
        self.mark_clean()
        return True, None
