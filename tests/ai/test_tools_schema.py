from synapse.ai.tools import TOOLS, TOOL_NAMES


def test_all_six_tools_present():
    assert TOOL_NAMES == (
        "generate_workflow",
        "modify_workflow",
        "write_python_script",
        "inspect_canvas",
        "explain_node",
        "read_node_output",
    )
    assert [t["name"] for t in TOOLS] == list(TOOL_NAMES)


def test_each_tool_has_required_keys():
    for t in TOOLS:
        assert isinstance(t.get("name"), str) and t["name"]
        assert isinstance(t.get("description"), str) and t["description"]
        schema = t.get("input_schema")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema


def test_write_python_script_requires_description():
    wps = next(t for t in TOOLS if t["name"] == "write_python_script")
    required = wps["input_schema"].get("required") or []
    assert "description" in required


def test_modify_workflow_operations_is_array():
    mw = next(t for t in TOOLS if t["name"] == "modify_workflow")
    ops = mw["input_schema"]["properties"]["operations"]
    assert ops["type"] == "array"
    item = ops.get("items", {})
    ops_enum = (item.get("properties", {}).get("op", {}).get("enum")
                or item.get("oneOf"))
    assert ops_enum is not None, "modify_workflow ops must be enumerated"


def test_read_node_output_requires_node_id():
    rno = next(t for t in TOOLS if t["name"] == "read_node_output")
    assert "node_id" in rno["input_schema"].get("required", [])
