from synapse.ai.prompts import BASE_SYSTEM_PROMPT, WRITE_PYTHON_SCRIPT_SUBPROMPT


def test_base_prompt_mentions_synapse_and_markdown():
    assert "Synapse" in BASE_SYSTEM_PROMPT
    assert "markdown" in BASE_SYSTEM_PROMPT.lower()


def test_base_prompt_forbids_raw_json_dumps():
    assert "raw JSON" in BASE_SYSTEM_PROMPT


def test_base_prompt_mentions_tools_and_clarifying_questions():
    low = BASE_SYSTEM_PROMPT.lower()
    assert "tool" in low
    assert "clarifying" in low or "clarify" in low


def test_write_python_script_subprompt_references_in_out_vars():
    assert "in_1" in WRITE_PYTHON_SCRIPT_SUBPROMPT
    assert "out_1" in WRITE_PYTHON_SCRIPT_SUBPROMPT


def test_write_python_script_subprompt_forbids_fences():
    low = WRITE_PYTHON_SCRIPT_SUBPROMPT.lower()
    assert "fence" in low or "no markdown" in low or "```" in WRITE_PYTHON_SCRIPT_SUBPROMPT
