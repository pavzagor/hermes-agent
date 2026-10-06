"""Rendered guidance routes by operation/branch, not incidental topical overlap.

Exercise the shared production assembler and real auto-load/config lookup; no
source-text assertions or mocked skill index/loaders.
"""

from types import SimpleNamespace

import pytest

from agent.prompt_builder import (
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    clear_skills_system_prompt_cache,
)
from agent.system_prompt import build_system_prompt_parts


@pytest.fixture
def guidance_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    for name, description in (
        ("hermes-agent", "Configure Hermes Agent"),
        ("operation-router", "Route the requested operation"),
        ("proof-branch", "Verify external changes"),
        ("adjacent-topic", "Background on a related subject"),
    ):
        directory = tmp_path / "skills" / "operations" / name
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n"
            f"# {name}\nComplete procedure for {name}.\n",
            encoding="utf-8",
        )
    clear_skills_system_prompt_cache(clear_snapshot=True)
    yield tmp_path
    clear_skills_system_prompt_cache(clear_snapshot=True)


def _agent(home, oneshot):
    names = {"skill_view", "skills_list", "terminal"}
    if not oneshot:
        names.add("skill_manage")
    return SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=names,
        _task_completion_guidance=True,
        _tool_use_enforcement=True,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _session_db=SimpleNamespace(db_path=str(home / "state.db")),
        model="test-model",
        provider="test",
        platform="cli",
        pass_session_id=False,
        session_id="operation-guidance-test",
    )


@pytest.mark.parametrize("oneshot", [False, True], ids=["interactive", "oneshot"])
def test_rendered_selection_requires_operation_and_concrete_branches(
    guidance_env, monkeypatch, oneshot
):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1" if oneshot else "0")
    parts = build_system_prompt_parts(_agent(guidance_env, oneshot))
    policy, index_tail = parts["volatile"].split("<available_skills>", 1)
    index = index_tail.split("</available_skills>", 1)[0]

    # Selection coaching must not filter discovery or suppress user conventions.
    for name in ("operation-router", "proof-branch", "adjacent-topic", "hermes-agent"):
        assert f"- {name}:" in index
    assert "governing operation" in policy
    assert "action, safety, or proof" in policy
    assert "topical overlap alone" in policy
    assert "uncertain" in policy and "expand discovery" in policy
    assert "MUST" in policy and "skill_view(name)" in policy
    assert "user's conventions and quality standards" in policy
    assert "tool-use, authorization, scope, or verification" in policy
    assert "even partially relevant" not in policy
    assert "Err on the side of loading" not in policy
    assert "Do not load general process skills" not in policy
    assert TOOL_USE_ENFORCEMENT_GUIDANCE in parts["stable"]
    assert "skill_view(name='hermes-agent')" in parts["stable"]
    assert "authoritative reference" in parts["stable"]
    assert ("offer to save as a skill" in policy) is not oneshot
    assert ("skill_manage" in policy) is not oneshot


@pytest.mark.parametrize("oneshot", [False, True], ids=["interactive", "oneshot"])
def test_full_current_auto_loaded_body_counts_loaded_but_pruning_requires_reload(
    guidance_env, monkeypatch, oneshot
):
    (guidance_env / "config.yaml").write_text(
        "skills:\n  auto_load: [hermes-agent]\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1" if oneshot else "0")
    agent = _agent(guidance_env, oneshot)
    parts = build_system_prompt_parts(agent)
    policy = parts["volatile"].split("<available_skills>", 1)[0]
    stable = parts["stable"]

    assert "Complete procedure for hermes-agent." in stable
    assert agent._auto_load_skills_result[1] == ["hermes-agent"]
    assert "full current body" in policy and "auto-loaded" in policy
    assert "counts as loaded" in policy
    assert "do not reload" in policy
    assert "summary" in policy and "[SKILL_PRUNED]" in policy
    assert "reload" in policy and "before relying on it" in policy
    # The standalone Hermes-help rule must not contradict current-body reuse.
    assert "full current body" in stable and "reuse" in stable
    assert "[SKILL_PRUNED]" in stable and "reload" in stable
    # Config/skill mutation cannot change the frozen auto-load prefix mid-session.
    (guidance_env / "config.yaml").write_text("skills:\n  auto_load: []\n", encoding="utf-8")
    rebuilt = build_system_prompt_parts(agent)
    assert rebuilt["stable"] == stable
