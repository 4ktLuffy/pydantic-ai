"""Workspace-independent researcher configuration."""

from pathlib import Path

import pytest

# `Researcher` builds `WebFetch(local=True)`, which needs the `web-fetch` extra.
pytest.importorskip('markdownify')

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.researcher import Researcher
from pydantic_ai_harness.tool_output_limits import LocalFileStore


async def test_researcher_with_local_store_needs_no_workspace(tmp_path: Path) -> None:
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[Researcher(store=LocalFileStore(tmp_path))])
    # TestModel rejects native web tools, but only after the workspace preflight succeeds.
    with pytest.raises(UserError, match='TestModel does not support built-in tools'):
        await agent.run('research')


@pytest.mark.parametrize('engine', ['temporal', 'prefect'])
def test_researcher_constructs_with_durability(engine: str) -> None:
    """Its instructions capability needs an `id` for durable execution, like Coder's (#9553)."""
    if engine == 'temporal':
        pytest.importorskip('temporalio')
        from pydantic_ai.durable_exec.temporal import TemporalDurability as Durability
    else:
        pytest.importorskip('prefect')
        from pydantic_ai.durable_exec.prefect import PrefectDurability as Durability

    Agent(TestModel(), name=f'researcher-{engine}', capabilities=[Researcher(), Durability()])
