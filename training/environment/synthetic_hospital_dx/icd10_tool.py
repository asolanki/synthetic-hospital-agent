"""Optional `icd10_search` tool: the public CMS ICD-10-CM 2025 code book (see icd10.py).

Off by default (`[env.taskset.task] icd10_tool = true` enables it), so official-harness results
are unchanged. Lookups run locally, never reach the simulator and do not spend EHR actions; they
use model turns like any other tool call. The index holds no benchmark labels.
"""

# No `from __future__ import annotations`: MCP builds the argument model from live annotations.
from typing import Annotated

import verifiers.v1 as vf
from pydantic import Field

from synthetic_hospital_dx import icd10


class ICD10Toolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "icd10"

    @vf.tool
    def search(self, query: Annotated[str, Field(
            description="A diagnosis name or clinical term, e.g. 'interstitial lung disease'")]) -> str:
        """Search the public ICD-10-CM 2025 code book (official titles, inclusion terms and the alphabetic index) and return the best-matching codes with their official titles. Use it to choose the ICD-10-CM code for each diagnosis before submitting. Lookups do not use your action budget."""
        return icd10.render(query, icd10.load().search(query))


if __name__ == "__main__":
    ICD10Toolset.run()
