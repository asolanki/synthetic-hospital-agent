"""The paper's EHR tools, served over MCP and forwarded one-to-one to the simulator's /env/step.

Names, descriptions and parameters mirror the function schemas that `/env/reset` hands out
(epic_sim/app/routers/agent.py TOOL_DEFINITIONS plus the patient_diagnosis submit schema in
eval/agents/prompts.py), so a model sees the same interface as in the paper's tool-use runs.
The simulator does the filtering (no assessment/plan sections, chart history instead of the
reference problem list), the budget accounting, and the scoring.

One difference from calling /env/step directly: arguments are validated against these
signatures before they reach the simulator, so a malformed call is rejected without costing
an action, and a submission whose lists are not lists of objects cannot crash the scorer.
"""

# No `from __future__ import annotations`: MCP builds each tool's argument model from the
# live annotations, and string annotations would not resolve in its namespace.
from typing import Annotated, Literal

import verifiers.v1 as vf
from pydantic import BaseModel, Field

from synthetic_hospital_dx import simulator
from synthetic_hospital_dx.state import EpisodeState

SUBMIT_TOOL = "submit_diagnosis"


class ActiveDiagnosis(BaseModel):
    icd10: str = Field(description="ICD-10-CM code (e.g., I21.01)")
    name: str = Field(description="Diagnosis display name")
    acuity: Literal["acute", "acute_on_chronic"]


class ChronicCondition(BaseModel):
    icd10: str = Field(description="ICD-10-CM code")
    name: str = Field(description="Condition display name")
    acuity: Literal["chronic"]


class EHRToolset(vf.Toolset[vf.ToolsetConfig, EpisodeState]):
    TOOL_PREFIX = None  # bare tool names, as in the paper

    async def _step(self, name: str, arguments: dict) -> str:
        state = self.state
        if state.episode_id is None:
            raise RuntimeError("no simulator episode for this rollout (task setup did not run)")
        if state.submitted:
            return "Episode already finished."
        out = await simulator.post("/env/step", {
            "episode_id": state.episode_id, "name": name,
            "arguments": {k: v for k, v in arguments.items() if v is not None}})
        state.actions = out["step"]
        observation = out["observation"]
        if isinstance(observation, dict) and observation.get("error"):
            state.tool_errors += 1
        if out["done"]:
            info = out["info"]
            state.submitted = True
            state.forced = bool(info.get("forced"))
            state.malformed = bool(info.get("malformed"))
            state.reward = float(out["reward"] or 0.0)
            state.metrics = {k: float(v) for k, v in (info.get("metrics") or {}).items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)}
        return out["observation_text"]

    @vf.tool
    async def search_patients(self,
                              name: Annotated[str | None, Field(description="Patient name to search for")] = None,
                              gender: Literal["male", "female"] | None = None,
                              mrn: Annotated[str | None, Field(description="Patient ID / MRN")] = None) -> str:
        """Search for patients by name, gender, or MRN (patient ID)."""
        return await self._step("search_patients", {"name": name, "gender": gender, "mrn": mrn})

    @vf.tool
    async def open_chart(self, patient_id: int) -> str:
        """Open a patient's chart. Returns demographics, active problems, recent encounters, medications, and allergies."""
        return await self._step("open_chart", {"patient_id": patient_id})

    @vf.tool
    async def view_encounters(self, patient_id: int) -> str:
        """List all encounters for a patient in chronological order."""
        return await self._step("view_encounters", {"patient_id": patient_id})

    @vf.tool
    async def view_encounter_detail(self, encounter_id: int) -> str:
        """View a specific encounter with all available EHR sections (filtered by role)."""
        return await self._step("view_encounter_detail", {"encounter_id": encounter_id})

    @vf.tool
    async def view_section(self, section_id: int) -> str:
        """View a single EHR section by its ID."""
        return await self._step("view_section", {"section_id": section_id})

    @vf.tool
    async def view_results(self, patient_id: int, result_type: Literal["labs", "imaging", "pathology"]) -> str:
        """View lab, imaging, or pathology results for a patient."""
        return await self._step("view_results", {"patient_id": patient_id, "result_type": result_type})

    @vf.tool
    async def search_chart(self, patient_id: int,
                           query: Annotated[str, Field(description="Search terms (e.g., 'chest pain', 'hemoglobin')")]) -> str:
        """Search across all EHR sections for a patient using keywords or clinical terms."""
        return await self._step("search_chart", {"patient_id": patient_id, "query": query})

    @vf.tool
    async def view_problem_list(self, patient_id: int) -> str:
        """View the active problem list (diagnoses) for a patient."""
        return await self._step("view_problem_list", {"patient_id": patient_id})

    @vf.tool
    async def view_medications(self, patient_id: int) -> str:
        """View current and historical medications for a patient."""
        return await self._step("view_medications", {"patient_id": patient_id})

    @vf.tool
    async def submit_diagnosis(
            self,
            active_diagnoses: Annotated[list[ActiveDiagnosis], Field(description="Active diagnoses (acute or acute_on_chronic)")],
            chronic_conditions: Annotated[list[ChronicCondition], Field(description="Chronic conditions")]) -> str:
        """Submit your complete diagnosis list for this patient. Include ALL active diagnoses and chronic conditions you identified, each with an ICD-10-CM code, a display name, and acuity classification."""
        return await self._step(SUBMIT_TOOL, {
            "active_diagnoses": [d.model_dump() for d in active_diagnoses],
            "chronic_conditions": [d.model_dump() for d in chronic_conditions]})


if __name__ == "__main__":
    EHRToolset.run()
