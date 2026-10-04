import verifiers.v1 as vf


class EpisodeState(vf.State):
    """Per-rollout state shared by the task (host) and the EHR tool server."""

    episode_id: str | None = None
    actions: int = 0               # environment steps taken, as counted by the simulator
    tool_errors: int = 0           # observations that came back as {"error": ...}
    submitted: bool = False
    forced: bool = False           # submitted after the action budget was spent
    malformed: bool = False
    from_text: bool = False        # answer was written as text and submitted by the env (forced turn)
    nudges: int = 0                # "please take an action" messages sent after tool-less replies
    reward: float | None = None
    metrics: dict[str, float] = {}  # the scorer's numeric metrics for the submission
