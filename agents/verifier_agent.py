# verifier_agent.py

from typing import Callable, Optional


def _noop(_: str) -> None:
    pass


class VerifierAgent:
    def __init__(self, log: Optional[Callable[[str], None]] = None):
        self._log = log or _noop

    def __call__(self, state: dict) -> dict:
        self._log("verifier: checking output")
        result = state.get("result")
        retries = state.get("retries", 0)

        if not isinstance(result, dict):
            return self._retry_or_end(retries, "Result missing or not a JSON object.")

        conf = result.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            return self._retry_or_end(
                retries,
                "Set 'confidence' to a number between 0 and 1 and keep required fields present."
            )

        return {"next": "end"}

    def _retry_or_end(self, retries: int, feedback: str) -> dict:
        if retries < 1:
            self._log("verifier: requesting retry")
            return {"retries": retries + 1, "correction": feedback, "next": "worker"}

        self._log("verifier: retries exhausted; marking needs_review")
        return {"next": "end", "needs_review": True}