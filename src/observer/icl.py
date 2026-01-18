"""ICL Observer for truth recovery experiments."""

import anthropic


ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit your True/False answer to the question",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "boolean"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "reasoning": {"type": "string"}
        },
        "required": ["answer", "confidence"]
    }
}


class ICLObserver:
    """Observer that uses in-context learning to recover truth from statements."""

    def __init__(self, model: str = "claude-haiku-4-5-20250514", client=None):
        """Initialize the ICL observer.

        Args:
            model: The Claude model to use for inference.
            client: Optional Anthropic client. If None, creates a new one.
        """
        self.model = model
        self.client = client or anthropic.Anthropic()
