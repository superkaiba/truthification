"""ICL Observer for truth recovery experiments."""

import anthropic
from dataclasses import dataclass


@dataclass
class ObserverResponse:
    """Response from the ICL observer."""
    answer: bool
    confidence: int
    reasoning: str | None = None


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

    def query(
        self,
        evidence: str,
        object_id: str,
        property_name: str,
        property_value: str,
    ) -> ObserverResponse:
        """
        Query the observer about a property value.

        Args:
            evidence: Formatted evidence string
            object_id: The object being queried
            property_name: The property being queried
            property_value: The claimed value to verify

        Returns:
            ObserverResponse with answer, confidence, and reasoning
        """
        system_prompt = """You are an observer trying to determine ground truth about objects in a world.
You will be given statements from various agents about object properties.
Some agents may be reliable, others may lie.
Your task is to determine the truth based on the evidence provided.

Use the submit_answer tool to provide your answer."""

        user_prompt = f"""Here are statements about {object_id}:

{evidence}

Question: Is {object_id} {property_value} (for {property_name})?

Analyze the evidence and submit your answer using the submit_answer tool."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt,
            tools=[ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "submit_answer"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract tool use response
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "submit_answer":
                return ObserverResponse(
                    answer=block.input["answer"],
                    confidence=block.input["confidence"],
                    reasoning=block.input.get("reasoning"),
                )

        # Fallback if no tool use (shouldn't happen with tool_choice)
        raise ValueError("No tool use response from model")
