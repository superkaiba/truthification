"""Prompt templates for ICL experiments."""

from enum import Enum


class Condition(Enum):
    """Experimental conditions for ICL."""
    NO_IDS = "no_ids"
    IDS_ONLY = "ids_only"
    IDS_AND_TASKS = "ids_and_tasks"
    ORACLE_RELIABILITY = "oracle_reliability"
    ORACLE_RELATIONSHIPS = "oracle_relationships"
    ORACLE_TRUTH_LABELS = "oracle_truth_labels"


class PromptBuilder:
    """Builds prompts for ICL experiments based on condition."""

    def __init__(self, condition: Condition):
        self.condition = condition

    def format_evidence(
        self,
        statements: list[dict],
        agent_metadata: dict,
        object_id: str,
        property_name: str,
    ) -> str:
        """Format evidence block for the given condition."""
        # Filter to relevant statements
        relevant = [
            s for s in statements
            if s["object_id"] == object_id and s["property_name"] == property_name
        ]

        if not relevant:
            return "No statements available about this object."

        lines = []
        for stmt in relevant:
            line = self._format_statement(stmt, agent_metadata)
            lines.append(line)

        return "\n".join(lines)

    def _format_statement(self, stmt: dict, agent_metadata: dict) -> str:
        """Format a single statement based on condition."""
        text = stmt["text"]
        agent_id = stmt["agent_id"]

        if self.condition == Condition.NO_IDS:
            return f"- {text}"

        elif self.condition == Condition.IDS_ONLY:
            return f"- {agent_id}: {text}"

        elif self.condition == Condition.IDS_AND_TASKS:
            task_desc = agent_metadata.get(agent_id, {}).get("task", {}).get("description", "unknown task")
            return f"- {agent_id} (task: {task_desc}): {text}"

        elif self.condition == Condition.ORACLE_RELIABILITY:
            # Compute reliability from metadata (% of truthful statements)
            reliability = self._compute_agent_reliability(agent_id, agent_metadata)
            return f"- {agent_id} (reliability: {reliability}%): {text}"

        elif self.condition == Condition.ORACLE_RELATIONSHIPS:
            task_desc = agent_metadata.get(agent_id, {}).get("task", {}).get("description", "unknown")
            is_truthful = stmt.get("is_truthful", True)
            truth_desc = "tells truth" if is_truthful else "lies"
            prop = stmt.get("property_name", "properties")
            return f"- {agent_id} ({truth_desc} about {prop}): {text}"

        elif self.condition == Condition.ORACLE_TRUTH_LABELS:
            label = "TRUE" if stmt.get("is_truthful", True) else "FALSE"
            return f"- {agent_id} [{label}]: {text}"

        return f"- {text}"

    def _compute_agent_reliability(self, agent_id: str, agent_metadata: dict) -> int:
        """Compute reliability percentage for an agent."""
        # Read reliability from metadata if available
        if agent_id in agent_metadata and "reliability" in agent_metadata[agent_id]:
            return agent_metadata[agent_id]["reliability"]
        # Default fallback
        return 50

    def build_system_prompt(self) -> str:
        """Build the system prompt for the observer."""
        return """You are an observer trying to determine ground truth about objects in a world.
You will be given statements from various agents about object properties.
Some agents may be reliable, others may lie.
Your task is to determine the truth based on the evidence provided.

Use the submit_answer tool to provide your answer."""

    def build_query(
        self,
        object_id: str,
        property_name: str,
        property_value: str,
        evidence: str,
    ) -> str:
        """Build the full query prompt."""
        return f"""Here are statements about {object_id}:

{evidence}

Question: Is {object_id} {property_value} (for {property_name})?

Analyze the evidence and submit your answer using the submit_answer tool."""
