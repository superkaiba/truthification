"""Prompt templates for ICL experiments."""

from enum import Enum


class Condition(Enum):
    """Experimental conditions for ICL."""
    # Isolated query conditions (only see statements about queried object/property)
    NO_IDS = "no_ids"
    IDS_ONLY = "ids_only"
    IDS_AND_TASKS = "ids_and_tasks"
    ORACLE_RELIABILITY = "oracle_reliability"
    ORACLE_AGENT_TYPE = "oracle_agent_type"  # Shows cooperative/adversarial
    ORACLE_TRUTH_LABELS = "oracle_truth_labels"
    # Full context conditions (see ALL statements for every query)
    FULL_CONTEXT_NO_IDS = "full_context_no_ids"
    FULL_CONTEXT_IDS = "full_context_ids"
    FULL_CONTEXT_IDS_TASKS = "full_context_ids_tasks"


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
            reliability = self._get_agent_reliability(agent_id, agent_metadata)
            return f"- {agent_id} (reliability: {reliability}%): {text}"

        elif self.condition == Condition.ORACLE_AGENT_TYPE:
            agent_type = agent_metadata.get(agent_id, {}).get("agent_type", "unknown")
            type_desc = "helpful" if agent_type == "cooperative" else "deceptive"
            return f"- {agent_id} ({type_desc}): {text}"

        elif self.condition == Condition.ORACLE_TRUTH_LABELS:
            label = "TRUE" if stmt.get("is_truthful", True) else "FALSE"
            return f"- {agent_id} [{label}]: {text}"

        return f"- {text}"

    def _get_agent_reliability(self, agent_id: str, agent_metadata: dict) -> int:
        """Get reliability percentage for an agent."""
        if agent_id in agent_metadata and "reliability" in agent_metadata[agent_id]:
            return agent_metadata[agent_id]["reliability"]
        return 50

    def build_system_prompt(self) -> str:
        """Build the system prompt for the observer."""
        return """You are an observer trying to determine ground truth about objects in a world.
You will be given statements from various agents about object properties.
Some agents may be reliable, others may provide incorrect information.
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

Question: Is {object_id}'s {property_name} equal to {property_value}?

Analyze the evidence and submit your answer using the submit_answer tool."""


class FullContextPromptBuilder:
    """Builds prompts showing ALL statements for full-context ICL experiments."""

    def __init__(self, condition: Condition):
        self.condition = condition

    def format_full_evidence(
        self,
        statements: list[dict],
        agent_metadata: dict,
    ) -> str:
        """Format ALL statements, grouped by agent."""
        from collections import defaultdict

        by_agent: dict[str, list[dict]] = defaultdict(list)
        for stmt in statements:
            by_agent[stmt["agent_id"]].append(stmt)

        sections = []
        for agent_id in sorted(by_agent.keys()):
            agent_stmts = by_agent[agent_id]

            # Build header based on condition
            if self.condition == Condition.FULL_CONTEXT_IDS_TASKS:
                task_desc = agent_metadata.get(agent_id, {}).get("task", {}).get(
                    "description", "unknown task"
                )
                header = f"{agent_id} (task: {task_desc}):"
            elif self.condition == Condition.FULL_CONTEXT_IDS:
                header = f"{agent_id}:"
            else:  # FULL_CONTEXT_NO_IDS
                header = "Statements:"

            # Format statements
            stmt_lines = []
            for stmt in agent_stmts:
                stmt_lines.append(f"  - {stmt['text']}")

            if self.condition == Condition.FULL_CONTEXT_NO_IDS:
                # For no_ids, don't group by agent - just list all statements
                sections.extend(stmt_lines)
            else:
                sections.append(f"{header}\n" + "\n".join(stmt_lines))

        if self.condition == Condition.FULL_CONTEXT_NO_IDS:
            return "\n".join(sections)
        return "\n\n".join(sections)

    def build_system_prompt(self) -> str:
        """Build the system prompt for full-context observer."""
        return """You are an observer trying to determine ground truth about objects in a world.
You will be given statements from various agents about object properties.
Some agents are RELIABLE and always tell the truth.
Other agents are UNRELIABLE and consistently provide false information.

IMPORTANT: Analyze patterns across ALL statements to identify which agents are reliable.
Look for:
- Agents who contradict other agents frequently
- Consistency patterns within each agent's statements
- Whether an agent's task might bias their observations

Use the submit_answer tool to provide your answer."""

    def build_query(
        self,
        full_evidence: str,
        object_id: str,
        property_name: str,
        property_value: str,
    ) -> str:
        """Build the full query prompt with all evidence."""
        return f"""=== ALL COLLECTED STATEMENTS ===

{full_evidence}

=== QUERY ===
Based on analyzing ALL the statements above, answer:
Is {object_id}'s {property_name} equal to {property_value}?

Think about which agents appear reliable based on consistency patterns, then submit your answer."""
