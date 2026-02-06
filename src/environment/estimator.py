"""External Estimator LLM that observes the game and infers truth without participating."""

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .agent import Statement
from .simulation import OracleQuery
from .world import ValueRule, World


@dataclass
class Estimator:
    """
    External LLM that observes the game and tries to infer truth.

    The estimator:
    - Is a passive observer (sees same info as observer based on condition)
    - Does not influence the game
    - Tries to infer properties and value rule from statements
    """

    model: str = "claude-sonnet-4-20250514"
    condition: str = "blind"  # Same as observer: "blind", "ids", "interests"
    _client: anthropic.Anthropic | None = field(default=None, repr=False)

    def __post_init__(self):
        if self._client is None:
            self._client = anthropic.Anthropic()

    def analyze_round(
        self,
        statements: list[Statement],
        oracle_results: list[OracleQuery],
        prior_beliefs: dict,
        agents: list[dict],
    ) -> dict:
        """
        Analyze a round and update beliefs.

        Args:
            statements: Statements made during the round
            oracle_results: Oracle query results from this round
            prior_beliefs: Estimator's beliefs before this round
            agents: Agent info (for "interests" condition)

        Returns:
            Updated beliefs dict with property_beliefs, value_rule_guess, reasoning
        """
        # Format statements based on condition
        stmt_text = self._format_statements(statements, agents)
        oracle_text = self._format_oracle_results(oracle_results)

        prompt = f"""You are analyzing a game where agents make claims about objects.
Your task: Infer what is TRUE about the objects based on the statements.

{stmt_text}

{oracle_text}

Your prior beliefs: {json.dumps(prior_beliefs, indent=2) if prior_beliefs else "None yet"}

Based on ALL information, provide your updated beliefs as JSON:
{{
  "property_beliefs": {{"object_id": {{"property": "value", ...}}, ...}},
  "value_rule_guess": {{"description": "...", "confidence": 0-100, "key_factors": [...]}},
  "reasoning": "..."
}}

Important: Only include property beliefs where you have some evidence.
Respond ONLY with the JSON object, no other text."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_response(response.content[0].text)

    def _format_statements(
        self,
        statements: list[Statement],
        agents: list[dict],
    ) -> str:
        """Format statements based on condition (like observer)."""
        if not statements:
            return "No statements yet."

        lines = ["== Statements =="]
        for stmt in statements:
            if self.condition == "blind":
                lines.append(f"- {stmt.text}")
            elif self.condition == "ids":
                lines.append(f"- {stmt.agent_id}: {stmt.text}")
            else:  # "interests"
                # Find agent's interest
                agent_info = next(
                    (a for a in agents if a.get("id") == stmt.agent_id), None
                )
                if agent_info:
                    interest_desc = agent_info.get("interest", {}).get(
                        "description", "unknown"
                    )
                    lines.append(
                        f"- {stmt.agent_id} (wants: {interest_desc}): {stmt.text}"
                    )
                else:
                    lines.append(f"- {stmt.agent_id}: {stmt.text}")

        return "\n".join(lines)

    def _format_oracle_results(self, oracle_results: list[OracleQuery]) -> str:
        """Format oracle results for the prompt."""
        if not oracle_results:
            return "No oracle queries this round."

        lines = ["== Oracle Results =="]
        for query in oracle_results:
            if query.query_type == "value":
                lines.append(f"- {query.object_id} has value: {query.result}")
            else:
                lines.append(
                    f"- {query.object_id}'s {query.property_name}: {query.result}"
                )

        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict:
        """Parse LLM response into beliefs dict."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        # Return empty beliefs on parse failure
        return {
            "property_beliefs": {},
            "value_rule_guess": {
                "description": "Unable to parse",
                "confidence": 0,
                "key_factors": [],
            },
            "reasoning": f"Parse error from: {text[:200]}",
        }

    def compute_accuracy(self, world: World, value_rule: ValueRule) -> dict:
        """
        Compute accuracy metrics against ground truth.

        This should be called after the game is complete with accumulated beliefs.
        """
        # This will be called from HiddenValueGame after the game ends
        # The beliefs are tracked in HiddenValueGame.estimator_beliefs
        return {}

    def compute_property_accuracy(
        self,
        beliefs: dict,
        world: World,
    ) -> float:
        """Compute accuracy of property beliefs vs ground truth.

        Computes accuracy over ALL properties of ALL objects, not just
        the properties the estimator stated beliefs about. This gives
        a meaningful metric that penalizes missing knowledge.

        Returns: correct_beliefs / total_properties
        """
        property_beliefs = beliefs.get("property_beliefs", {})

        correct = 0
        total = 0

        # Iterate over ALL objects and ALL their properties
        for obj_id in world.list_objects():
            obj = world.get_object(obj_id)
            if obj is None:
                continue

            believed_props = property_beliefs.get(obj_id, {})

            # Check each property of this object
            for prop_def in world.property_definitions:
                prop_name = prop_def.name
                true_value = obj.get_property(prop_name)
                if true_value is None:
                    continue

                total += 1

                # Check if estimator has a belief about this property
                if prop_name in believed_props:
                    believed_value = believed_props[prop_name]
                    if str(believed_value).lower() == str(true_value).lower():
                        correct += 1

        return correct / total if total > 0 else 0.0

    def compute_rule_inference_accuracy(
        self,
        beliefs: dict,
        value_rule: ValueRule,
        world: World,
    ) -> float:
        """
        Compute how well estimator inferred the value rule.

        Approach: Check if inferred key factors match actual rule conditions.
        """
        rule_guess = beliefs.get("value_rule_guess", {})
        if not rule_guess:
            return 0.0

        # Extract property names that actually matter from the true rule
        true_factors = set()
        for condition in value_rule.conditions:
            desc_lower = condition.description.lower()
            for prop in world.property_definitions:
                if prop.name.lower() in desc_lower:
                    true_factors.add(prop.name.lower())

        # Compare to inferred factors
        inferred_factors = set(f.lower() for f in rule_guess.get("key_factors", []))

        if not true_factors:
            return 1.0 if not inferred_factors else 0.0

        # Compute F1-like score
        if not inferred_factors:
            return 0.0

        intersection = true_factors & inferred_factors
        precision = len(intersection) / len(inferred_factors)
        recall = len(intersection) / len(true_factors)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1
