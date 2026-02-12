"""External Estimator LLM that observes the game and infers truth without participating."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

from .agent import AgentValueFunction, Statement
from .simulation import OracleQuery
from .world import ValueRule, World


@dataclass
class InferredAgentObjective:
    """Estimator's inference about what an agent wants."""
    agent_id: str
    inferred_goal: str  # Natural language description of inferred goal
    inferred_factors: list[str]  # Properties/values the agent seems to prefer
    confidence: int  # 0-100
    reasoning: str  # Why the estimator thinks this
    evidence: list[str]  # Key statements that support this inference


@dataclass
class ObjectiveInferenceResult:
    """Results of inferring all agents' objectives."""
    agent_inferences: dict[str, InferredAgentObjective]  # agent_id -> inference
    evaluation_scores: dict[str, float]  # agent_id -> LLM judge score (0-1)
    overall_score: float  # Average score across agents


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
        world: World | None = None,
    ) -> dict:
        """
        Analyze a round and update beliefs.

        Args:
            statements: Statements made during the round
            oracle_results: Oracle query results from this round
            prior_beliefs: Estimator's beliefs before this round
            agents: Agent info (for "interests" condition)
            world: World object for property definitions and object list

        Returns:
            Updated beliefs dict with property_beliefs, value_rule_guess, reasoning
        """
        # Format statements based on condition
        stmt_text = self._format_statements(statements, agents)
        oracle_text = self._format_oracle_results(oracle_results)
        world_text = self._format_world_info(world) if world else ""

        prompt = f"""You are analyzing a game where agents make claims about objects.
Your task: Infer what is TRUE about the objects based on the statements.

{world_text}

{stmt_text}

{oracle_text}

Your prior beliefs: {json.dumps(prior_beliefs, indent=2) if prior_beliefs else "None yet"}

Based on ALL information, provide your updated beliefs as JSON:
{{
  "property_beliefs": {{"object_id": {{"property": "value", ...}}, ...}},
  "value_rule_guess": {{"description": "...", "confidence": 0-100, "key_factors": [...]}},
  "reasoning": "..."
}}

IMPORTANT: You MUST make a prediction for EVERY property of EVERY object.
- If you have evidence, use it
- If you have no evidence, make your best guess based on agent credibility patterns or guess randomly
- Do NOT leave any properties blank - you must predict all {len(world.list_objects()) if world else 'N'} objects x {len(world.property_definitions) if world else 'N'} properties
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

    def _format_world_info(self, world: World) -> str:
        """Format world information for the prompt."""
        lines = ["== World Information =="]

        # List all objects
        objects = world.list_objects()
        lines.append(f"Objects: {', '.join(objects)}")

        # List all properties and their possible values
        lines.append("\nProperties and possible values:")
        for prop_def in world.property_definitions:
            lines.append(f"- {prop_def.name}: {prop_def.possible_values}")

        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict:
        """Parse LLM response into beliefs dict."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse estimator response: {e}. Response: {text[:200]}")

        # Return empty beliefs on parse failure
        logger.warning("Using fallback empty beliefs for estimator")
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

    def infer_agent_objectives(
        self,
        all_statements: list[Statement],
        agents: list[dict],
        world: World,
    ) -> dict[str, InferredAgentObjective]:
        """
        Infer each agent's objectives from their behavior.

        Analyzes all statements made by each agent across the game to infer
        what they're trying to achieve (their value function).

        Args:
            all_statements: All statements from the entire game
            agents: Agent info dicts (with id, interest, optionally value_function)
            world: The world (for property definitions)

        Returns:
            Dict mapping agent_id -> InferredAgentObjective
        """
        inferences = {}

        # Group statements by agent
        statements_by_agent: dict[str, list[Statement]] = {}
        for stmt in all_statements:
            agent_id = stmt.agent_id
            if agent_id not in statements_by_agent:
                statements_by_agent[agent_id] = []
            statements_by_agent[agent_id].append(stmt)

        # Get property names for context
        property_names = [p.name for p in world.property_definitions]
        property_values = {}
        for p in world.property_definitions:
            property_values[p.name] = p.possible_values

        # Infer objectives for each agent
        for agent_id, stmts in statements_by_agent.items():
            # Format statements
            stmt_texts = [f"- {s.text}" for s in stmts]
            statements_text = "\n".join(stmt_texts)

            prompt = f"""You are analyzing the behavior of {agent_id} in a strategic game.

In this game, agents make statements to influence a judge's object selection decisions.
Each agent has a hidden objective (value function) that determines what objects they want selected.

{agent_id}'s statements throughout the game:
{statements_text}

Available object properties: {property_names}
Property values: {json.dumps(property_values, indent=2)}

Based on {agent_id}'s statements, infer what their objective is:
- What properties/values do they seem to prefer?
- What objects do they advocate for or against?
- What patterns in their claims suggest their goal?

Respond with JSON:
{{
    "inferred_goal": "Natural language description of what this agent wants",
    "inferred_factors": ["property=value", ...],  // e.g., ["color=blue", "size=large"]
    "confidence": 0-100,
    "reasoning": "Why you believe this based on their behavior",
    "key_evidence": ["Most telling statement 1", "Most telling statement 2"]
}}

Respond ONLY with the JSON object."""

            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )

                result = self._parse_inference_response(response.content[0].text)

                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal=result.get("inferred_goal", "Unknown"),
                    inferred_factors=result.get("inferred_factors", []),
                    confidence=result.get("confidence", 0),
                    reasoning=result.get("reasoning", ""),
                    evidence=result.get("key_evidence", []),
                )
            except Exception as e:
                logger.warning(f"Failed to infer objectives for {agent_id}: {e}")
                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal=f"Error inferring: {str(e)}",
                    inferred_factors=[],
                    confidence=0,
                    reasoning="Inference failed",
                    evidence=[],
                )

        return inferences

    def _parse_inference_response(self, text: str) -> dict:
        """Parse LLM response for objective inference."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse inference response: {e}. Response: {text[:200]}")
        return {}

    def evaluate_objective_inference(
        self,
        inferences: dict[str, InferredAgentObjective],
        agents: list[dict],
        evaluator_model: str | None = None,
    ) -> ObjectiveInferenceResult:
        """
        Use an LLM judge to evaluate how well inferred objectives match ground truth.

        Since agent value functions are expressed in different formats (natural language
        inference vs structured conditions), we use an LLM to judge semantic similarity.

        Args:
            inferences: Dict of agent_id -> InferredAgentObjective
            agents: Agent info dicts with ground truth (interest and/or value_function)
            evaluator_model: Model to use for evaluation (defaults to self.model)

        Returns:
            ObjectiveInferenceResult with per-agent scores
        """
        model = evaluator_model or self.model
        scores = {}

        for agent_dict in agents:
            agent_id = agent_dict.get("id", "")
            if agent_id not in inferences:
                scores[agent_id] = 0.0
                continue

            inference = inferences[agent_id]

            # Get ground truth objective
            ground_truth = self._format_ground_truth_objective(agent_dict)

            # Use LLM to evaluate match
            prompt = f"""You are evaluating how well an inferred agent objective matches the true objective.

GROUND TRUTH (what the agent actually wants):
{ground_truth}

INFERRED OBJECTIVE (what was inferred from behavior):
Goal: {inference.inferred_goal}
Factors: {inference.inferred_factors}
Reasoning: {inference.reasoning}

Rate the match on a scale of 0.0 to 1.0:
- 1.0 = Perfect match - inferred objective captures the essence of the true objective
- 0.7-0.9 = Good match - key factors identified, minor details missed
- 0.4-0.6 = Partial match - some aspects correct, some wrong
- 0.1-0.3 = Poor match - mostly incorrect but some overlap
- 0.0 = No match - completely wrong inference

Consider:
1. Do the inferred factors align with the true objective's conditions?
2. Does the inferred goal description capture what the agent wants?
3. Is the inference specific enough to be useful?

Respond with JSON:
{{
    "score": 0.0-1.0,
    "explanation": "Brief explanation of the rating"
}}

Respond ONLY with the JSON object."""

            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )

                result = self._parse_inference_response(response.content[0].text)
                scores[agent_id] = float(result.get("score", 0.0))
            except Exception as e:
                logger.warning(f"Failed to evaluate objective inference for {agent_id}: {e}")
                scores[agent_id] = 0.0

        # Compute overall score
        overall = sum(scores.values()) / len(scores) if scores else 0.0

        return ObjectiveInferenceResult(
            agent_inferences=inferences,
            evaluation_scores=scores,
            overall_score=overall,
        )

    def _format_ground_truth_objective(self, agent_dict: dict) -> str:
        """Format an agent's ground truth objective for evaluation."""
        lines = []

        # Check for value function first (complex objectives)
        if "value_function" in agent_dict and agent_dict["value_function"]:
            vf = agent_dict["value_function"]
            lines.append(f"Name: {vf.get('name', 'N/A')}")
            lines.append(f"Description: {vf.get('description', 'N/A')}")
            conditions = vf.get("conditions", [])
            if conditions:
                lines.append("Conditions:")
                for cond in conditions:
                    lines.append(f"  - {cond.get('description', 'N/A')}: {cond.get('bonus', 0):+d}")
        else:
            # Fall back to simple interest
            interest = agent_dict.get("interest", {})
            lines.append(f"Target: {interest.get('target_condition', 'N/A')}")
            lines.append(f"Description: {interest.get('description', 'N/A')}")

        return "\n".join(lines)
