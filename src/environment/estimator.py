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
    inference_mode: str = "freeform"  # Mode used: freeform, multiple_choice_N, structured
    selected_option: int | None = None  # For multiple choice: which option was selected (0-indexed)
    n_options: int | None = None  # For multiple choice: total number of options


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

    def infer_agent_objectives_multiple_choice(
        self,
        all_statements: list[Statement],
        agents: list[dict],
        world: World,
        n_choices: int = 4,
    ) -> dict[str, InferredAgentObjective]:
        """
        Infer each agent's objectives using multiple-choice format.

        Instead of freeform generation, the estimator selects from N candidate
        objectives (1 correct + N-1 distractors).

        Args:
            all_statements: All statements from the entire game
            agents: Agent info dicts (with id, interest, optionally value_function)
            world: The world (for property definitions)
            n_choices: Number of options to present (2, 4, 8, or 16)

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

        # Get property info for generating distractors
        property_values = {}
        for p in world.property_definitions:
            property_values[p.name] = p.possible_values

        # Infer objectives for each agent
        for agent_dict in agents:
            agent_id = agent_dict.get("id", "")
            stmts = statements_by_agent.get(agent_id, [])

            if not stmts:
                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal="No statements to analyze",
                    inferred_factors=[],
                    confidence=0,
                    reasoning="Agent made no statements",
                    evidence=[],
                    inference_mode=f"multiple_choice_{n_choices}",
                    selected_option=None,
                    n_options=n_choices,
                )
                continue

            # Generate candidate options (1 correct + n-1 distractors)
            correct_objective = self._format_ground_truth_objective(agent_dict)
            distractors = self._generate_distractor_objectives(
                agent_dict, agents, property_values, n_choices - 1
            )

            # Shuffle options
            import random
            options = [correct_objective] + distractors
            correct_idx = 0
            # Create a deterministic shuffle based on agent_id for reproducibility
            rng = random.Random(hash(agent_id))
            shuffled_indices = list(range(len(options)))
            rng.shuffle(shuffled_indices)
            options = [options[i] for i in shuffled_indices]
            correct_idx = shuffled_indices.index(0)  # Where the correct answer ended up

            # Format statements
            stmt_texts = [f"- {s.text}" for s in stmts]
            statements_text = "\n".join(stmt_texts)

            # Format options
            option_letters = "ABCDEFGHIJKLMNOP"
            options_text = "\n".join(
                f"{option_letters[i]}. {opt}" for i, opt in enumerate(options)
            )

            prompt = f"""You are analyzing the behavior of {agent_id} in a strategic game.

In this game, agents make statements to influence a judge's object selection decisions.
Each agent has a hidden objective (value function) that determines what objects they want selected.

{agent_id}'s statements throughout the game:
{statements_text}

Based on these statements, which of the following objectives best describes what {agent_id} wants?

{options_text}

Respond with JSON:
{{
    "selected": "A" or "B" or ... (the letter of your choice),
    "confidence": 0-100,
    "reasoning": "Why you chose this option based on the statements"
}}

Respond ONLY with the JSON object."""

            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )

                result = self._parse_inference_response(response.content[0].text)
                selected_letter = result.get("selected", "A").upper()
                selected_idx = option_letters.index(selected_letter) if selected_letter in option_letters else 0

                # Check if correct
                is_correct = (selected_idx == correct_idx)

                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal=options[selected_idx],
                    inferred_factors=[],  # Not applicable for multiple choice
                    confidence=result.get("confidence", 0),
                    reasoning=result.get("reasoning", ""),
                    evidence=[s.text for s in stmts[:2]],  # First 2 statements as evidence
                    inference_mode=f"multiple_choice_{n_choices}",
                    selected_option=selected_idx,
                    n_options=n_choices,
                )
            except Exception as e:
                logger.warning(f"Failed to infer objectives (MC) for {agent_id}: {e}")
                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal=f"Error inferring: {str(e)}",
                    inferred_factors=[],
                    confidence=0,
                    reasoning="Inference failed",
                    evidence=[],
                    inference_mode=f"multiple_choice_{n_choices}",
                    selected_option=None,
                    n_options=n_choices,
                )

        return inferences

    def _generate_distractor_objectives(
        self,
        agent_dict: dict,
        all_agents: list[dict],
        property_values: dict[str, list],
        n_distractors: int,
    ) -> list[str]:
        """Generate plausible but incorrect objective distractors.

        Strategy:
        1. Use other agents' actual objectives (if available)
        2. Generate random property combinations
        3. Flip conditions from the true objective
        """
        import random
        distractors = []
        rng = random.Random(hash(agent_dict.get("id", "")))

        # Strategy 1: Other agents' objectives
        for other in all_agents:
            if other.get("id") != agent_dict.get("id"):
                other_obj = self._format_ground_truth_objective(other)
                if other_obj and other_obj not in distractors:
                    distractors.append(other_obj)

        # Strategy 2: Random property combinations
        properties = list(property_values.keys())
        for _ in range(n_distractors * 2):  # Generate extra, then trim
            if len(distractors) >= n_distractors:
                break
            prop = rng.choice(properties)
            val = rng.choice(property_values[prop])
            if prop == "is_dangerous":
                distractor = f"Values objects that are: Object is {'dangerous' if val else 'not dangerous'}"
            else:
                distractor = f"Values objects that are: Object is {val}"

            # Add bonus info
            bonus = rng.choice([15, 20, 25, 30])
            distractor = f"{distractor} (+{bonus} bonus)"

            if distractor not in distractors:
                distractors.append(distractor)

        # Strategy 3: If we still need more, create compound distractors
        while len(distractors) < n_distractors:
            props = rng.sample(properties, min(2, len(properties)))
            vals = [rng.choice(property_values[p]) for p in props]
            distractor = f"Values objects that are: {' AND '.join(f'{p}={v}' for p, v in zip(props, vals))}"
            if distractor not in distractors:
                distractors.append(distractor)

        return distractors[:n_distractors]

    def infer_agent_objectives_structured(
        self,
        all_statements: list[Statement],
        agents: list[dict],
        world: World,
    ) -> dict[str, InferredAgentObjective]:
        """
        Infer each agent's objectives using structured factor selection.

        Instead of freeform text, the estimator selects from enumerated
        property=value pairs with confidence weights.

        Args:
            all_statements: All statements from the entire game
            agents: Agent info dicts
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

        # Build enumerated factor list
        factor_list = []
        for p in world.property_definitions:
            for v in p.possible_values:
                factor_list.append(f"{p.name}={v}")

        factors_text = "\n".join(f"{i+1}. {f}" for i, f in enumerate(factor_list))

        # Infer objectives for each agent
        for agent_dict in agents:
            agent_id = agent_dict.get("id", "")
            stmts = statements_by_agent.get(agent_id, [])

            if not stmts:
                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal="No statements to analyze",
                    inferred_factors=[],
                    confidence=0,
                    reasoning="Agent made no statements",
                    evidence=[],
                    inference_mode="structured",
                )
                continue

            # Format statements
            stmt_texts = [f"- {s.text}" for s in stmts]
            statements_text = "\n".join(stmt_texts)

            prompt = f"""You are analyzing the behavior of {agent_id} in a strategic game.

In this game, agents make statements to influence a judge's object selection decisions.
Each agent has a hidden objective that determines what objects they want selected.

{agent_id}'s statements throughout the game:
{statements_text}

Here are all possible factors an agent could value:
{factors_text}

Based on the statements, identify which factors this agent seems to prefer.
For each factor, assign a likelihood score from 0-100.

Respond with JSON:
{{
    "selected_factors": [
        {{"factor": "property=value", "likelihood": 0-100}},
        ...
    ],
    "primary_factor": "The single most likely factor",
    "confidence": 0-100,
    "reasoning": "Why you believe these are the agent's preferred factors"
}}

Include the top 3-5 most likely factors. Respond ONLY with the JSON object."""

            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}],
                )

                result = self._parse_inference_response(response.content[0].text)

                # Extract selected factors
                selected = result.get("selected_factors", [])
                inferred_factors = [
                    f.get("factor", "") for f in selected
                    if f.get("likelihood", 0) >= 50  # Only include high-confidence factors
                ]

                primary = result.get("primary_factor", "")
                if primary and primary not in inferred_factors:
                    inferred_factors.insert(0, primary)

                # Build goal description from factors
                if inferred_factors:
                    goal = f"Values objects with: {', '.join(inferred_factors[:3])}"
                else:
                    goal = "Unable to determine objective"

                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal=goal,
                    inferred_factors=inferred_factors,
                    confidence=result.get("confidence", 0),
                    reasoning=result.get("reasoning", ""),
                    evidence=[s.text for s in stmts[:2]],
                    inference_mode="structured",
                )
            except Exception as e:
                logger.warning(f"Failed to infer objectives (structured) for {agent_id}: {e}")
                inferences[agent_id] = InferredAgentObjective(
                    agent_id=agent_id,
                    inferred_goal=f"Error inferring: {str(e)}",
                    inferred_factors=[],
                    confidence=0,
                    reasoning="Inference failed",
                    evidence=[],
                    inference_mode="structured",
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
