#!/usr/bin/env python
"""Streamlit dashboard for viewing Hidden Value Game experiment results."""

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hidden Value Game Dashboard",
    page_icon="🎮",
    layout="wide",
)


def load_game_result(path: str | Path) -> dict[str, Any]:
    """Load a game result JSON file."""
    with open(path) as f:
        return json.load(f)


def find_result_files() -> list[Path]:
    """Find all result JSON files."""
    patterns = [
        "outputs/test/*.json",  # Test results
        "outputs/hidden_value/**/games/**/*.json",  # Individual game results
        "outputs/hidden_value/**/result*.json",
        "outputs/debate_structure_test/**/summary.json",  # Debate structure test summaries
        "outputs/debate_structure_test/**/game_*.json",  # Debate structure tests
        "outputs/debate_structure/**/summary.json",  # Full debate structure summaries
        "outputs/debate_structure/**/game_*.json",  # Full debate structure experiments
        "results/*/minimal_test_result.json",
        "results/**/*.json",
    ]
    files = []
    for pattern in patterns:
        files.extend(Path(".").glob(pattern))

    # Filter and prioritize files
    summary_files = []
    individual_games = []
    aggregate_files = []

    for f in files:
        if f.name == "summary.json":
            summary_files.append(f)
        elif "/games/" in str(f) or "minimal_test" in str(f) or "seed_" in f.name or f.name.startswith("game_"):
            individual_games.append(f)
        else:
            aggregate_files.append(f)

    # Sort each category by mtime (newest first)
    summary_files = sorted(set(summary_files), key=lambda p: p.stat().st_mtime, reverse=True)
    individual_games = sorted(set(individual_games), key=lambda p: p.stat().st_mtime, reverse=True)
    aggregate_files = sorted(set(aggregate_files), key=lambda p: p.stat().st_mtime, reverse=True)

    # Return: summaries first, then individual games, then aggregates
    return summary_files + individual_games + aggregate_files


def render_sidebar(result: dict) -> None:
    """Render sidebar with config summary and quick metrics."""
    st.sidebar.header("Game Configuration")

    config = result.get("config", {})
    st.sidebar.write(f"**Objects:** {config.get('n_objects', 'N/A')}")
    st.sidebar.write(f"**Agents:** {config.get('n_agents', 'N/A')}")
    st.sidebar.write(f"**Rounds:** {config.get('n_rounds', 'N/A')}")
    st.sidebar.write(f"**Oracle Budget:** {config.get('oracle_budget', 'N/A')}")
    st.sidebar.write(f"**Selection Size:** {config.get('selection_size', 'N/A')}")
    st.sidebar.write(f"**Condition:** {config.get('condition', 'N/A')}")
    st.sidebar.write(f"**Rule Complexity:** {config.get('rule_complexity', 'N/A')}")

    st.sidebar.divider()
    st.sidebar.header("Quick Metrics")

    metrics = result.get("metrics", {})
    st.sidebar.metric(
        "Selection Accuracy",
        f"{metrics.get('selection_accuracy', 0) * 100:.1f}%",
    )
    st.sidebar.metric(
        "Property Accuracy",
        f"{metrics.get('property_accuracy', 0) * 100:.1f}%",
    )
    st.sidebar.metric(
        "Optimal Overlap",
        f"{metrics.get('optimal_overlap', 0)}/{config.get('selection_size', 'N/A')}",
    )


def render_world_tab(result: dict) -> None:
    """Render the World Overview tab."""
    st.header("World Overview")

    world_state = result.get("world_state", {})
    objects = world_state.get("objects", {})
    computed_values = world_state.get("computed_values", {})
    value_rule = result.get("value_rule", {})
    final_selection = result.get("final_selection", [])

    # Value Rule Display
    st.subheader("Value Rule")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write(f"**Name:** {value_rule.get('name', 'N/A')}")
        st.write(f"**Description:** {value_rule.get('description', 'N/A')}")
    with col2:
        conditions = value_rule.get("conditions", [])
        if conditions:
            st.write("**Conditions:**")
            for cond in conditions:
                st.write(f"  - {cond.get('description', 'N/A')}: +{cond.get('bonus', 0)}")

    st.divider()

    # Objects Table
    st.subheader("Objects")

    # Build objects dataframe
    rows = []
    for obj_id, obj_data in objects.items():
        props = obj_data.get("properties", {})
        row = {
            "ID": obj_id,
            "Color": props.get("color", "N/A"),
            "Shape": props.get("shape", "N/A"),
            "Size": props.get("size", "N/A"),
            "Material": props.get("material", "N/A"),
            "Dangerous": props.get("is_dangerous", "N/A"),
            "Base Value": obj_data.get("base_value", 0),
            "True Value": computed_values.get(obj_id, 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("True Value", ascending=False)

    # Find optimal selection (top N by true value)
    selection_size = result.get("config", {}).get("selection_size", 3)
    optimal_ids = df.nlargest(selection_size, "True Value")["ID"].tolist()

    # Style function to highlight selections
    def highlight_selections(row):
        styles = [""] * len(row)
        obj_id = row["ID"]
        if obj_id in final_selection and obj_id in optimal_ids:
            # Both selected and optimal - gold
            styles = ["background-color: #FFD700"] * len(row)
        elif obj_id in final_selection:
            # Selected but not optimal - light blue
            styles = ["background-color: #ADD8E6"] * len(row)
        elif obj_id in optimal_ids:
            # Optimal but not selected - light green
            styles = ["background-color: #90EE90"] * len(row)
        return styles

    styled_df = df.style.apply(highlight_selections, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Legend
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟡 **Gold:** Selected AND Optimal")
    with col2:
        st.markdown("🔵 **Blue:** Observer Selected")
    with col3:
        st.markdown("🟢 **Green:** Optimal (not selected)")

    # Selection Comparison
    st.divider()
    st.subheader("Selection Comparison")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Observer's Selection:**")
        for obj_id in final_selection:
            value = computed_values.get(obj_id, 0)
            st.write(f"  - {obj_id}: {value}")
        st.write(f"  **Total:** {sum(computed_values.get(oid, 0) for oid in final_selection)}")

    with col2:
        st.write("**Optimal Selection:**")
        for obj_id in optimal_ids:
            value = computed_values.get(obj_id, 0)
            st.write(f"  - {obj_id}: {value}")
        st.write(f"  **Total:** {result.get('metrics', {}).get('optimal_value', 0)}")

    with col3:
        metrics = result.get("metrics", {})
        st.metric("Value Captured", f"{metrics.get('selection_accuracy', 0) * 100:.1f}%")
        st.metric(
            "Overlap",
            f"{metrics.get('optimal_overlap', 0)}/{selection_size}",
        )


def render_transcript_tab(result: dict) -> None:
    """Render round-by-round conversation as group chat."""
    st.header("Conversation")

    rounds = result.get("rounds", [])
    if not rounds:
        st.warning("No rounds data found.")
        return

    # === Ground Truth Panel (top) ===
    world_state = result.get("world_state", {})
    objects = world_state.get("objects", {})
    computed_values = world_state.get("computed_values", {})
    value_rule = result.get("value_rule", {})

    with st.expander("📊 Ground Truth (Objects & Values)", expanded=True):
        # Value rule
        st.markdown(f"**Value Rule:** {value_rule.get('description', 'N/A')}")
        conditions = value_rule.get("conditions", [])
        if conditions:
            cond_text = ", ".join(f"{c.get('description')}: +{c.get('bonus')}" for c in conditions)
            st.markdown(f"**Conditions:** {cond_text}")

        st.markdown("---")

        # Objects table
        rows = []
        for obj_id, obj_data in objects.items():
            props = obj_data.get("properties", {})
            row = {"ID": obj_id, "Value": computed_values.get(obj_id, 0)}
            row.update(props)
            rows.append(row)

        if rows:
            obj_df = pd.DataFrame(rows)
            # Sort by value descending
            obj_df = obj_df.sort_values("Value", ascending=False)
            st.dataframe(obj_df, use_container_width=True, height=200)

    st.markdown("---")

    # Agent info and colors
    agents = result.get("agents", [])
    agent_interests = {a.get("id"): a.get("interest", {}).get("description", "") for a in agents}

    # Assign colors to agents (distinct, accessible colors)
    agent_colors = [
        "#E3F2FD",  # Light blue
        "#FCE4EC",  # Light pink
        "#E8F5E9",  # Light green
        "#FFF3E0",  # Light orange
        "#F3E5F5",  # Light purple
        "#E0F7FA",  # Light cyan
    ]
    agent_border_colors = [
        "#1976D2",  # Blue
        "#C2185B",  # Pink
        "#388E3C",  # Green
        "#F57C00",  # Orange
        "#7B1FA2",  # Purple
        "#0097A7",  # Cyan
    ]
    agent_ids = [a.get("id") for a in agents]
    agent_color_map = {
        aid: (agent_colors[i % len(agent_colors)], agent_border_colors[i % len(agent_border_colors)])
        for i, aid in enumerate(agent_ids)
    }

    for round_data in rounds:
        st.subheader(f"Round {round_data['round_number']}")

        # Render as chat messages
        statements = round_data.get("agent_statements", [])
        for stmt in statements:
            agent_id = stmt.get("agent_id", "Unknown")
            text = stmt.get("text", "")
            thinking = stmt.get("thinking")

            # Get agent colors
            bg_color, border_color = agent_color_map.get(agent_id, ("#F5F5F5", "#9E9E9E"))
            interest = agent_interests.get(agent_id, "")

            # Styled message container
            goal_text = f" · <i>Goal: {interest}</i>" if interest else ""
            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    border-left: 4px solid {border_color};
                    padding: 12px 16px;
                    margin: 8px 0;
                    border-radius: 8px;
                ">
                    <div style="font-weight: bold; color: {border_color}; margin-bottom: 4px;">
                        {agent_id}{goal_text}
                    </div>
                    <div style="color: #333;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show thinking in expander if available
            if thinking:
                with st.expander(f"🧠 {agent_id}'s Reasoning (CoT)"):
                    st.markdown(thinking)

        # Oracle result inline
        oracle = round_data.get("oracle_query")
        if oracle:
            query_type = oracle.get("query_type", "unknown")
            obj_id = oracle.get("object_id", "unknown")
            result_val = oracle.get("result", "N/A")

            if query_type == "property":
                prop_name = oracle.get("property_name", "unknown")
                query_text = f"{prop_name} of {obj_id}"
            else:
                query_text = f"{query_type} of {obj_id}"

            st.markdown(
                f"""
                <div style="
                    background-color: #FFF8E1;
                    border-left: 4px solid #FFC107;
                    padding: 12px 16px;
                    margin: 8px 0;
                    border-radius: 8px;
                ">
                    <div style="font-weight: bold; color: #F57F17; margin-bottom: 4px;">
                        🔮 Oracle
                    </div>
                    <div style="color: #333;"><b>Query:</b> {query_text}</div>
                    <div style="color: #333;"><b>Result:</b> {result_val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Judge's picks and reasoning after each round (private - agents don't see this)
        observer_picks = round_data.get("observer_current_picks", [])
        observer_reasoning = round_data.get("observer_reasoning")
        remaining_after = round_data.get("remaining_objects", [])
        round_metrics = round_data.get("round_metrics", {})

        if observer_picks or observer_reasoning:
            # Use native Streamlit container with info styling
            with st.container():
                st.markdown("**:blue[Judge's Private Analysis]**")

                # Show per-pick details with optimality
                if round_metrics and round_metrics.get("per_pick_details"):
                    for pick_detail in round_metrics["per_pick_details"]:
                        pick_id = pick_detail.get("id", "?")
                        pick_val = pick_detail.get("value", 0)
                        was_optimal = pick_detail.get("was_optimal", False)
                        opt_marker = " :green[(optimal)]" if was_optimal else " :red[(not optimal)]"
                        st.markdown(f"- **{pick_id}**: value {pick_val}{opt_marker}")

                    # Show round totals
                    st.markdown(f"**Round Value:** {round_metrics.get('picks_value', 0)} | "
                               f"**Cumulative:** {round_metrics.get('cumulative_value', 0)} | "
                               f"**Optimal Picks So Far:** {round_metrics.get('cumulative_optimal_count', 0)}")
                elif observer_picks:
                    # Fallback for old data without round_metrics
                    picks_with_values = []
                    total_value = 0
                    for pick in observer_picks:
                        val = computed_values.get(pick, 0)
                        picks_with_values.append(f"{pick} ({val})")
                        total_value += val
                    st.markdown(f"**Picks This Round:** {', '.join(picks_with_values)}")
                    st.markdown(f"**Value Gained:** {total_value}")

                if observer_reasoning:
                    st.text(observer_reasoning)

        # Show remaining objects after this round
        if remaining_after:
            remaining_with_values = []
            for obj_id in remaining_after:
                val = computed_values.get(obj_id, 0)
                remaining_with_values.append(f"{obj_id} ({val})")
            st.caption(f"**Remaining Objects ({len(remaining_after)}):** {', '.join(remaining_with_values)}")

        st.divider()

    # === Observer Summary at the End ===
    st.header("Observer Summary")

    final_selection = result.get("final_selection", [])
    metrics = result.get("metrics", {})
    config = result.get("config", {})
    selection_size = config.get("selection_size", len(final_selection))

    # Calculate optimal selection
    all_objects = list(objects.keys())
    sorted_by_value = sorted(all_objects, key=lambda x: computed_values.get(x, 0), reverse=True)
    optimal_ids = sorted_by_value[:selection_size]

    # Show picks by round with metrics progression
    st.subheader("Picks by Round")

    # Build progression data for chart
    progression_data = []
    total_value = 0
    all_picks = []

    for round_data in rounds:
        round_num = round_data.get("round_number", "?")
        picks = round_data.get("observer_current_picks", [])
        round_metrics = round_data.get("round_metrics", {})

        for pick in picks:
            val = computed_values.get(pick, 0)
            total_value += val
            all_picks.append(pick)
            is_optimal = pick in optimal_ids
            opt_marker = " :green[*]" if is_optimal else ""
            st.markdown(f"  Round {round_num}: **{pick}** (value: {val}){opt_marker}")

        # Add to progression data
        if round_metrics:
            progression_data.append({
                "Round": round_num,
                "Cumulative Value": round_metrics.get("cumulative_value", 0),
                "Optimal Picks": round_metrics.get("cumulative_optimal_count", 0),
            })

    st.markdown("_:green[*] = optimal pick_")

    # Show progression chart if we have data
    if progression_data:
        st.subheader("Value Progression")
        prog_df = pd.DataFrame(progression_data)
        st.line_chart(prog_df.set_index("Round")["Cumulative Value"])

    # Final selection summary
    st.subheader("Final Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Value",
            f"{metrics.get('total_value', total_value)}",
            delta=f"of {metrics.get('optimal_value', 'N/A')} optimal",
        )

    with col2:
        accuracy = metrics.get('selection_accuracy', 0)
        st.metric(
            "Selection Accuracy",
            f"{accuracy * 100:.1f}%",
        )

    with col3:
        overlap = metrics.get('optimal_overlap', 0)
        st.metric(
            "Optimal Overlap",
            f"{overlap}/{selection_size}",
        )

    # Show final selection vs optimal
    st.subheader("Selection Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Observer's Selection:**")
        for obj_id in final_selection:
            val = computed_values.get(obj_id, 0)
            is_optimal = obj_id in optimal_ids
            marker = " (optimal)" if is_optimal else ""
            st.write(f"  - {obj_id}: {val}{marker}")
        st.write(f"  **Total:** {sum(computed_values.get(o, 0) for o in final_selection)}")

    with col2:
        st.write("**Optimal Selection:**")
        for obj_id in optimal_ids:
            val = computed_values.get(obj_id, 0)
            was_picked = obj_id in final_selection
            marker = " (picked)" if was_picked else " (missed)"
            st.write(f"  - {obj_id}: {val}{marker}")
        st.write(f"  **Total:** {sum(computed_values.get(o, 0) for o in optimal_ids)}")

    # === Observer's Final Beliefs ===
    st.header("Observer's Final Beliefs")

    # Inferred Rule
    inferred_rule = result.get("inferred_rule") or {}
    if inferred_rule:
        st.subheader("Inferred Value Rule")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Description:** {inferred_rule.get('description', 'N/A')}")
            key_factors = inferred_rule.get('key_factors', [])
            if key_factors:
                st.write(f"**Key Factors:** {', '.join(key_factors)}")
        with col2:
            st.metric("Confidence", f"{inferred_rule.get('confidence', 0)}%")

        # Compare to actual rule
        actual_rule = result.get("value_rule", {})
        with st.expander("Compare to Actual Rule"):
            st.write(f"**Actual:** {actual_rule.get('description', 'N/A')}")
            conditions = actual_rule.get("conditions", [])
            if conditions:
                cond_text = ", ".join(f"{c.get('description')}: +{c.get('bonus')}" for c in conditions)
                st.write(f"**Conditions:** {cond_text}")
    else:
        st.info("No inferred rule recorded.")

    st.divider()

    # Property & Value Beliefs vs Ground Truth
    st.subheader("Observer Beliefs vs Ground Truth")

    observer_prop_beliefs = result.get("observer_property_beliefs") or {}
    observer_value_beliefs = result.get("observer_value_beliefs") or {}

    # Build comparison table for ALL objects
    rows = []
    for obj_id in sorted(objects.keys()):
        true_props = objects.get(obj_id, {}).get("properties", {})
        believed_props = observer_prop_beliefs.get(obj_id, {})
        true_value = computed_values.get(obj_id, 0)
        believed_value = observer_value_beliefs.get(obj_id)

        row = {
            "Object": obj_id,
            "True Value": true_value,
            "Believed Value": believed_value if believed_value is not None else "-",
        }

        # Add each property: show "believed / true" format
        for prop_name in ["color", "size", "shape", "material"]:
            true_val = true_props.get(prop_name, "?")
            believed_val = believed_props.get(prop_name)

            if believed_val is None:
                row[f"{prop_name} (B/T)"] = f"- / {true_val}"
            else:
                true_str = str(true_val).lower().strip()
                believed_str = str(believed_val).lower().strip()
                is_correct = believed_str == true_str
                marker = "" if is_correct else " X"
                row[f"{prop_name} (B/T)"] = f"{believed_val} / {true_val}{marker}"

        rows.append(row)

    if rows:
        beliefs_df = pd.DataFrame(rows)
        st.dataframe(beliefs_df, use_container_width=True)
        st.caption("Format: Believed / True. X = incorrect belief.")

        # Show accuracy metrics
        metrics = result.get("metrics", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Property Accuracy", f"{metrics.get('property_accuracy', 0) * 100:.1f}%")
        with col2:
            st.metric("Value Prediction Accuracy", f"{metrics.get('value_prediction_accuracy', 0) * 100:.1f}%")
        with col3:
            st.metric("Rule Inference Accuracy", f"{metrics.get('rule_inference_accuracy', 0) * 100:.1f}%")
    else:
        st.info("No beliefs recorded.")

    # Raw beliefs JSON (collapsed)
    final_round = rounds[-1] if rounds else {}
    observer_beliefs = final_round.get("observer_beliefs") or {}
    if observer_beliefs:
        with st.expander("View Raw Observer State", expanded=False):
            st.json(observer_beliefs)


def render_agents_tab(result: dict) -> None:
    """Render the Agent Analysis tab."""
    st.header("Agent Analysis")

    agents = result.get("agents", [])
    if not agents:
        st.warning("No agent data found.")
        return

    metrics = result.get("metrics", {})
    agent_win_rates = metrics.get("agent_win_rates", {})
    config = result.get("config", {})

    # Check if agents have value functions
    has_value_functions = config.get("use_agent_value_functions", False)

    # Agent Cards
    cols = st.columns(len(agents))
    for i, agent in enumerate(agents):
        with cols[i]:
            agent_id = agent.get("id", f"Agent_{i}")
            st.subheader(agent_id)

            # Show value function if present
            value_function = agent.get("value_function")
            if value_function:
                st.write(f"**Value Function:** {value_function.get('name', 'N/A')}")
                st.write(f"**Goal:** {value_function.get('description', 'N/A')}")
                conditions = value_function.get("conditions", [])
                if conditions:
                    with st.expander("Value Conditions"):
                        for cond in conditions:
                            st.write(f"- {cond.get('description', 'N/A')}: {cond.get('bonus', 0):+d}")
            else:
                # Show simple interest
                interest = agent.get("interest", {})
                st.write(f"**Goal:** {interest.get('description', 'N/A')}")
                st.write(f"**Target Condition:** {interest.get('target_condition', 'N/A')}")

            st.write(f"**Model:** {agent.get('model', 'N/A')}")

            win_rate = agent_win_rates.get(agent_id, 0)
            st.metric("Win Rate", f"{win_rate * 100:.0f}%")

    # Agent Value Progression (if value functions are used)
    if has_value_functions:
        st.divider()
        st.subheader("Agent Value Progression")

        # Extract progression data from rounds
        rounds = result.get("rounds", [])
        progression_data = []

        for r in rounds:
            round_metrics = r.get("round_metrics", {})
            if round_metrics:
                row = {"Round": r.get("round_number", 0)}
                agent_cumulative = round_metrics.get("agent_cumulative_value", {})
                for agent in agents:
                    agent_id = agent.get("id", "")
                    row[agent_id] = agent_cumulative.get(agent_id, 0)
                progression_data.append(row)

        if progression_data:
            prog_df = pd.DataFrame(progression_data)

            # Line chart showing agent cumulative value over rounds
            agent_ids = [a.get("id", "") for a in agents]
            chart_df = prog_df.set_index("Round")[agent_ids]
            st.line_chart(chart_df)

            # Final totals
            st.subheader("Final Agent Values")
            final_row = progression_data[-1] if progression_data else {}
            final_cols = st.columns(len(agents))
            for i, agent in enumerate(agents):
                with final_cols[i]:
                    agent_id = agent.get("id", "")
                    final_value = final_row.get(agent_id, 0)
                    st.metric(agent_id, f"{final_value:+d}")

            # Per-round value gain table
            with st.expander("Per-Round Value Breakdown"):
                round_value_rows = []
                for r in rounds:
                    round_metrics = r.get("round_metrics", {})
                    if round_metrics:
                        row = {"Round": r.get("round_number", 0)}
                        agent_round_val = round_metrics.get("agent_round_value", {})
                        for agent in agents:
                            agent_id = agent.get("id", "")
                            row[f"{agent_id} (round)"] = agent_round_val.get(agent_id, 0)
                        round_value_rows.append(row)

                if round_value_rows:
                    round_val_df = pd.DataFrame(round_value_rows)
                    st.dataframe(round_val_df, use_container_width=True)

    st.divider()

    # Statement Counts
    st.subheader("Statement Counts")

    # Collect all statements from all rounds
    all_statements = []
    for round_data in result.get("rounds", []):
        for stmt in round_data.get("agent_statements", []):
            all_statements.append(stmt)

    if not all_statements:
        st.info("No statements to analyze.")
        return

    # Build count table
    count_data = []
    for agent in agents:
        agent_id = agent.get("id")
        agent_stmts = [s for s in all_statements if s.get("agent_id") == agent_id]
        count_data.append({
            "Agent": agent_id,
            "Total Statements": len(agent_stmts),
        })

    count_df = pd.DataFrame(count_data)
    st.dataframe(count_df, use_container_width=True)

    # All Statements by Agent
    st.subheader("All Statements")

    for agent in agents:
        agent_id = agent.get("id")
        agent_stmts = [s for s in all_statements if s.get("agent_id") == agent_id]

        with st.expander(f"{agent_id}'s Statements ({len(agent_stmts)} total)"):
            for i, stmt in enumerate(agent_stmts, 1):
                st.write(f"{i}. {stmt.get('text', '')}")


def render_metrics_tab(result: dict) -> None:
    """Render the Metrics & Baselines tab."""
    st.header("Metrics & Baselines")

    metrics = result.get("metrics", {})

    # Metric Cards
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Selection Accuracy",
            f"{metrics.get('selection_accuracy', 0) * 100:.1f}%",
        )
    with col2:
        st.metric(
            "Property Accuracy",
            f"{metrics.get('property_accuracy', 0) * 100:.1f}%",
        )
    with col3:
        st.metric(
            "Rule Inference Accuracy",
            f"{metrics.get('rule_inference_accuracy', 0) * 100:.1f}%",
        )
    with col4:
        st.metric(
            "Rule Confidence",
            f"{metrics.get('rule_confidence', 0)}%",
        )

    st.divider()

    # Baseline Comparison
    st.subheader("Baseline Comparison")

    # Build comparison data
    observer_value = metrics.get("total_value", 0)
    optimal_value = metrics.get("optimal_value", 1)
    random_value = metrics.get("random_selection_value", 0)
    single_agent_values = metrics.get("single_agent_trust_values", {})

    comparison_data = [
        {"Method": "Observer", "Value": observer_value, "Accuracy": observer_value / optimal_value if optimal_value else 0},
        {"Method": "Optimal", "Value": optimal_value, "Accuracy": 1.0},
        {"Method": "Random", "Value": random_value, "Accuracy": random_value / optimal_value if optimal_value else 0},
    ]

    for agent_id, value in single_agent_values.items():
        comparison_data.append({
            "Method": f"Trust {agent_id}",
            "Value": value,
            "Accuracy": value / optimal_value if optimal_value else 0,
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Display as table
    st.dataframe(comparison_df, use_container_width=True)

    # Bar chart
    st.bar_chart(comparison_df.set_index("Method")["Value"])

    st.divider()

    # Oracle Efficiency
    st.subheader("Oracle Usage")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Queries Used", metrics.get("oracle_queries_used", 0))
    with col2:
        st.metric("Budget", metrics.get("oracle_budget", 0))
    with col3:
        st.metric("Efficiency", f"{metrics.get('oracle_efficiency', 0):.1f}")

    # Value comparisons
    st.subheader("Value Comparisons")

    col1, col2 = st.columns(2)
    with col1:
        vs_random = metrics.get("value_vs_random", 0)
        delta_color = "normal" if vs_random >= 0 else "inverse"
        st.metric(
            "vs Random",
            f"{observer_value}",
            delta=f"{vs_random:+.0f}",
            delta_color=delta_color,
        )
    with col2:
        vs_best = metrics.get("value_vs_best_agent", 0)
        delta_color = "normal" if vs_best >= 0 else "inverse"
        st.metric(
            "vs Best Single Agent",
            f"{observer_value}",
            delta=f"{vs_best:+.0f}",
            delta_color=delta_color,
        )


def render_estimator_tab(result: dict) -> None:
    """Render Estimator analysis tab."""
    st.header("Estimator Analysis")

    if not result.get("estimator_beliefs"):
        st.info("Estimator not enabled for this game. Enable with `enable_estimator: true` in config.")
        return

    estimator_beliefs = result.get("estimator_beliefs") or {}
    estimator_rule = result.get("estimator_inferred_rule") or {}
    estimator_metrics = result.get("estimator_metrics") or {}
    world_state = result.get("world_state") or {}
    objects = world_state.get("objects") or {}

    # Metrics comparison
    st.subheader("Accuracy Comparison: Observer vs Estimator")

    observer_metrics = result.get("metrics", {})

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Observer**")
        st.metric(
            "Property Accuracy",
            f"{observer_metrics.get('property_accuracy', 0) * 100:.1f}%",
        )
        st.metric(
            "Rule Inference Accuracy",
            f"{observer_metrics.get('rule_inference_accuracy', 0) * 100:.1f}%",
        )

    with col2:
        st.write("**Estimator**")
        st.metric(
            "Property Accuracy",
            f"{estimator_metrics.get('property_accuracy', 0) * 100:.1f}%",
        )
        st.metric(
            "Rule Inference Accuracy",
            f"{estimator_metrics.get('rule_inference_accuracy', 0) * 100:.1f}%",
        )

    st.divider()

    # Estimator's Rule Guess
    st.subheader("Estimator's Rule Inference")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Actual Rule:**")
        value_rule = result.get("value_rule", {})
        st.write(f"Name: {value_rule.get('name', 'N/A')}")
        st.write(f"Description: {value_rule.get('description', 'N/A')}")
        conditions = value_rule.get("conditions", [])
        if conditions:
            st.write("Conditions:")
            for cond in conditions:
                st.write(f"  - {cond.get('description')}: +{cond.get('bonus')}")

    with col2:
        st.write("**Estimator's Guess:**")
        st.write(f"Description: {estimator_rule.get('description', 'N/A')}")
        st.write(f"Confidence: {estimator_rule.get('confidence', 0)}%")
        key_factors = estimator_rule.get("key_factors", [])
        if key_factors:
            st.write(f"Key Factors: {', '.join(key_factors)}")

    st.divider()

    # Estimator Property Beliefs vs Ground Truth
    st.subheader("Estimator Property Beliefs vs Ground Truth")

    if not estimator_beliefs:
        st.info("No property beliefs recorded by estimator.")
    else:
        # Build comparison table
        rows = []
        total_correct = 0
        total_beliefs = 0

        for obj_id, beliefs in estimator_beliefs.items():
            true_props = objects.get(obj_id, {}).get("properties", {})

            for prop_name, believed_value in beliefs.items():
                true_value = true_props.get(prop_name, "N/A")

                # Normalize for comparison
                believed_str = str(believed_value).lower().strip()
                true_str = str(true_value).lower().strip()

                is_correct = believed_str == true_str
                if not is_correct:
                    # Check for common equivalences
                    if believed_str in ("circular", "circle") and true_str in ("circular", "circle"):
                        is_correct = True

                total_beliefs += 1
                if is_correct:
                    total_correct += 1

                rows.append({
                    "Object": obj_id,
                    "Property": prop_name,
                    "Estimator Belief": believed_value,
                    "True Value": true_value,
                    "Correct": "Yes" if is_correct else "No",
                })

        if rows:
            beliefs_df = pd.DataFrame(rows)

            # Style correct/incorrect
            def style_correct(val):
                if val == "Yes":
                    return "background-color: #90EE90"
                elif val == "No":
                    return "background-color: #FFB6C1"
                return ""

            styled_df = beliefs_df.style.map(style_correct, subset=["Correct"])
            st.dataframe(styled_df, use_container_width=True)

            st.metric(
                "Estimator Property Accuracy (computed)",
                f"{100 * total_correct / total_beliefs:.1f}%" if total_beliefs > 0 else "N/A",
            )
        else:
            st.info("No property beliefs to compare.")

    # Agent Objective Inference Section
    st.divider()
    st.subheader("Agent Objective Inference")

    agent_objective_inference = result.get("agent_objective_inference")
    agent_objective_scores = result.get("agent_objective_scores", {})
    agent_objective_overall_score = result.get("agent_objective_overall_score")

    if not agent_objective_inference:
        st.info("Agent objective inference not enabled for this game. Enable with `infer_agent_objectives: true` in config.")
    else:
        # Overall score
        if agent_objective_overall_score is not None:
            st.metric(
                "Overall Objective Inference Score",
                f"{agent_objective_overall_score * 100:.1f}%",
                help="Average LLM judge score for how well the estimator inferred agent objectives"
            )

        st.divider()

        # Show inference for each agent
        agents = result.get("agents", [])
        agent_dict = {a.get("id"): a for a in agents}

        for agent_id, inference in agent_objective_inference.items():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**{agent_id}**")

                # Show inferred objective
                st.write(f"**Inferred Goal:** {inference.get('inferred_goal', 'N/A')}")

                inferred_factors = inference.get("inferred_factors", [])
                if inferred_factors:
                    st.write(f"**Inferred Factors:** {', '.join(inferred_factors)}")

                st.write(f"**Reasoning:** {inference.get('reasoning', 'N/A')}")

                evidence = inference.get("evidence", [])
                if evidence:
                    with st.expander("Key Evidence"):
                        for e in evidence:
                            st.write(f"- {e}")

            with col2:
                # Show score
                score = agent_objective_scores.get(agent_id, 0)
                st.metric(
                    "Inference Score",
                    f"{score * 100:.0f}%",
                    help="LLM judge score for this inference"
                )
                st.write(f"Confidence: {inference.get('confidence', 0)}%")

            # Show ground truth comparison
            agent_info = agent_dict.get(agent_id, {})
            with st.expander(f"Ground Truth for {agent_id}"):
                value_function = agent_info.get("value_function")
                if value_function:
                    st.write(f"**Name:** {value_function.get('name', 'N/A')}")
                    st.write(f"**Description:** {value_function.get('description', 'N/A')}")
                    conditions = value_function.get("conditions", [])
                    if conditions:
                        st.write("**Conditions:**")
                        for cond in conditions:
                            st.write(f"  - {cond.get('description', 'N/A')}: {cond.get('bonus', 0):+d}")
                else:
                    interest = agent_info.get("interest", {})
                    st.write(f"**Target:** {interest.get('target_condition', 'N/A')}")
                    st.write(f"**Description:** {interest.get('description', 'N/A')}")

            st.divider()


def render_truth_recovery_tab(result: dict) -> None:
    """Render the Truth Recovery tab."""
    st.header("Truth Recovery")

    # Inferred Rule vs Actual Rule
    st.subheader("Rule Inference")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Actual Rule:**")
        value_rule = result.get("value_rule", {})
        st.write(f"Name: {value_rule.get('name', 'N/A')}")
        st.write(f"Description: {value_rule.get('description', 'N/A')}")
        conditions = value_rule.get("conditions", [])
        if conditions:
            st.write("Conditions:")
            for cond in conditions:
                st.write(f"  - {cond.get('description')}: +{cond.get('bonus')}")

    with col2:
        st.write("**Observer's Inferred Rule:**")
        inferred = result.get("inferred_rule") or {}
        st.write(f"Description: {inferred.get('description', 'N/A')}")
        st.write(f"Confidence: {inferred.get('confidence', 0)}%")
        key_factors = inferred.get("key_factors", [])
        if key_factors:
            st.write(f"Key Factors: {', '.join(key_factors)}")

    metrics = result.get("metrics", {})
    st.metric(
        "Rule Inference Accuracy",
        f"{metrics.get('rule_inference_accuracy', 0) * 100:.1f}%",
    )

    st.divider()

    # Property Beliefs vs Ground Truth
    st.subheader("Property Beliefs vs Ground Truth")

    observer_beliefs = result.get("observer_property_beliefs") or {}
    world_state = result.get("world_state") or {}
    objects = world_state.get("objects") or {}

    if not observer_beliefs:
        st.info("No property beliefs recorded.")
        return

    # Build comparison table
    rows = []
    total_correct = 0
    total_beliefs = 0

    for obj_id, beliefs in observer_beliefs.items():
        true_props = objects.get(obj_id, {}).get("properties", {})

        for prop_name, believed_value in beliefs.items():
            true_value = true_props.get(prop_name, "N/A")

            # Normalize for comparison (handle string variations)
            believed_str = str(believed_value).lower().strip()
            true_str = str(true_value).lower().strip()

            # Handle common variations
            is_correct = believed_str == true_str
            if not is_correct:
                # Check for common equivalences
                if believed_str in ("circular", "circle") and true_str in ("circular", "circle"):
                    is_correct = True

            total_beliefs += 1
            if is_correct:
                total_correct += 1

            rows.append({
                "Object": obj_id,
                "Property": prop_name,
                "Believed": believed_value,
                "True": true_value,
                "Correct": "Yes" if is_correct else "No",
            })

    if rows:
        beliefs_df = pd.DataFrame(rows)

        # Style correct/incorrect
        def style_correct(val):
            if val == "Yes":
                return "background-color: #90EE90"
            elif val == "No":
                return "background-color: #FFB6C1"
            return ""

        styled_df = beliefs_df.style.map(style_correct, subset=["Correct"])
        st.dataframe(styled_df, use_container_width=True)

        st.metric(
            "Property Accuracy (beliefs)",
            f"{100 * total_correct / total_beliefs:.1f}%" if total_beliefs > 0 else "N/A",
        )
    else:
        st.info("No property beliefs to compare.")


def is_aggregate_file(data: dict) -> bool:
    """Check if the loaded data is an aggregate results file (not a single game)."""
    # Aggregate files have keys like 'information_conditions' or a list at root
    return "information_conditions" in data or "rule_complexity" in data or isinstance(data, list)


def is_debate_structure_summary(data: dict) -> bool:
    """Check if this is a debate structure test summary file."""
    return "results" in data and isinstance(data.get("results"), list) and \
           any("turn_structure" in r for r in data.get("results", []))


def render_debate_structure_summary(data: dict, file_path: Path) -> None:
    """Render a summary view for debate structure experiment results."""
    st.header("Debate Structure Experiment Results")

    config = data.get("config", {})
    results = data.get("results", [])
    total_time = data.get("total_time_seconds", 0)

    # Config summary
    st.subheader("Experiment Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Objects:** {config.get('n_objects', 'N/A')}")
        st.write(f"**Agents:** {config.get('n_agents', 'N/A')}")
    with col2:
        st.write(f"**Rounds:** {config.get('n_rounds', 'N/A')}")
        st.write(f"**Oracle Budget:** {config.get('oracle_budget', 'N/A')}")
    with col3:
        st.write(f"**Selection Size:** {config.get('selection_size', 'N/A')}")
        st.write(f"**Total Time:** {total_time/60:.1f} min")

    st.divider()

    # Results table
    st.subheader("Results by Condition")

    # Build dataframe
    rows = []
    for r in results:
        if "error" in r:
            continue
        rows.append({
            "Turn Structure": r.get("turn_structure", "N/A"),
            "Oracle Timing": r.get("oracle_timing", "N/A"),
            "Est. Prop Acc": f"{r.get('estimator_property_accuracy', 0)*100:.0f}%",
            "Obs. Prop Acc": f"{r.get('observer_property_accuracy', 0)*100:.0f}%",
            "Est. Advantage": f"{r.get('estimator_advantage_property', 0)*100:+.0f}%",
            "Selection Acc": f"{r.get('selection_accuracy', 0)*100:.0f}%",
            "Est. Rule Acc": f"{r.get('estimator_rule_accuracy', 0)*100:.0f}%",
            "Obs. Rule Acc": f"{r.get('observer_rule_accuracy', 0)*100:.0f}%",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    st.divider()

    # Pivot table: Turn Structure x Oracle Timing for Estimator Property Accuracy
    st.subheader("Estimator Property Accuracy Heatmap")

    pivot_data = {}
    for r in results:
        if "error" in r:
            continue
        ts = r.get("turn_structure", "N/A")
        ot = r.get("oracle_timing", "N/A")
        est_acc = r.get("estimator_property_accuracy", 0) * 100
        pivot_data[(ts, ot)] = est_acc

    # Create pivot table
    turn_structures = ["interleaved", "batch", "simultaneous", "sequential"]
    oracle_timings = ["before_response", "after_statements"]

    pivot_rows = []
    for ts in turn_structures:
        row = {"Turn Structure": ts}
        for ot in oracle_timings:
            row[ot] = pivot_data.get((ts, ot), "N/A")
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)

    # Style with conditional formatting
    def highlight_values(val):
        if isinstance(val, (int, float)):
            if val >= 80:
                return "background-color: #90EE90"  # Green
            elif val >= 50:
                return "background-color: #FFFACD"  # Light yellow
            else:
                return "background-color: #FFB6C1"  # Light red
        return ""

    styled_df = pivot_df.style.map(
        highlight_values,
        subset=["before_response", "after_statements"]
    )
    st.dataframe(styled_df, use_container_width=True)

    st.divider()

    # Bar chart comparison
    st.subheader("Comparison Chart")

    chart_data = []
    for r in results:
        if "error" in r:
            continue
        chart_data.append({
            "Condition": f"{r.get('turn_structure', 'N/A')}\n{r.get('oracle_timing', 'N/A')}",
            "Estimator": r.get("estimator_property_accuracy", 0) * 100,
            "Observer": r.get("observer_property_accuracy", 0) * 100,
        })

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        chart_df = chart_df.set_index("Condition")
        st.bar_chart(chart_df)

    st.divider()

    # Key findings
    st.subheader("Key Findings")

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x.get("estimator_property_accuracy", 0))
        worst = min(valid_results, key=lambda x: x.get("estimator_property_accuracy", 0))

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Best for Estimator:** {best.get('turn_structure')} + {best.get('oracle_timing')}")
            st.metric("Estimator Accuracy", f"{best.get('estimator_property_accuracy', 0)*100:.0f}%")

        with col2:
            st.error(f"**Worst for Estimator:** {worst.get('turn_structure')} + {worst.get('oracle_timing')}")
            st.metric("Estimator Accuracy", f"{worst.get('estimator_property_accuracy', 0)*100:.0f}%")

    # Link to individual games
    st.divider()
    st.subheader("Individual Games")
    st.info(f"Individual game results are saved in: {file_path.parent}")

    # List available game files
    game_files = list(file_path.parent.glob("game_*.json"))
    if game_files:
        for gf in sorted(game_files):
            st.write(f"- `{gf.name}`")


def render_aggregate_view(data: dict, file_path: Path) -> None:
    """Render a view for aggregate result files."""
    st.header("Aggregate Results")
    st.info(f"This file contains aggregate experiment data: {file_path.name}")

    # Display the raw structure
    st.json(data)


def main():
    st.title("Hidden Value Game Dashboard")

    # Sidebar: File selection - show all files at once
    st.sidebar.header("Result Files")

    result_files = find_result_files()

    if not result_files:
        st.warning("No result files found.")
        st.info("Run an experiment first, or enter a path manually below.")

        manual_path = st.text_input("Enter path to result JSON file:")
        if manual_path and Path(manual_path).exists():
            result_files = [Path(manual_path)]
        else:
            return

    # Create file options with display names
    file_options = {f"{p.parent.name}/{p.name}": p for p in result_files}

    # Show all files as radio buttons in sidebar
    selected_name = st.sidebar.radio(
        "Select a result file:",
        options=list(file_options.keys()),
        key="file_selector",
        label_visibility="collapsed",
    )

    if not selected_name:
        return

    selected_file = file_options[selected_name]

    st.sidebar.divider()
    st.sidebar.caption(f"**Selected:** {selected_file}")

    # Load data
    data = load_game_result(selected_file)

    # Check if it's a debate structure summary
    if is_debate_structure_summary(data):
        render_debate_structure_summary(data, selected_file)
        return

    # Check if it's an aggregate file
    if is_aggregate_file(data):
        render_aggregate_view(data, selected_file)
        return

    # Render sidebar with config and metrics
    render_sidebar(data)

    # Show timestamp if available
    timestamp = data.get("timestamp", "")
    if timestamp:
        st.sidebar.write(f"**Timestamp:** {timestamp[:19]}")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "World Overview",
        "Transcript",
        "Agent Analysis",
        "Metrics",
        "Truth Recovery",
        "Estimator",
    ])

    with tab1:
        render_world_tab(data)

    with tab2:
        render_transcript_tab(data)

    with tab3:
        render_agents_tab(data)

    with tab4:
        render_metrics_tab(data)

    with tab5:
        render_truth_recovery_tab(data)

    with tab6:
        render_estimator_tab(data)


if __name__ == "__main__":
    main()
