#!/usr/bin/env python
"""Streamlit dashboard for inspecting ICL experiment data."""

import json
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ICL Experiment Dashboard",
    page_icon="🔍",
    layout="wide",
)


def load_debug_data(path: str) -> dict:
    """Load debug JSON file."""
    with open(path) as f:
        return json.load(f)


def main():
    st.title("ICL Experiment Dashboard")

    # Sidebar: File selection
    st.sidebar.header("Data Source")

    # Find available debug files
    output_dirs = [
        Path("outputs/icl_experiment"),
        Path("outputs/icl_test"),
    ]

    debug_files = []
    for output_dir in output_dirs:
        if output_dir.exists():
            debug_files.extend(output_dir.glob("seed_*_debug.json"))

    if not debug_files:
        st.warning("No debug files found. Run an experiment with the updated run_icl.py first.")
        st.info("Debug files are saved as `outputs/<experiment>/seed_<N>_debug.json`")

        # Allow manual path input
        manual_path = st.text_input("Or enter path to debug JSON file:")
        if manual_path and Path(manual_path).exists():
            debug_files = [Path(manual_path)]
        else:
            return

    selected_file = st.sidebar.selectbox(
        "Select debug file",
        options=sorted(debug_files, reverse=True),
        format_func=lambda p: f"{p.parent.name}/{p.name}",
    )

    if not selected_file:
        return

    # Load data
    data = load_debug_data(selected_file)

    st.sidebar.success(f"Loaded seed {data['metadata']['seed']}")
    st.sidebar.write(f"Timestamp: {data['metadata']['timestamp'][:19]}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Statements", "❓ Queries", "👥 Agents"])

    # === STATEMENTS TAB ===
    with tab1:
        st.header("Statements Browser")

        statements = data["statements"]
        if not statements:
            st.warning("No statements found.")
        else:
            # Build dataframe
            df = pd.DataFrame(statements)

            # Filters
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                agents = ["All"] + sorted(df["agent_id"].unique().tolist())
                selected_agent = st.selectbox("Filter by Agent", agents)

            with col2:
                objects = ["All"] + sorted(df["object_id"].unique().tolist())
                selected_object = st.selectbox("Filter by Object", objects)

            with col3:
                properties = ["All"] + sorted(df["property_name"].unique().tolist())
                selected_property = st.selectbox("Filter by Property", properties)

            with col4:
                truthful_filter = st.selectbox(
                    "Truthfulness",
                    ["All", "Truthful only", "Deceptive only"],
                )

            # Apply filters
            filtered_df = df.copy()
            if selected_agent != "All":
                filtered_df = filtered_df[filtered_df["agent_id"] == selected_agent]
            if selected_object != "All":
                filtered_df = filtered_df[filtered_df["object_id"] == selected_object]
            if selected_property != "All":
                filtered_df = filtered_df[filtered_df["property_name"] == selected_property]
            if truthful_filter == "Truthful only":
                filtered_df = filtered_df[filtered_df["is_truthful"] == True]
            elif truthful_filter == "Deceptive only":
                filtered_df = filtered_df[filtered_df["is_truthful"] == False]

            st.write(f"Showing {len(filtered_df)} of {len(df)} statements")

            # Display columns
            display_cols = [
                "agent_id", "target_id", "object_id", "property_name",
                "claimed_value", "ground_truth", "is_truthful", "relationship"
            ]

            # Style truthful column
            def style_truthful(val):
                if val:
                    return "background-color: #90EE90"  # light green
                return "background-color: #FFB6C1"  # light pink

            styled_df = filtered_df[display_cols].style.applymap(
                style_truthful, subset=["is_truthful"]
            )

            st.dataframe(styled_df, use_container_width=True, height=400)

            # Click to see full text
            st.subheader("Statement Details")
            if len(filtered_df) > 0:
                idx = st.number_input(
                    "Statement index (from filtered list)",
                    min_value=0,
                    max_value=len(filtered_df) - 1,
                    value=0,
                )
                stmt = filtered_df.iloc[idx]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Full Statement Text:**")
                    st.info(stmt["text"])
                with col2:
                    st.markdown("**Metadata:**")
                    st.json({
                        "agent": stmt["agent_id"],
                        "target": stmt["target_id"],
                        "relationship": stmt["relationship"],
                        "object": stmt["object_id"],
                        "property": stmt["property_name"],
                        "claimed": stmt["claimed_value"],
                        "ground_truth": stmt["ground_truth"],
                        "is_truthful": stmt["is_truthful"],
                    })

    # === QUERIES TAB ===
    with tab2:
        st.header("Query Inspector")

        queries = data["queries"]
        if not queries:
            st.warning("No queries found.")
        else:
            # Query selector
            query_options = [
                f"{q['object_id']} / {q['property_name']} = {q['property_value']}"
                for q in queries
            ]

            col1, col2 = st.columns([2, 1])
            with col1:
                selected_query_idx = st.selectbox(
                    "Select query",
                    range(len(queries)),
                    format_func=lambda i: query_options[i],
                )
            with col2:
                # Filter to interesting queries
                show_only = st.radio(
                    "Show",
                    ["All", "Contested only", "Wrong predictions only"],
                    horizontal=True,
                )

            query = queries[selected_query_idx]

            st.subheader(f"Query: Is {query['object_id']}'s {query['property_name']} = {query['property_value']}?")

            col1, col2, col3 = st.columns(3)
            col1.metric("Ground Truth", "Yes" if query["ground_truth"] else "No")
            col2.metric("Category", query["category"])

            # Show relevant statements
            st.subheader("Relevant Statements")
            relevant_stmts = [
                s for s in data["statements"]
                if s["object_id"] == query["object_id"]
                and s["property_name"] == query["property_name"]
            ]

            if relevant_stmts:
                for stmt in relevant_stmts:
                    truthful_icon = "✅" if stmt["is_truthful"] else "❌"
                    st.markdown(
                        f"**{stmt['agent_id']}** → {stmt['target_id']} ({stmt['relationship']}): "
                        f"*\"{stmt['text']}\"* {truthful_icon}"
                    )
                    st.caption(f"Claimed: {stmt['claimed_value']}, Truth: {stmt['ground_truth']}")
            else:
                st.info("No statements about this query.")

            # Observer predictions comparison
            st.subheader("Observer Predictions")

            results = query.get("results", {})
            if not results:
                st.warning("No prediction results found for this query.")
            else:
                # Build comparison table
                rows = []
                for condition, models in results.items():
                    for model, pred_data in models.items():
                        rows.append({
                            "Condition": condition,
                            "Model": model.split("/")[-1],  # Shorten model name
                            "Prediction": "Yes" if pred_data["prediction"] else "No",
                            "Confidence": f"{pred_data['confidence']}%",
                            "Correct": "✅" if pred_data["prediction"] == query["ground_truth"] else "❌",
                        })

                pred_df = pd.DataFrame(rows)
                st.dataframe(pred_df, use_container_width=True)

                # Show evidence for selected condition
                st.subheader("Evidence Shown to Observer")
                available_conditions = list(results.keys())
                if available_conditions:
                    selected_condition = st.selectbox(
                        "Select condition to view evidence",
                        available_conditions,
                    )

                    # Get first model's evidence (they're the same for same condition)
                    first_model = list(results[selected_condition].keys())[0]
                    evidence = results[selected_condition][first_model].get("evidence_shown", "")

                    st.code(evidence, language=None)

    # === AGENTS TAB ===
    with tab3:
        st.header("Agent Overview")

        agents = data["agents"]
        if not agents:
            st.warning("No agent data found.")
        else:
            # Agent cards
            cols = st.columns(len(agents))

            for i, (agent_id, agent_data) in enumerate(agents.items()):
                with cols[i]:
                    st.subheader(agent_id)

                    reliability = agent_data.get("computed_reliability", "N/A")
                    st.metric("Reliability", f"{reliability}%")

                    task = agent_data.get("task", {})
                    st.write(f"**Task:** {task.get('name', 'N/A')}")
                    st.caption(task.get("description", ""))

                    relationships = agent_data.get("relationships", {})
                    if relationships:
                        st.write("**Relationships:**")
                        for other_id, rel in relationships.items():
                            icon = "🤝" if rel == "cooperative" else "⚔️"
                            st.write(f"  {icon} {other_id}: {rel}")

            # Detailed stats
            st.subheader("Statement Breakdown")

            statements = data["statements"]
            if statements:
                # Per-agent stats
                agent_stats = []
                for agent_id in agents.keys():
                    agent_stmts = [s for s in statements if s["agent_id"] == agent_id]
                    total = len(agent_stmts)
                    truthful = sum(1 for s in agent_stmts if s["is_truthful"])

                    # By relationship
                    adversarial = [s for s in agent_stmts if s["relationship"] == "adversarial"]
                    adv_truthful = sum(1 for s in adversarial if s["is_truthful"])

                    cooperative = [s for s in agent_stmts if s["relationship"] == "cooperative"]
                    coop_truthful = sum(1 for s in cooperative if s["is_truthful"])

                    agent_stats.append({
                        "Agent": agent_id,
                        "Total": total,
                        "Truthful": truthful,
                        "Deceptive": total - truthful,
                        "Reliability": f"{100*truthful/total:.0f}%" if total > 0 else "N/A",
                        "Adversarial (T/F)": f"{adv_truthful}/{len(adversarial)-adv_truthful}" if adversarial else "N/A",
                        "Cooperative (T/F)": f"{coop_truthful}/{len(cooperative)-coop_truthful}" if cooperative else "N/A",
                    })

                stats_df = pd.DataFrame(agent_stats)
                st.dataframe(stats_df, use_container_width=True)

                # Lies by property
                st.subheader("Deceptions by Property")

                lies = [s for s in statements if not s["is_truthful"]]
                if lies:
                    lies_df = pd.DataFrame(lies)
                    by_prop = lies_df.groupby(["agent_id", "property_name"]).size().unstack(fill_value=0)
                    st.dataframe(by_prop, use_container_width=True)


if __name__ == "__main__":
    main()
