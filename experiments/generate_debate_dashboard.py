#!/usr/bin/env python3
"""Generate an HTML dashboard for debate structure experiment results with full transcripts and accuracy progression."""

import json
from pathlib import Path
from datetime import datetime


def load_game(path: Path) -> dict:
    """Load a game result JSON file."""
    with open(path) as f:
        return json.load(f)


def get_turn_structure_description(turn_structure: str, oracle_timing: str) -> str:
    """Get a human-readable description of the debate structure."""
    descriptions = {
        "interleaved": "Agents alternate statements (A₁ → B₁ → A₂ → B₂ → ...)",
        "batch": "Each agent speaks in a block (A₁ → A₂ → A₃ → B₁ → B₂ → B₃)",
        "simultaneous": "Both agents submit statements at the same time (no back-and-forth)",
        "sequential": "One agent completes all statements, then the other responds",
    }
    oracle_desc = {
        "before_response": "Statements → 🔮 Oracle → Agents respond to oracle",
        "after_statements": "Statements → 🔮 Oracle (no agent response)",
    }
    ts_desc = descriptions.get(turn_structure, turn_structure)
    ot_desc = oracle_desc.get(oracle_timing, oracle_timing)
    return f"{ts_desc}<br><small><b>Oracle timing:</b> {ot_desc}</small>"


def render_transcript(game: dict) -> str:
    """Render the conversation transcript as HTML."""
    rounds = game.get("rounds", [])
    world_state = game.get("world_state", {})
    computed_values = world_state.get("computed_values", {})
    config = game.get("config", {})
    turn_structure = config.get("turn_structure", "interleaved")
    oracle_timing = config.get("oracle_timing", "before_response")

    # Agent colors
    agent_colors = {
        "Agent_A": ("#E3F2FD", "#1976D2"),  # Light blue, blue border
        "Agent_B": ("#FCE4EC", "#C2185B"),  # Light pink, pink border
    }

    html = []
    
    # Add turn structure description at the top
    structure_desc = get_turn_structure_description(turn_structure, oracle_timing)
    html.append(f'''
        <div class="structure-info" style="background:#FFF3E0; padding:10px; border-radius:8px; margin-bottom:15px; border-left:4px solid #FF9800;">
            <b>Debate Structure:</b> {structure_desc}
        </div>
    ''')

    def render_oracle(oracle_data):
        """Render oracle query as HTML."""
        if not oracle_data:
            return ""
        query_type = oracle_data.get("query_type", "unknown")
        obj_id = oracle_data.get("object_id", "unknown")
        result = oracle_data.get("result", "N/A")
        if query_type == "property":
            prop_name = oracle_data.get("property_name", "unknown")
            query_text = f"{prop_name} of {obj_id}"
        else:
            query_text = f"{query_type} of {obj_id}"
        return f'''
            <div class="message oracle" style="background:#E8F5E9; border-left:4px solid #4CAF50;">
                <div class="agent-name" style="color:#4CAF50;">🔮 Oracle</div>
                <div class="message-text"><b>Query:</b> {query_text}<br><b>Result:</b> {result}</div>
            </div>
        '''

    for round_data in rounds:
        round_num = round_data.get("round_number", "?")
        html.append(f'<div class="round"><h4>Round {round_num}</h4>')

        oracle = round_data.get("oracle_query")
        all_statements = round_data.get("agent_statements", [])
        
        # Split statements into initial and oracle responses
        initial_statements = [s for s in all_statements if not s.get("is_oracle_response")]
        oracle_responses = [s for s in all_statements if s.get("is_oracle_response")]
        
        # For rendering, we use initial_statements first, then oracle, then responses
        statements = initial_statements
        
        # Group and render statements based on turn structure
        if turn_structure == "simultaneous":
            # Group simultaneous statements together
            html.append('<div class="simultaneous-group" style="display:flex; gap:10px; flex-wrap:wrap;">')
            for i, stmt in enumerate(statements):
                agent_id = stmt.get("agent_id", "Unknown")
                text = stmt.get("text", "")
                bg, border = agent_colors.get(agent_id, ("#F5F5F5", "#9E9E9E"))
                turn_num = i // 2 + 1  # Group pairs as simultaneous turns
                
                html.append(f'''
                    <div class="message" style="background:{bg}; border-left:4px solid {border}; flex:1; min-width:45%;">
                        <div class="agent-name" style="color:{border};">
                            <span class="turn-badge" style="background:{border}; color:white; padding:2px 6px; border-radius:10px; font-size:10px; margin-right:5px;">T{turn_num}</span>
                            {agent_id} <span style="font-size:10px; color:#666;">⚡ simultaneous</span>
                        </div>
                        <div class="message-text">{text}</div>
                    </div>
                ''')
                # Close and reopen group every 2 statements
                if (i + 1) % 2 == 0 and i < len(statements) - 1:
                    html.append('</div><div class="simultaneous-group" style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">')
            html.append('</div>')
        else:
            # Sequential display with turn numbers and structure indicators
            for i, stmt in enumerate(statements):
                agent_id = stmt.get("agent_id", "Unknown")
                text = stmt.get("text", "")
                bg, border = agent_colors.get(agent_id, ("#F5F5F5", "#9E9E9E"))
                
                # Determine turn indicator based on structure
                if turn_structure == "interleaved":
                    turn_label = f"#{i+1}"
                    flow_icon = "↔️" if i > 0 else ""
                elif turn_structure == "batch":
                    # Count position within agent's batch
                    same_agent_count = sum(1 for s in statements[:i+1] if s.get("agent_id") == agent_id)
                    turn_label = f"{agent_id[0]}{same_agent_count}"  # e.g., "A1", "A2", "B1"
                    flow_icon = "📦"
                elif turn_structure == "sequential":
                    same_agent_count = sum(1 for s in statements[:i+1] if s.get("agent_id") == agent_id)
                    turn_label = f"{agent_id[0]}{same_agent_count}"
                    flow_icon = "➡️"
                else:
                    turn_label = f"#{i+1}"
                    flow_icon = ""

                html.append(f'''
                    <div class="message" style="background:{bg}; border-left:4px solid {border};">
                        <div class="agent-name" style="color:{border};">
                            <span class="turn-badge" style="background:{border}; color:white; padding:2px 6px; border-radius:10px; font-size:10px; margin-right:5px;">{turn_label}</span>
                            {agent_id} {flow_icon}
                        </div>
                        <div class="message-text">{text}</div>
                    </div>
                ''')

        # Show oracle after initial statements (regardless of timing mode)
        if oracle:
            html.append(render_oracle(oracle))
        
        # Show oracle responses (agents responding to oracle result)
        if oracle_responses:
            html.append('<div class="oracle-responses" style="border-left: 3px solid #4CAF50; margin-left: 10px; padding-left: 10px;">')
            html.append('<small style="color:#4CAF50; font-weight:bold;">↳ Responses to Oracle:</small>')
            for stmt in oracle_responses:
                agent_id = stmt.get("agent_id", "Unknown")
                text = stmt.get("text", "")
                bg, border = agent_colors.get(agent_id, ("#F5F5F5", "#9E9E9E"))
                html.append(f'''
                    <div class="message" style="background:{bg}; border-left:4px solid {border};">
                        <div class="agent-name" style="color:{border};">
                            <span class="turn-badge" style="background:#4CAF50; color:white; padding:2px 6px; border-radius:10px; font-size:10px; margin-right:5px;">🔮</span>
                            {agent_id} <span style="font-size:10px; color:#666;">responding to oracle</span>
                        </div>
                        <div class="message-text">{text}</div>
                    </div>
                ''')
            html.append('</div>')

        # Observer picks
        picks = round_data.get("observer_current_picks", [])
        reasoning = round_data.get("observer_reasoning", "")
        if picks or reasoning:
            picks_str = ", ".join(f"{p} ({computed_values.get(p, '?')})" for p in picks)
            html.append(f'''
                <div class="message observer">
                    <div class="agent-name">Observer (Judge)</div>
                    <div class="message-text">
                        <b>Picks:</b> {picks_str}<br>
                        <b>Reasoning:</b> {reasoning[:500]}{'...' if len(reasoning) > 500 else ''}
                    </div>
                </div>
            ''')

        html.append('</div>')  # Close round

    return "\n".join(html)


def render_world_info(game: dict) -> str:
    """Render world state (objects, values, rule) as HTML."""
    world_state = game.get("world_state", {})
    objects = world_state.get("objects", {})
    computed_values = world_state.get("computed_values", {})
    value_rule = game.get("value_rule", {})
    agents = game.get("agents", [])

    # Value rule
    rule_html = f'''
        <div class="rule-box">
            <b>Value Rule:</b> {value_rule.get("description", "N/A")}<br>
            <b>Conditions:</b> {", ".join(f"{c.get('description')}: +{c.get('bonus')}" for c in value_rule.get("conditions", []))}
        </div>
    '''

    # Agents
    agents_html = '<div class="agents-box">'
    for agent in agents:
        interest = agent.get("interest", {})
        agents_html += f'<span class="agent-tag">{agent.get("id")}: {interest.get("description", "N/A")}</span>'
    agents_html += '</div>'

    # Objects table
    table_html = '''
        <table class="objects-table">
            <thead>
                <tr><th>ID</th><th>Color</th><th>Shape</th><th>Size</th><th>Material</th><th>Value</th></tr>
            </thead>
            <tbody>
    '''

    # Sort by value descending
    sorted_objects = sorted(objects.items(), key=lambda x: computed_values.get(x[0], 0), reverse=True)
    for obj_id, obj_data in sorted_objects:
        props = obj_data.get("properties", {})
        value = computed_values.get(obj_id, 0)
        table_html += f'''
            <tr>
                <td>{obj_id}</td>
                <td>{props.get("color", "?")}</td>
                <td>{props.get("shape", "?")}</td>
                <td>{props.get("size", "?")}</td>
                <td>{props.get("material", "?")}</td>
                <td><b>{value}</b></td>
            </tr>
        '''

    table_html += '</tbody></table>'

    return rule_html + agents_html + table_html


def render_metrics(game: dict) -> str:
    """Render game metrics as HTML."""
    metrics = game.get("metrics", {})
    estimator_metrics = game.get("estimator_metrics", {})

    obs_prop = metrics.get("property_accuracy", 0) * 100
    obs_rule = metrics.get("rule_inference_accuracy", 0) * 100
    sel_acc = metrics.get("selection_accuracy", 0) * 100

    est_prop = estimator_metrics.get("property_accuracy", 0) * 100 if estimator_metrics else 0
    est_rule = estimator_metrics.get("rule_inference_accuracy", 0) * 100 if estimator_metrics else 0

    return f'''
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">{est_prop:.0f}%</div>
                <div class="metric-label">Est. Prop Acc</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{obs_prop:.0f}%</div>
                <div class="metric-label">Obs. Prop Acc</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{sel_acc:.0f}%</div>
                <div class="metric-label">Selection Acc</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{est_prop - obs_prop:+.0f}%</div>
                <div class="metric-label">Est. Advantage</div>
            </div>
        </div>
    '''


def render_accuracy_progression(game: dict) -> str:
    """Render accuracy progression chart and table as HTML."""
    accuracy_progression = game.get("accuracy_progression", [])
    rounds = game.get("rounds", [])

    if not accuracy_progression and rounds:
        # Try to build from round_metrics
        accuracy_progression = []
        for r in rounds:
            rm = r.get("round_metrics", {})
            if rm:
                accuracy_progression.append({
                    "round": r.get("round_number", 0),
                    "judge_property_accuracy": rm.get("judge_property_accuracy", 0),
                    "judge_rule_accuracy": rm.get("judge_rule_accuracy", 0),
                    "estimator_property_accuracy": rm.get("estimator_property_accuracy", 0),
                    "estimator_rule_accuracy": rm.get("estimator_rule_accuracy", 0),
                    "cumulative_value": rm.get("cumulative_value", 0),
                    "decision_quality": rm.get("decision_quality", 0),
                    "agent_success": rm.get("agent_success", {}),
                })

    if not accuracy_progression:
        return "<p>No accuracy progression data available.</p>"

    # Extract agent IDs from first round with agent_success
    agent_ids = []
    for p in accuracy_progression:
        agent_success = p.get("agent_success", {})
        if agent_success:
            agent_ids = list(agent_success.keys())
            break

    # Build table with agent success columns
    agent_headers = "".join(f"<th>{aid} Success</th>" for aid in agent_ids)
    table_html = f'''
        <h4>Accuracy Progression Table</h4>
        <table class="objects-table">
            <thead>
                <tr>
                    <th>Round</th>
                    <th>Judge Prop</th>
                    <th>Est Prop</th>
                    <th>Judge Rule</th>
                    <th>Est Rule</th>
                    {agent_headers}
                    <th>Cumul Value</th>
                    <th>Decision Q</th>
                </tr>
            </thead>
            <tbody>
    '''

    for p in accuracy_progression:
        judge_prop = p.get("judge_property_accuracy", 0) * 100
        est_prop = p.get("estimator_property_accuracy", 0) * 100
        judge_rule = p.get("judge_rule_accuracy", 0) * 100
        est_rule = p.get("estimator_rule_accuracy", 0) * 100
        cumul = p.get("cumulative_value", 0)
        dq = p.get("decision_quality", 0) * 100
        
        # Agent success columns
        agent_success = p.get("agent_success", {})
        agent_cells = ""
        for aid in agent_ids:
            as_data = agent_success.get(aid, {})
            rate = as_data.get("rate", 0) * 100
            matched = as_data.get("matched", 0)
            total = as_data.get("total", 0)
            agent_cells += f"<td>{rate:.0f}% ({matched}/{total})</td>"

        table_html += f'''
            <tr>
                <td>{p.get("round", "?")}</td>
                <td>{judge_prop:.1f}%</td>
                <td>{est_prop:.1f}%</td>
                <td>{judge_rule:.1f}%</td>
                <td>{est_rule:.1f}%</td>
                {agent_cells}
                <td>{cumul}</td>
                <td>{dq:.1f}%</td>
            </tr>
        '''

    table_html += '</tbody></table>'

    # Build simple SVG line charts
    chart_html = render_progression_chart(accuracy_progression)
    agent_chart_html = render_agent_success_chart(accuracy_progression, agent_ids)

    return chart_html + agent_chart_html + table_html


def render_progression_chart(progression: list) -> str:
    """Render a simple SVG line chart showing accuracy over rounds."""
    if not progression:
        return ""

    width = 400
    height = 200
    padding = 40
    chart_width = width - 2 * padding
    chart_height = height - 2 * padding

    n_rounds = len(progression)
    if n_rounds < 2:
        return ""

    x_step = chart_width / (n_rounds - 1) if n_rounds > 1 else chart_width

    def y_coord(value):
        """Convert 0-1 value to y coordinate (inverted because SVG y goes down)."""
        return padding + chart_height * (1 - value)

    def x_coord(idx):
        """Convert round index to x coordinate."""
        return padding + idx * x_step

    # Build paths for each metric
    judge_prop_points = []
    est_prop_points = []
    judge_rule_points = []
    est_rule_points = []

    for i, p in enumerate(progression):
        x = x_coord(i)
        judge_prop_points.append(f"{x},{y_coord(p.get('judge_property_accuracy', 0))}")
        est_prop_points.append(f"{x},{y_coord(p.get('estimator_property_accuracy', 0))}")
        judge_rule_points.append(f"{x},{y_coord(p.get('judge_rule_accuracy', 0))}")
        est_rule_points.append(f"{x},{y_coord(p.get('estimator_rule_accuracy', 0))}")

    # Legend colors
    colors = {
        "Judge Prop": "#1976D2",
        "Est Prop": "#4CAF50",
        "Judge Rule": "#9C27B0",
        "Est Rule": "#FF9800",
    }

    svg = f'''
        <svg width="{width}" height="{height + 60}" style="background: #fafafa; border-radius: 8px;">
            <!-- Grid lines -->
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#ddd" stroke-width="1"/>
            <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#ddd" stroke-width="1"/>

            <!-- Y-axis labels -->
            <text x="{padding - 5}" y="{padding}" text-anchor="end" font-size="10" fill="#666">100%</text>
            <text x="{padding - 5}" y="{padding + chart_height/2}" text-anchor="end" font-size="10" fill="#666">50%</text>
            <text x="{padding - 5}" y="{height - padding}" text-anchor="end" font-size="10" fill="#666">0%</text>

            <!-- X-axis labels -->
    '''

    for i, p in enumerate(progression):
        x = x_coord(i)
        svg += f'<text x="{x}" y="{height - padding + 15}" text-anchor="middle" font-size="10" fill="#666">R{p.get("round", i+1)}</text>'

    # Draw lines
    svg += f'''
            <!-- Judge Property (blue) -->
            <polyline points="{' '.join(judge_prop_points)}" fill="none" stroke="{colors['Judge Prop']}" stroke-width="2"/>

            <!-- Estimator Property (green) -->
            <polyline points="{' '.join(est_prop_points)}" fill="none" stroke="{colors['Est Prop']}" stroke-width="2"/>

            <!-- Judge Rule (purple) -->
            <polyline points="{' '.join(judge_rule_points)}" fill="none" stroke="{colors['Judge Rule']}" stroke-width="2" stroke-dasharray="5,5"/>

            <!-- Estimator Rule (orange) -->
            <polyline points="{' '.join(est_rule_points)}" fill="none" stroke="{colors['Est Rule']}" stroke-width="2" stroke-dasharray="5,5"/>

            <!-- Legend -->
            <rect x="{padding}" y="{height + 5}" width="15" height="10" fill="{colors['Judge Prop']}"/>
            <text x="{padding + 20}" y="{height + 13}" font-size="10" fill="#333">Judge Prop</text>

            <rect x="{padding + 90}" y="{height + 5}" width="15" height="10" fill="{colors['Est Prop']}"/>
            <text x="{padding + 110}" y="{height + 13}" font-size="10" fill="#333">Est Prop</text>

            <rect x="{padding}" y="{height + 22}" width="15" height="10" fill="{colors['Judge Rule']}"/>
            <text x="{padding + 20}" y="{height + 30}" font-size="10" fill="#333">Judge Rule</text>

            <rect x="{padding + 90}" y="{height + 22}" width="15" height="10" fill="{colors['Est Rule']}"/>
            <text x="{padding + 110}" y="{height + 30}" font-size="10" fill="#333">Est Rule</text>
        </svg>
    '''

    return f'<h4>Accuracy Over Rounds</h4><div class="chart-container">{svg}</div>'


def render_agent_success_chart(progression: list, agent_ids: list) -> str:
    """Render a simple SVG line chart showing agent success rates over rounds."""
    if not progression or not agent_ids:
        return ""

    width = 400
    height = 200
    padding = 40
    chart_width = width - 2 * padding
    chart_height = height - 2 * padding

    n_rounds = len(progression)
    if n_rounds < 2:
        return ""

    x_step = chart_width / (n_rounds - 1) if n_rounds > 1 else chart_width

    def y_coord(value):
        """Convert 0-1 value to y coordinate (inverted because SVG y goes down)."""
        return padding + chart_height * (1 - value)

    def x_coord(idx):
        """Convert round index to x coordinate."""
        return padding + idx * x_step

    # Colors for agents
    agent_colors = {
        agent_ids[0]: "#E53935" if len(agent_ids) > 0 else "#666",  # Red for Agent_A
        agent_ids[1]: "#1E88E5" if len(agent_ids) > 1 else "#666",  # Blue for Agent_B
    }

    # Build paths for each agent
    agent_points = {aid: [] for aid in agent_ids}

    for i, p in enumerate(progression):
        x = x_coord(i)
        agent_success = p.get("agent_success", {})
        for aid in agent_ids:
            rate = agent_success.get(aid, {}).get("rate", 0)
            agent_points[aid].append(f"{x},{y_coord(rate)}")

    svg = f'''
        <svg width="{width}" height="{height + 40}" style="background: #fafafa; border-radius: 8px;">
            <!-- Grid lines -->
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#ddd" stroke-width="1"/>
            <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#ddd" stroke-width="1"/>
            
            <!-- 50% reference line -->
            <line x1="{padding}" y1="{y_coord(0.5)}" x2="{width - padding}" y2="{y_coord(0.5)}" stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>

            <!-- Y-axis labels -->
            <text x="{padding - 5}" y="{padding}" text-anchor="end" font-size="10" fill="#666">100%</text>
            <text x="{padding - 5}" y="{y_coord(0.5)}" text-anchor="end" font-size="10" fill="#666">50%</text>
            <text x="{padding - 5}" y="{height - padding}" text-anchor="end" font-size="10" fill="#666">0%</text>

            <!-- X-axis labels -->
    '''

    for i, p in enumerate(progression):
        x = x_coord(i)
        svg += f'<text x="{x}" y="{height - padding + 15}" text-anchor="middle" font-size="10" fill="#666">R{p.get("round", i+1)}</text>'

    # Draw lines for each agent
    for aid in agent_ids:
        color = agent_colors.get(aid, "#666")
        points_str = ' '.join(agent_points[aid])
        svg += f'''
            <polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="2"/>
        '''

    # Legend
    legend_x = padding
    for i, aid in enumerate(agent_ids):
        color = agent_colors.get(aid, "#666")
        offset = i * 100
        svg += f'''
            <rect x="{legend_x + offset}" y="{height + 5}" width="15" height="10" fill="{color}"/>
            <text x="{legend_x + offset + 20}" y="{height + 13}" font-size="10" fill="#333">{aid}</text>
        '''

    svg += '</svg>'

    return f'<h4>Agent Success Over Rounds</h4><div class="chart-container">{svg}</div>'


def render_aggregate_progression(summary_data: list) -> str:
    """Render aggregate accuracy progression comparison across conditions."""
    html = ['<h3>Accuracy Progression Comparison</h3>']

    # Create a comparison table
    html.append('''
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Condition</th>
                    <th>Round 1 Est%</th>
                    <th>Round 2 Est%</th>
                    <th>Round 3 Est%</th>
                    <th>Change R1→R3</th>
                </tr>
            </thead>
            <tbody>
    ''')

    for cond in summary_data:
        prog = cond.get("accuracy_progression", [])
        ts = cond.get("turn_structure", "?")
        ot = cond.get("oracle_timing", "?")

        r1 = prog[0].get("estimator_property_accuracy", 0) * 100 if len(prog) > 0 else 0
        r2 = prog[1].get("estimator_property_accuracy", 0) * 100 if len(prog) > 1 else 0
        r3 = prog[2].get("estimator_property_accuracy", 0) * 100 if len(prog) > 2 else 0
        change = r3 - r1 if prog else 0

        html.append(f'''
            <tr>
                <td><span class="tag tag-ts">{ts}</span> <span class="tag tag-ot">{ot}</span></td>
                <td>{r1:.1f}%</td>
                <td>{r2:.1f}%</td>
                <td>{r3:.1f}%</td>
                <td class="{'good' if change > 0 else 'bad' if change < 0 else ''}">{change:+.1f}%</td>
            </tr>
        ''')

    html.append('</tbody></table>')

    return '\n'.join(html)


def generate_dashboard(results_dir: Path, output_path: Path):
    """Generate the full HTML dashboard."""

    # Load summary
    summary_path = results_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    # Check if summary is the new format (list) or old format (dict)
    if isinstance(summary, list):
        summary_data = summary
        config = summary_data[0].get("config", {}) if summary_data else {}
        results = summary_data
    else:
        config = summary.get("config", {})
        results = summary.get("results", [])
        summary_data = results

    # Load all game files
    games = {}
    for game_file in results_dir.glob("game_*.json"):
        game_data = load_game(game_file)
        # Extract turn_structure and oracle_timing from filename
        name = game_file.stem  # e.g., "game_interleaved_before_response"
        parts = name.replace("game_", "").split("_")
        if len(parts) >= 2:
            ts = parts[0]
            ot = "_".join(parts[1:])
            games[(ts, ot)] = game_data

    # Also try loading from all_results.json
    all_results_path = results_dir / "all_results.json"
    if all_results_path.exists():
        with open(all_results_path) as f:
            all_results = json.load(f)
        # Use first result per condition
        for r in all_results:
            key = (r.get("turn_structure"), r.get("oracle_timing"))
            if key not in games:
                games[key] = r

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Debate Structure Experiment Dashboard</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f0f2f5;
            line-height: 1.5;
        }}
        h1, h2, h3, h4 {{ color: #333; margin-top: 0; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .summary-table th, .summary-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .summary-table th {{ background: #f8f9fa; font-weight: 600; }}
        .summary-table tr:hover {{ background: #f8f9fa; }}
        .good {{ background: #d4edda !important; color: #155724; }}
        .bad {{ background: #f8d7da !important; color: #721c24; }}

        /* Game cards */
        .game-section {{
            margin: 30px 0;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
        }}
        .game-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .game-header:hover {{ opacity: 0.95; }}
        .game-header h3 {{ margin: 0; }}
        .game-content {{
            display: none;
            padding: 20px;
            background: white;
        }}
        .game-content.active {{ display: block; }}

        /* Transcript */
        .transcript {{ max-height: 600px; overflow-y: auto; }}
        .round {{ margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .round h4 {{ margin: 0 0 10px 0; color: #666; }}
        .message {{
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 8px;
        }}
        .agent-name {{ font-weight: bold; margin-bottom: 4px; font-size: 14px; }}
        .message-text {{ color: #333; }}
        .oracle {{
            background: #FFF8E1;
            border-left: 4px solid #FFC107;
        }}
        .oracle .agent-name {{ color: #F57F17; }}
        .observer {{
            background: #E8F5E9;
            border-left: 4px solid #4CAF50;
        }}
        .observer .agent-name {{ color: #2E7D32; }}

        /* World info */
        .rule-box {{
            background: #fff3e0;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .agents-box {{ margin: 10px 0; }}
        .agent-tag {{
            display: inline-block;
            padding: 4px 12px;
            margin: 4px;
            border-radius: 20px;
            font-size: 13px;
        }}
        .agent-tag:nth-child(1) {{ background: #E3F2FD; color: #1976D2; }}
        .agent-tag:nth-child(2) {{ background: #FCE4EC; color: #C2185B; }}
        .objects-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .objects-table th, .objects-table td {{
            padding: 8px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }}
        .objects-table th {{ background: #f5f5f5; }}

        /* Metrics */
        .metrics-grid {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            min-width: 100px;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ font-size: 12px; color: #666; }}

        /* Tabs */
        .tabs {{ display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 15px; }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
        }}
        .tab:hover {{ background: #f5f5f5; }}
        .tab.active {{ border-bottom-color: #667eea; color: #667eea; font-weight: 600; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Toggle arrow */
        .toggle-arrow {{ font-size: 20px; transition: transform 0.3s; }}
        .game-section.open .toggle-arrow {{ transform: rotate(180deg); }}

        /* Tags */
        .tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin: 2px;
        }}
        .tag-ts {{ background: #e3f2fd; color: #1976d2; }}
        .tag-ot {{ background: #fce4ec; color: #c2185b; }}

        /* Chart */
        .chart-container {{
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <h1>Debate Structure Experiment Dashboard</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
       Config: {config.get("n_objects", "?")} objects, {config.get("n_rounds", "?")} rounds,
       oracle budget {config.get("oracle_budget", "?")}</p>

    <div class="card">
        <h2>Results Summary</h2>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Turn Structure</th>
                    <th>Oracle Timing</th>
                    <th>Est. Prop Acc</th>
                    <th>Obs. Prop Acc</th>
                    <th>Est. Advantage</th>
                    <th>Selection Acc</th>
                </tr>
            </thead>
            <tbody>
'''

    # Add summary rows
    for r in summary_data:
        if isinstance(r, dict) and "error" not in r:
            # Handle both old and new summary formats
            if "estimator_property_accuracy_mean" in r:
                # New format (aggregate)
                est_acc = r.get("estimator_property_accuracy_mean", 0) * 100
                obs_acc = r.get("observer_property_accuracy_mean", 0) * 100
                sel_acc = r.get("selection_accuracy_mean", 0) * 100
            else:
                # Old format or per-game
                est_acc = r.get("estimator_property_accuracy", 0) * 100
                obs_acc = r.get("observer_property_accuracy", 0) * 100
                sel_acc = r.get("selection_accuracy", 0) * 100

            adv = est_acc - obs_acc

            row_class = "good" if est_acc >= 80 else ("bad" if est_acc < 30 else "")

            html += f'''
                <tr class="{row_class}">
                    <td><span class="tag tag-ts">{r.get("turn_structure")}</span></td>
                    <td><span class="tag tag-ot">{r.get("oracle_timing")}</span></td>
                    <td><b>{est_acc:.0f}%</b></td>
                    <td>{obs_acc:.0f}%</td>
                    <td>{adv:+.0f}%</td>
                    <td>{sel_acc:.0f}%</td>
                </tr>
'''

    html += '''
            </tbody>
        </table>
    </div>
'''

    # Add aggregate progression comparison
    if summary_data and any(s.get("accuracy_progression") for s in summary_data):
        html += f'<div class="card">{render_aggregate_progression(summary_data)}</div>'

    html += '''
    <h2>Individual Games (click to expand)</h2>
'''

    # Add game sections
    turn_structures = ["interleaved", "batch", "simultaneous", "sequential"]
    oracle_timings = ["before_response", "after_statements"]

    for ts in turn_structures:
        for ot in oracle_timings:
            game = games.get((ts, ot))
            if not game:
                continue

            # Get metrics for header
            est_metrics = game.get("estimator_metrics", {})
            metrics = game.get("metrics", {})
            est_prop = est_metrics.get("property_accuracy", 0) * 100 if est_metrics else 0
            obs_prop = metrics.get("property_accuracy", 0) * 100

            game_id = f"{ts}_{ot}"

            html += f'''
    <div class="game-section" id="game-{game_id}">
        <div class="game-header" onclick="toggleGame('{game_id}')">
            <h3>
                <span class="tag tag-ts">{ts}</span>
                <span class="tag tag-ot">{ot}</span>
                &nbsp; Est: {est_prop:.0f}% | Obs: {obs_prop:.0f}%
            </h3>
            <span class="toggle-arrow">V</span>
        </div>
        <div class="game-content" id="content-{game_id}">
            <div class="tabs">
                <div class="tab active" onclick="showTab('{game_id}', 'transcript')">Transcript</div>
                <div class="tab" onclick="showTab('{game_id}', 'world')">World & Objects</div>
                <div class="tab" onclick="showTab('{game_id}', 'metrics')">Metrics</div>
                <div class="tab" onclick="showTab('{game_id}', 'progression')">Accuracy Progression</div>
            </div>

            <div class="tab-content active" id="{game_id}-transcript">
                <div class="transcript">
                    {render_transcript(game)}
                </div>
            </div>

            <div class="tab-content" id="{game_id}-world">
                {render_world_info(game)}
            </div>

            <div class="tab-content" id="{game_id}-metrics">
                {render_metrics(game)}
            </div>

            <div class="tab-content" id="{game_id}-progression">
                {render_accuracy_progression(game)}
            </div>
        </div>
    </div>
'''

    html += '''
    <script>
        function toggleGame(gameId) {
            const section = document.getElementById('game-' + gameId);
            const content = document.getElementById('content-' + gameId);
            section.classList.toggle('open');
            content.classList.toggle('active');
        }

        function showTab(gameId, tabName) {
            // Hide all tab contents for this game
            const contents = document.querySelectorAll('#content-' + gameId + ' .tab-content');
            contents.forEach(c => c.classList.remove('active'));

            // Deactivate all tabs
            const tabs = document.querySelectorAll('#content-' + gameId + ' .tab');
            tabs.forEach(t => t.classList.remove('active'));

            // Show selected tab content
            document.getElementById(gameId + '-' + tabName).classList.add('active');

            // Activate clicked tab
            event.target.classList.add('active');
        }

        // Expand first game by default
        document.addEventListener('DOMContentLoaded', function() {
            const firstGame = document.querySelector('.game-section');
            if (firstGame) {
                firstGame.classList.add('open');
                firstGame.querySelector('.game-content').classList.add('active');
            }
        });
    </script>
</body>
</html>
'''

    # Write output
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard generated: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use provided path
        results_path = Path(sys.argv[1])
        if results_path.is_dir():
            output_path = results_path / "dashboard.html"
            generate_dashboard(results_path, output_path)
            print(f"Open: {output_path}")
        else:
            print(f"Not a directory: {results_path}")
    else:
        # Find most recent in default location
        results_dir = Path("outputs/debate_structure")
        subdirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)
        if subdirs:
            latest = subdirs[0]
            output_path = latest / "dashboard.html"
            generate_dashboard(latest, output_path)
            print(f"Open: {output_path}")
        else:
            print(f"No results found in {results_dir}")
