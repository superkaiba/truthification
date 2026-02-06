#!/usr/bin/env python3
"""Generate static HTML site for GitHub Pages from experiment results.

This script generates:
1. An index page (docs/index.html) linking all experiments
2. Individual experiment dashboards (docs/experiments/{name}/index.html)
3. Copies any plots/images to the appropriate folders

Usage:
    python scripts/generate_static_site.py
    python scripts/generate_static_site.py --output-dir docs
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def load_json_safe(path: Path) -> dict | None:
    """Load JSON file, returning None on error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def find_experiments(results_dir: Path) -> list[dict]:
    """Find all experiment directories and extract metadata."""
    experiments = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith("."):
            continue

        exp_info = {
            "name": exp_dir.name,
            "path": exp_dir,
            "readme": None,
            "summary": None,
            "plots": [],
            "json_files": [],
            "mtime": exp_dir.stat().st_mtime,
        }

        # Look for README
        readme_path = exp_dir / "README.md"
        if readme_path.exists():
            exp_info["readme"] = readme_path.read_text()

        # Look for summary JSON
        for pattern in ["summary.json", "*_summary.json", "aggregate_*.json"]:
            for f in exp_dir.glob(pattern):
                data = load_json_safe(f)
                if data:
                    exp_info["summary"] = data
                    exp_info["summary_file"] = f.name
                    break
            if exp_info["summary"]:
                break

        # Find plots
        for ext in ["*.png", "*.jpg", "*.svg"]:
            exp_info["plots"].extend(exp_dir.glob(ext))

        # Find JSON result files
        for f in exp_dir.glob("*.json"):
            exp_info["json_files"].append(f.name)

        experiments.append(exp_info)

    # Sort by modification time (newest first)
    experiments.sort(key=lambda x: x["mtime"], reverse=True)
    return experiments


def extract_readme_title(readme: str) -> str:
    """Extract title from README markdown."""
    for line in readme.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def extract_readme_summary(readme: str) -> str:
    """Extract first paragraph as summary."""
    lines = readme.split("\n")
    in_content = False
    summary_lines = []

    for line in lines:
        if line.startswith("# "):
            in_content = True
            continue
        if in_content:
            if line.startswith("## "):
                break
            if line.strip():
                summary_lines.append(line.strip())
            elif summary_lines:
                break

    return " ".join(summary_lines[:3])[:300]


def generate_experiment_page(exp: dict, output_dir: Path) -> None:
    """Generate a single experiment's dashboard page."""
    exp_output = output_dir / "experiments" / exp["name"]
    exp_output.mkdir(parents=True, exist_ok=True)

    # Copy plots
    for plot in exp["plots"]:
        shutil.copy(plot, exp_output / plot.name)

    # Generate HTML
    title = extract_readme_title(exp["readme"]) if exp["readme"] else exp["name"]

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Truthification Experiments</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header a {{ color: #fff; opacity: 0.8; }}
        .header a:hover {{ opacity: 1; }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .card h2 {{ margin-top: 0; color: #444; }}
        .plots {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        .plot {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .plot-caption {{
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }}
        .readme {{
            line-height: 1.6;
        }}
        .readme h1, .readme h2, .readme h3 {{
            color: #333;
            margin-top: 1.5em;
        }}
        .readme code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .readme pre {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }}
        .readme table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        .readme th, .readme td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }}
        .readme th {{ background: #f5f5f5; }}
        .files {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .file-link {{
            display: inline-block;
            padding: 8px 15px;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 20px;
            text-decoration: none;
            font-size: 14px;
        }}
        .file-link:hover {{
            background: #bbdefb;
        }}
        .metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 8px;
            text-align: center;
            min-width: 120px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <a href="../../index.html">&larr; Back to Experiments</a>
        <h1>{title}</h1>
        <p>Last updated: {datetime.fromtimestamp(exp["mtime"]).strftime("%Y-%m-%d %H:%M")}</p>
    </div>
'''

    # Add summary metrics if available
    if exp["summary"]:
        html += '    <div class="card">\n        <h2>Summary Metrics</h2>\n        <div class="metrics">\n'
        summary = exp["summary"]

        # Handle different summary formats
        metrics_to_show = []
        if "metrics" in summary:
            m = summary["metrics"]
            metrics_to_show = [
                ("Selection Accuracy", f"{m.get('selection_accuracy', 0)*100:.1f}%"),
                ("Property Accuracy", f"{m.get('property_accuracy', 0)*100:.1f}%"),
                ("Rule Inference", f"{m.get('rule_inference_accuracy', 0)*100:.1f}%"),
            ]
        elif "results" in summary:
            # Aggregate format
            results = summary["results"]
            if results:
                avg_sel = sum(r.get("selection_accuracy", 0) for r in results) / len(results)
                metrics_to_show = [
                    ("Avg Selection Accuracy", f"{avg_sel*100:.1f}%"),
                    ("Conditions Tested", str(len(results))),
                ]
        elif isinstance(summary, dict):
            # Try to extract any accuracy metrics
            for key in ["selection_accuracy", "property_accuracy", "estimator_property_accuracy"]:
                if key in summary:
                    label = key.replace("_", " ").title()
                    val = summary[key]
                    if isinstance(val, float):
                        metrics_to_show.append((label, f"{val*100:.1f}%"))

        for label, value in metrics_to_show:
            html += f'''            <div class="metric">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
'''
        html += '        </div>\n    </div>\n'

    # Add plots
    if exp["plots"]:
        html += '    <div class="card">\n        <h2>Visualizations</h2>\n        <div class="plots">\n'
        for plot in exp["plots"]:
            caption = plot.stem.replace("_", " ").title()
            html += f'''            <div class="plot">
                <img src="{plot.name}" alt="{caption}">
                <div class="plot-caption">{caption}</div>
            </div>
'''
        html += '        </div>\n    </div>\n'

    # Add README content
    if exp["readme"]:
        # Simple markdown to HTML conversion
        readme_html = convert_markdown_to_html(exp["readme"])
        html += f'''    <div class="card readme">
        <h2>Documentation</h2>
        {readme_html}
    </div>
'''

    # Add file links
    if exp["json_files"]:
        html += '    <div class="card">\n        <h2>Data Files</h2>\n        <div class="files">\n'
        for f in exp["json_files"]:
            html += f'            <a class="file-link" href="../../results/{exp["name"]}/{f}">{f}</a>\n'
        html += '        </div>\n    </div>\n'

    html += '''</body>
</html>
'''

    (exp_output / "index.html").write_text(html)


def convert_markdown_to_html(md: str) -> str:
    """Simple markdown to HTML conversion."""
    import re

    html = md

    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Code blocks
    html = re.sub(r'```(\w*)\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)

    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)

    # Images (convert to local paths)
    html = re.sub(r'!\[([^\]]*)\]\(([^\)]+)\)', r'<img src="\2" alt="\1">', html)

    # Lists
    lines = html.split('\n')
    in_list = False
    result = []
    for line in lines:
        if line.strip().startswith('- '):
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{line.strip()[2:]}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            # Paragraphs
            if line.strip() and not line.strip().startswith('<'):
                result.append(f'<p>{line}</p>')
            else:
                result.append(line)
    if in_list:
        result.append('</ul>')

    return '\n'.join(result)


def generate_index_page(experiments: list[dict], output_dir: Path) -> None:
    """Generate the main index page."""
    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Truthification Experiments Dashboard</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 0;
            opacity: 0.9;
            font-size: 1.1em;
        }
        .experiments {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }
        .experiment-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .experiment-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .experiment-card a {
            display: block;
            padding: 25px;
            text-decoration: none;
            color: inherit;
        }
        .experiment-card h2 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.3em;
        }
        .experiment-card .summary {
            color: #666;
            font-size: 0.95em;
            line-height: 1.5;
            margin-bottom: 15px;
        }
        .experiment-card .meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.85em;
            color: #999;
        }
        .experiment-card .tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
        }
        .tag-plot { background: #e3f2fd; color: #1976d2; }
        .tag-data { background: #e8f5e9; color: #388e3c; }
        .tag-doc { background: #fff3e0; color: #f57c00; }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        .footer a {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Truthification Experiments</h1>
        <p>Research on teaching LLMs to distinguish facts, claims, and opinions</p>
    </div>

    <div class="experiments">
'''

    for exp in experiments:
        title = extract_readme_title(exp["readme"]) if exp["readme"] else exp["name"].replace("_", " ").title()
        summary = extract_readme_summary(exp["readme"]) if exp["readme"] else f"{len(exp['json_files'])} data files"
        date = datetime.fromtimestamp(exp["mtime"]).strftime("%Y-%m-%d")

        # Tags
        tags = []
        if exp["plots"]:
            tags.append(f'<span class="tag tag-plot">{len(exp["plots"])} plots</span>')
        if exp["json_files"]:
            tags.append(f'<span class="tag tag-data">{len(exp["json_files"])} files</span>')
        if exp["readme"]:
            tags.append('<span class="tag tag-doc">documented</span>')
        tags_html = "".join(tags)

        html += f'''        <div class="experiment-card">
            <a href="experiments/{exp["name"]}/index.html">
                <h2>{title}</h2>
                <p class="summary">{summary}</p>
                <div class="tags">{tags_html}</div>
                <div class="meta">
                    <span>{exp["name"]}</span>
                    <span>{date}</span>
                </div>
            </a>
        </div>
'''

    html += f'''    </div>

    <div class="footer">
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><a href="https://github.com/thomasjiralerspong/truthification">View on GitHub</a></p>
    </div>
</body>
</html>
'''

    (output_dir / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description="Generate static site for GitHub Pages")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                       help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=Path, default=Path("docs"),
                       help="Output directory for static site")
    args = parser.parse_args()

    print(f"Scanning experiments in {args.results_dir}...")
    experiments = find_experiments(args.results_dir)
    print(f"Found {len(experiments)} experiments")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate index page
    print("Generating index page...")
    generate_index_page(experiments, args.output_dir)

    # Generate individual experiment pages
    for exp in experiments:
        print(f"  Generating page for {exp['name']}...")
        generate_experiment_page(exp, args.output_dir)

    # Create a simple .nojekyll file for GitHub Pages
    (args.output_dir / ".nojekyll").touch()

    print(f"\nStatic site generated in {args.output_dir}/")
    print(f"  Index: {args.output_dir}/index.html")
    print(f"  Experiments: {args.output_dir}/experiments/")


if __name__ == "__main__":
    main()
