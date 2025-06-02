from agentpro.tools import Tool
from typing import Dict, Any, Union
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from jinja2 import Template
import io
import base64

class ReportAgent(Tool):
    name: str = "Signal Report Agent"
    description: str = "Generates structured reports and dashboards for signal analysis."
    action_type: str = "generate_report"
    input_format: str = """{
        'title': str,                       # Report title
        'summary': str,                     # High-level summary or diagnosis
        'metrics': Dict[str, float],        # e.g., {'Peak Frequency (Hz)': 60.0, 'SNR (dB)': 25.6}
        'time_series': Dict[str, List[float]], # e.g., {'time': [...], 'signal': [...]}
        'diagnosis': str,                   # Human-readable explanation
        'recommendations': str              # Suggestions for engineers
    }"""

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Union[str, Dict]]:
        report_md = self._generate_markdown_report(input_data)
        dashboard_json = self._generate_dashboard_data(input_data)
        return {
            "markdown_report": report_md,
            "dashboard_data": dashboard_json,
            "message": "Report and dashboard successfully generated."
        }

    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        template_str = """
# {{ title }}

## 🧠 Summary
{{ summary }}

## 📊 Key Metrics
{% for key, value in metrics.items() %}
- **{{ key }}**: {{ value }}
{% endfor %}

## 📈 Signal Plot
![Signal](data:image/png;base64,{{ plot_image }})

## 💡 Diagnosis
{{ diagnosis }}

## 🔧 Recommendations
{{ recommendations }}
"""
        # Generate the plot image
        plot_image = self._plot_to_base64(data["time_series"])

        # Render markdown with Jinja2
        template = Template(template_str)
        return template.render(
            title=data["title"],
            summary=data["summary"],
            metrics=data["metrics"],
            diagnosis=data["diagnosis"],
            recommendations=data["recommendations"],
            plot_image=plot_image
        )

    def _plot_to_base64(self, time_series: Dict[str, Any]) -> str:
        plt.figure(figsize=(8, 3))
        plt.plot(time_series["time"], time_series["signal"])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Signal Overview")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _generate_dashboard_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame({
            "Time": data["time_series"]["time"],
            "Signal": data["time_series"]["signal"]
        })
        fig = px.line(df, x="Time", y="Signal", title="Signal over Time")
        return {
            "plotly_json": fig.to_dict(),
            "metrics": data["metrics"]
        }
