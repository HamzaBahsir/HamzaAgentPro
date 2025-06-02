from agentpro.tools import Tool
from typing import Dict, Any, Union
import openai # Assumes OpenAI is integrated in your project

class SignalDiagnosticAgent(Tool):
    name: str = "Signal Diagnostic Agent"
    description: str = (
        "Analyzes signal anomalies/features and provides diagnostic interpretation using LLM."
    )
    action_type: str = "diagnose_signal"
    input_format: str = """{
        'feature_summary': str,  # Description or features of the signal, e.g., "Strong 60Hz tone with broadband noise"
        'context': str           # Optional context like device type or known issues
    }"""

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Union[str, Dict[str, str]]]:
        feature_summary = input_data.get("feature_summary", "")
        context = input_data.get("context", "")

        prompt = self._build_prompt(feature_summary, context)
        client = OpenAI()

        try:
            response = client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert signal diagnostics engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            diagnosis = response.choices[0].message['content'].strip()

            return {
                "diagnosis": diagnosis,
                "message": "Signal diagnostic complete."
            }

        except Exception as e:
            return {
                "diagnosis": "An error occurred while processing your request.",
                "error": str(e)
            }

    def _build_prompt(self, feature_summary: str, context: str) -> str:
        return (
            f"The following signal features were detected: {feature_summary}.\n"
            f"Context: {context if context else 'No additional context'}.\n\n"
            "Please provide:\n"
            "1. A possible cause of the observed signal characteristics.\n"
            "2. Suggested next steps for debugging or investigation.\n"
            "3. Any related known patterns or issues.\n"
        )
