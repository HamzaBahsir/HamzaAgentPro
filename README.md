# Signal Analysis App
Signal Analysis App is a Streamlit-powered AI assistant that uses a modular ReAct-style agent pipeline (built using AgentPro) for generating, analyzing, and diagnosing signals. It’s ideal for simulating sensor data, running frequency analysis, detecting anomalies, classifying signals, and generating reports — all in one UI.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License: Apache 2.0">
</p>

## 📚 Features
🔊 Signal Generation: sine, square, chirp, multi-tone, and noisy waveforms

⚡ FFT/STFT Analysis & Anomaly Detection (via Isolation Forest)

🧠 Signal Classification using pretrained ResNet-18

💬 GPT-4o-based Diagnosis with human-readable insights

📄 Auto-generated Markdown + Plotly Reports

🧩 Modular architecture using the AgentPro agent-tool design

## Quick Start

### Installation
Install agentpro repository using pip:

```bash
pip install git+https://github.com/traversaal-ai/AgentPro.git -q
```
<!--
### Configuration

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
TRAVERSAAL_ARES_API_KEY=your_traversaal_ares_api_key
```
Ares internet tool: Searches the internet for real-time information using the Traversaal Ares API. To get `TRAVERSAAL_ARES_API_KEY`. Follow these steps:

1. Go to the [Traversaal API platform](https://api.traversaal.ai/)
2. Log in or create an account
3. Click **"Create new secret key"**
4. Copy the generated key and paste in `.env` file :

### Running the Agent

From the command line:

```bash
pip install -r requirements.txt
streamlit run app.py
```

You’ll see a UI open in your browser with options to generate, analyze, and diagnose signals interactively.. -->


## 🧱 Modular Agent Pipeline
This app connects four agents in a ReAct-style loop:

### 🧪 1. SignalGeneratorAgent
Action: "generate_signal"

Types: "sine", "square", "chirp", "noise", "noisy_sine", "multi_sine"

Output: time and signal array in dictionary format

### 📊 2. SignalAnalyzerAgent
Action: "analyze_signal"

Performs: FFT, STFT, anomaly detection, classification (ResNet-18)

Output: frequency bins, STFT image, labels, anomalies

### 🩺 3. SignalDiagnosticAgent
Action: "diagnose_signal"

Uses GPT-4o to interpret analyzer results

Output: root cause + next step recommendation

### 📝 4. ReportAgent
Action: "generate_report"

Output: Markdown report with base64-encoded plots + Plotly dashboard JSON

## ✨ Example Usage

from agentpro import ReactAgent
from signal_agents import (
    SignalGeneratorAgent,
    SignalAnalyzerAgent,
    SignalDiagnosticAgent,
    ReportAgent,
)
from agentpro import create_model
import os

#### Create model using OpenAI GPT-4o
model = create_model("openai", "gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

#### Initialize agents
tools = [
    SignalGeneratorAgent(),
    SignalAnalyzerAgent(),
    SignalDiagnosticAgent(),
    ReportAgent()
]

#### Initialize ReAct Agent
agent = ReactAgent(model=model, tools=tools)

#### Run query
query = "Generate a 100Hz sine wave, analyze and diagnose it, and give me a report"
result = agent.run(query)

print(result.final_answer)
## 🧰 Project Structure
## Project Structure

```
AgentPro/
├── agentpro/
│   ├── __init__.py
│   ├── react_agent.py                  # Core AgentPro class implementing react-style agent framework
│   ├── agent.py                        # Action, Observation, ThoughtStep, AgentResponse classes
│   ├── model.py                        # Model classes 
│   ├── tools/                          # folder for all tool classes
│       ├── __init__.py
│       ├── base_tool.py
│       ├── duckduckgo_tool.py
│       ├── calculator_tool.py
│       ├── userinput_tool.py
│       ├── ares_tool.py
│       ├── traversaalpro_rag_tool.py
│       ├── slide_generation_tool.py
│       └── yfinance_tool.py
│       └── signal_analyzer.py
│       └── signal_generator.py
│       └── signal_diagnostic.py
│       └── report_agent.py
├── cookbook/
│   ├── Traversaal x Optimized AI Hackathon 2025
│   ├── quick_start.ipynb
│   └── custool_tool.ipynb      
├── main.py                             # Entrypoint to run the agent
├── requirements.txt                    # Dependencies
├── README.md                           # Project overview, usage instructions, and documentation
├── setup.py       
├── app.py                              # Streamlit app
├── pyproject.toml     
└── LICENSE.txt                         # Open-source license information (Apache License 2.0)
```

## Requirements
- Python 3.8+
- OpenAI API key
- Traversaal Ares API key for internet search (Optional)

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for more details.
