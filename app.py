import streamlit as st
import openai
from agentpro import create_model, ReactAgent
from agentpro.tools import AresInternetTool, UserInputTool, CalculateTool , SignalGeneratorAgent, SignalAnalyzerAgent, SignalDiagnosticAgent, ReportAgent  # Adjust import path

# Set your API keys here or use Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
ares_key = st.secrets["ARES_KEY"]

st.set_page_config(page_title="Signal Analysis Assistant", layout="wide")
st.title("🔍 Signal Analysis AI Agent")
st.header("Signal Analysis Agent")

# Show example query for guidance only (disabled textarea)
sample_query = """
Please simulate a multi-tone sine wave with frequencies 100Hz and 300Hz (duration = 2 seconds, sampling_rate = 2000 Hz)
Once the signal is generated:
1. Compute its FFT and identify any prominent peaks.
2. Detect if there is any unexpected frequency component (for example, a DC bias or 60 Hz hum).
3. Hypothesize likely causes for any anomaly.
4. Finally, give me a markdown-formatted summary that includes:
  - Time-domain plot (as a base64 image or description)
  - Frequency-domain plot (as a base64 image or description)
  - A bullet-list of detected peaks
  - A plain-English diagnosis of what might have gone wrong in the simulated signal.
"""

with st.expander("Example query (read-only)"):
    st.text_area("Sample Query", value=sample_query, height=200, disabled=True)

input_mode = st.radio("Choose input mode:", ["Enter full query text", "Set parameters"])

if input_mode == "Enter full query text":
    user_query = st.text_area("Enter your full query here", height=250)
    analyze_btn = st.sidebar.button("Run Signal Analysis")
else:
    # Sidebar parameters
    st.sidebar.header("Signal Simulation Parameters")
    duration = st.sidebar.slider("Duration (s)", 1, 10, 2)
    sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 500, 10000, 2000)
    frequencies = st.sidebar.text_input("Frequencies (comma-separated)", "50,150,300")
    amplitudes = st.sidebar.text_input("Amplitudes (comma-separated)", "1,0.5,0.2")
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    analyze_btn = st.sidebar.button("Run Signal Analysis")

if analyze_btn:
    with st.spinner("Generating and analyzing signal..."):
        try:
            # Initialize model and tools
            model = create_model(provider="openai", model_name="gpt-4o", api_key=openai.api_key)
            tools = [
                AresInternetTool(ares_key), UserInputTool(), CalculateTool(),
                SignalGeneratorAgent(), SignalAnalyzerAgent(), SignalDiagnosticAgent(), ReportAgent()
            ]
            agent = ReactAgent(model=model, tools=tools)

            if input_mode == "Enter full query text":
                if not user_query.strip():
                    st.warning("Please enter a query in the text area.")
                    st.stop()
                query = user_query
            else:
                # Formulate query for agent
                query = (
                    f"Generate a multi-tone sine wave with the following parameters:\n"
                    f"- Duration: {duration} seconds\n"
                    f"- Sampling Rate: {sampling_rate} Hz\n"
                    f"- Frequencies: {frequencies} Hz\n"
                    f"- Amplitudes: {amplitudes}\n"
                    f"- Add Gaussian noise with level {noise_level}.\n"
                    "Then:\n"
                    "1. Perform FFT and detect peaks.\n"
                    "2. Identify anomalies (e.g., DC bias, 60Hz hum).\n"
                    "3. Provide diagnostic explanation.\n"
                    "4. Output markdown summary with plots, metrics, diagnosis, and recommendations."
                )

            response = agent.run(query)
            st.markdown(response.final_answer, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("👈 Set parameters or enter query and click 'Run Signal Analysis' to begin.")
