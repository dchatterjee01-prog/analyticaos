# AnalyticaOS

**Autonomous AI Data Scientist & Executive Strategy Consultant**

Built by **Daipayan Chatterjee**

AnalyticaOS turns any dataset into boardroom-ready intelligence. Upload a CSV or Excel file and let the system clean it, explore it, test hypotheses, train machine learning models, detect anomalies, forecast trends, and synthesize everything into an executive briefing — plus a downloadable Word report — without writing a line of code.

---

## What it does

| Module | Capability |
|---|---|
| 🧹 Data Cleaning | Missing value analysis, duplicate detection, type fixing, auto-cleaning |
| 🔬 EDA | Statistical summaries, correlation analysis, outlier detection, time series |
| 📊 Pivot Tables | Pivot builder, Top N analysis, time intelligence, Pareto charts |
| 📈 Visualizations | Chart builder, dashboards, cross-tab and geo views |
| 📐 Statistical Engine | T-Test, Chi-Square, ANOVA, normality testing with plain-language interpretation |
| 🤖 Machine Learning | Auto problem-type detection, training, evaluation, exportable models |
| 🧠 Auto Questions | Autonomous question generation and analysis roadmap from a dataset profile |
| 🤖 Multi-Agent System | An orchestrated pipeline of AI agents (data quality, insight, modeling) that audits a dataset and produces findings + recommendations |
| 🔮 Forecasting | Holt-Winters exponential smoothing time series forecasts |
| 🚨 Anomaly Detection | Isolation Forest + PCA-based outlier detection |
| 💬 Ask Your Data | Natural-language Q&A over your dataset, powered by Gemini 2.5 Flash (schema-only context, sandboxed execution — your raw data is never sent to the LLM) |
| 🏛️ Executive Console | A synthesized strategic briefing with a Corporate Health Index and prioritized action matrix |
| 📄 Report Generator | One-click Word (.docx) report combining every section above |

## Tech stack

- **Frontend/App framework:** Streamlit
- **Data processing:** pandas, NumPy
- **Statistics:** SciPy, statsmodels
- **Machine learning:** scikit-learn
- **Visualization:** Plotly
- **Document generation:** python-docx
- **LLM integration:** Google Gemini 2.5 Flash (`google-genai`)
- **Language:** Python 3.12

## Running it locally

```bash
git clone https://github.com/dchatterjee01-prog/analyticaos.git
cd analyticaos
conda create -n analyticaos python=3.12
conda activate analyticaos
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (this file is gitignored — never commit it) with:
```toml
GEMINI_API_KEY = "your-key-here"
```

Then run:
```bash
streamlit run app.py
```

## Deployment

This app is designed to run on **Streamlit Community Cloud**. Secrets (API keys, auth credentials) are configured via the Community Cloud dashboard's secrets manager, not committed to the repository.

## Architecture notes

- Every analysis phase persists its results in `st.session_state`, so later phases (e.g. the Executive Console, Report Generator) can reuse earlier findings without recomputation.
- The Multi-Agent System (`agents/` package) is decoupled from the Streamlit UI layer — agents are pure Python with no Streamlit imports, so they're independently testable.
- The `pages/` directory follows one-module-per-feature, each exposing a `show()` entry point called by the central router in `app.py`.

## License

See [LICENSE](LICENSE).

## Author

**Daipayan Chatterjee**
GitHub: [@dchatterjee01-prog](https://github.com/dchatterjee01-prog)
