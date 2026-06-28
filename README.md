<p align="center">
  <img src="logo.svg" alt="AnalyticaOS logo" width="80" height="80"/>
</p>

# AnalyticaOS

**Autonomous AI Data Scientist & Executive Strategy Consultant**  
Built by **[Daipayan Chatterjee](https://github.com/dchatterjee01-prog)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://analyticaos-daipayan.streamlit.app/)

AnalyticaOS turns any dataset into boardroom-ready intelligence. Upload a CSV or Excel file and let the system clean it, explore it, test hypotheses, train machine learning models, build neural networks, detect anomalies, forecast trends, and synthesize everything into an executive briefing — plus a downloadable Word report — without writing a single line of code.

---

## 🚀 Live Demo

**Try it here:** [AnalyticaOS on Streamlit Community Cloud](https://analyticaos-daipayan.streamlit.app/)

> Requires Google login (OIDC via Streamlit Community Cloud). Local dev works without login — see [Running it locally](#-running-it-locally).

---

## ⚙️ What it does — 31 modules across 6 groups

### 📁 Data
| Module | Capability |
|---|---|
| 📁 **Upload Data** | CSV / Excel upload, preview, session state management |
| 🗂️ **Multi-Sheet Excel** | Sheet-level intelligence across multi-tab Excel workbooks |
| 🧹 **Data Cleaning** | Missing value analysis, duplicate detection, type fixing, auto-cleaning |

### 🔌 SQL Engine
| Module | Capability |
|---|---|
| 🔌 **SQL Connect** | Connect to PostgreSQL, MySQL, SQLite, and more |
| 🧠 **NL-to-SQL** | Natural language → SQL query generation via Gemini |
| 🪟 **SQL Builders** | Window functions, CTEs, and advanced query builders |
| 🔬 **SQL Explainer** | Plain-language breakdown of any SQL query |
| 🗺️ **Schema Map** | Visual entity-relationship diagram of connected databases |
| 🏗️ **Visual Query Builder** | Drag-and-drop SQL construction without typing |
| 🏛️ **Warehouse Advisor** | Indexing, partitioning, and optimization recommendations |
| 🤖 **Auto SQL Analytics** | Fully automated insight generation from SQL connections |

### 🔬 Analysis
| Module | Capability |
|---|---|
| 🔬 **EDA** | Statistical summaries, correlation analysis, outlier detection, time series |
| 📊 **Pivot Tables** | Pivot builder, Top N analysis, time intelligence, Pareto charts |
| 📈 **Visualizations** | Chart builder, dashboards, cross-tab and geo views |
| 📐 **Statistical Engine** | T-Test, Chi-Square, ANOVA, normality testing with plain-language interpretation |
| 🧪 **A/B Testing** | Experiment design, significance testing, power analysis |
| 🔗 **Causal Inference** | DoWhy treatment effect estimation and confounder handling |
| 🗄️ **SQL Agent** | Autonomous LangChain agent that writes and runs SQL against live connections |

### 🤖 Intelligence
| Module | Capability |
|---|---|
| 🤖 **Machine Learning** | Auto problem-type detection, training, evaluation, exportable models |
| 🧬 **AutoML Engine** | TabularPredictor automated leaderboards via AutoGluon |
| ⚡ **Deep Learning** | Custom PyTorch Feedforward Neural Networks (Classification & Regression) |
| 🧠 **Auto Questions** | Automatically surfaces the most statistically interesting questions in your data |
| 🤖 **Multi-Agent System** | Coordinated LangChain agents for parallel analytical reasoning |
| 🔮 **Forecasting** | Holt-Winters exponential smoothing and ARIMA/SARIMA models |
| 🚨 **Anomaly Detection** | Isolation Forest + PCA-based outlier detection |

### 📊 Advanced Analytics
| Module | Capability |
|---|---|
| 📊 **Dashboard Composer** | Build and export multi-chart analytical dashboards |
| 👥 **Cohort Analysis** | Retention curves, cohort heatmaps, LTV segmentation |
| 🔽 **Funnel Analysis** | Step-by-step conversion analysis with drop-off breakdown |

### 🏛️ Decision Intelligence
| Module | Capability |
|---|---|
| 💬 **Ask Your Data** | Autonomous LangChain + Gemini agent that writes and executes Pandas code locally |
| 🏛️ **Executive Console** | AI-generated strategic narratives, SWOT analysis, and KPI breakdowns |
| 📄 **Report Generator** | Automated generation of .docx files containing all charts, findings, and analysis |
| 🧮 **Optimization Engine** | Linear/integer programming for resource allocation and operational decisions |

---

## 🛠️ Tech Stack

| Layer | Libraries |
|---|---|
| **Frontend / Framework** | Streamlit |
| **Auth** | Streamlit OIDC (`st.login()` / `st.logout()`) via Google |
| **Data processing** | Pandas 3.0, NumPy, Polars |
| **Statistics & Math** | SciPy, statsmodels, networkx |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, CatBoost, AutoGluon |
| **Deep Learning** | PyTorch (CPU-optimized) |
| **Causal Inference** | DoWhy |
| **Visualization** | Plotly, Kaleido |
| **LLMs & Agents** | LangChain, Google Gemini 2.5 Flash (`google-genai`) |
| **Document generation** | `python-docx` |
| **Language** | Python 3.12 |

---

## 💻 Running it locally

**1. Clone the repository:**
```bash
git clone https://github.com/dchatterjee01-prog/analyticaos.git
cd analyticaos
```

**2. Create and activate the Conda environment:**
```bash
conda create -n analyticaos python=3.12 -y
conda activate analyticaos
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up secrets (optional — only needed for Google login and Gemini):**

Create `.streamlit/secrets.toml`:
```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "your-cookie-secret"
client_id = "your-google-client-id"
client_secret = "your-google-client-secret"

[gemini]
api_key = "your-gemini-api-key"
```

> If you skip the `[auth]` section, the app runs without login in local dev mode — the auth gate falls back gracefully.

**5. Run:**
```bash
streamlit run app.py
```

---

## 🔐 Authentication

AnalyticaOS uses Streamlit's native OIDC login (`st.login()` / `st.logout()`). On Streamlit Community Cloud, Google OAuth is configured via `secrets.toml`. Locally, the app detects the absence of an `[auth]` secrets section and skips the login gate entirely — no credential setup required for development.

---

## 📁 Project Structure

```
analyticaos/
├── app.py                  # Main entry point — auth gate, global CSS, NAV router, sidebar
├── config/
│   └── settings.py         # Color palette, app metadata (do not edit colors)
├── pages/                  # One module per file — all 31 page show() functions live here
│   ├── cleaning.py
│   ├── eda.py
│   ├── viz.py
│   ├── pivot.py
│   ├── stats.py
│   ├── experiments.py
│   ├── causal.py
│   ├── ml.py
│   ├── automl.py
│   ├── deep_learning.py
│   ├── forecast.py
│   ├── anomaly.py
│   ├── questions.py
│   ├── agents_ui.py
│   ├── sql_agent_ui.py
│   ├── executive.py
│   ├── report.py
│   ├── nlpqa.py
│   ├── optimization.py
│   ├── dashboard.py
│   ├── cohorts.py
│   ├── funnel.py
│   ├── excel_intelligence_ui.py
│   ├── upload.py
│   ├── sql_connect.py
│   ├── sql_nlquery.py
│   ├── sql_window.py
│   ├── sql_explainer.py
│   ├── sql_schema_map.py
│   ├── sql_visual_builder.py
│   ├── sql_warehouse.py
│   └── sql_auto_analytics.py
├── .streamlit/
│   └── secrets.toml        # Not committed — see setup instructions above
└── requirements.txt
```

---

## 🎨 Design System

AnalyticaOS uses a fixed "Scientific Blue" light theme defined entirely in `config/settings.py`:

| Token | Value | Usage |
|---|---|---|
| `PRIMARY_COLOR` | `#2A5C8C` | Brand, nav active state, headings |
| `BACKGROUND_COLOR` | `#FBFCFE` | App background |
| `SURFACE_COLOR` | `#EFF4FA` | Cards, sidebar |
| `TEXT_COLOR` | `#1B3A5C` | Body text |
| `BORDER_COLOR` | `#DCE4EC` | All borders |
| `MUTED_COLOR` | `#5C7290` | Secondary text, labels |

---

## 👤 Author

**Daipayan Chatterjee**  
GitHub: [@dchatterjee01-prog](https://github.com/dchatterjee01-prog)  
Live app: [analyticaos-daipayan.streamlit.app](https://analyticaos-daipayan.streamlit.app/)

---

*AnalyticaOS — turning raw data into boardroom-ready intelligence, autonomously.*