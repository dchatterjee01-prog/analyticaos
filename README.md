<p align="center">
  <img src="logo.svg" alt="AnalyticaOS logo" width="80" height="80"/>
</p>

# AnalyticaOS

**Autonomous AI Data Scientist & Executive Strategy Consultant**  
Built by **[Daipayan Chatterjee](https://github.com/dchatterjee01-prog)**

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/dchatterjee01/analyticaos-demo)

AnalyticaOS turns any dataset into boardroom-ready intelligence. Upload a CSV or Excel file and let the system clean it, explore it, test hypotheses, train machine learning models, build neural networks, detect anomalies, forecast trends, and synthesize everything into an executive briefing — plus a downloadable Word report — without writing a single line of code.

---

## 🚀 Live Demo

**Try it here:** [AnalyticaOS on Hugging Face Spaces](https://huggingface.co/spaces/dchatterjee01/analyticaos-demo)

> No login required. Local dev also works without login — see [Running it locally](#-running-it-locally).

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
| **Data processing** | Pandas 3.0, NumPy, Polars |
| **Statistics & Math** | SciPy, statsmodels, networkx |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, CatBoost, AutoGluon |
| **Deep Learning** | PyTorch (CPU-optimized) |
| **Causal Inference** | DoWhy |
| **Visualization** | Plotly, Kaleido |
| **LLMs & Agents** | LangChain, Google Gemini 2.0 Flash (`google-genai`) |
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

**4. Set up secrets (optional — only needed for Gemini-powered pages):**

Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

> If you skip this, the app runs fully — only Gemini-powered pages (Report Generator, NLP QA, Executive Console) will show an error when invoked.

**5. Run:**
```bash
streamlit run app.py
```

---

## 📁 Project Structure