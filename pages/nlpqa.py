import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import textwrap
from google import genai
from config.settings import (
    PRIMARY_COLOR, SURFACE_COLOR, ACCENT_COLOR,
    TEXT_COLOR, BACKGROUND_COLOR
)

GEMINI_MODEL = "gemini-2.5-flash"

# ── CSS ──────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown(f"""
    <style>
    .section-header {{
        color: {PRIMARY_COLOR};
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid {ACCENT_COLOR};
        padding-left: 0.6rem;
    }}
    .chat-bubble-user {{
        background: {PRIMARY_COLOR}22;
        border: 1px solid {PRIMARY_COLOR}55;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: {TEXT_COLOR};
    }}
    .chat-bubble-answer {{
        background: {SURFACE_COLOR};
        border: 1px solid #2a2a45;
        border-left: 4px solid {ACCENT_COLOR};
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        color: {TEXT_COLOR};
    }}
    .code-caption {{
        font-size: 0.75rem;
        color: {TEXT_COLOR};
        opacity: 0.6;
        margin-top: 0.3rem;
    }}
    .warn-badge {{
        background: #FF6B6B22;
        border: 1px solid #FF6B6B;
        color: #FF6B6B;
        border-radius: 20px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── SCHEMA SUMMARY (sent to LLM instead of raw data) ──────────────────────────
def _build_schema_summary(df: pd.DataFrame) -> str:
    lines = [f"DataFrame `df` has {len(df)} rows and {len(df.columns)} columns.\n"]
    lines.append("Columns:")

    for col in df.columns:
        n_null = df[col].isnull().sum()

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            lines.append(
                f"  - '{col}' (numeric): "
                f"min={desc['min']:.2f}, max={desc['max']:.2f}, "
                f"mean={desc['mean']:.2f}, nulls={n_null}"
            )
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            lines.append(
                f"  - '{col}' (datetime): "
                f"range {df[col].min()} to {df[col].max()}, nulls={n_null}"
            )
        else:
            n_unique = df[col].nunique()
            sample_vals = df[col].dropna().unique()[:5].tolist()
            lines.append(
                f"  - '{col}' (text/categorical): "
                f"{n_unique} unique values, nulls={n_null}, "
                f"examples={sample_vals}"
            )

    return "\n".join(lines)


# ── PROMPT BUILDING ────────────────────────────────────────────────────────────
def _build_prompt(schema_summary: str, question: str) -> str:
    return textwrap.dedent(f"""
    You are a pandas code generator. You are given the schema of a pandas
    DataFrame called `df` (NOT the actual data — only column names, dtypes,
    and summary statistics).

    {schema_summary}

    User question: "{question}"

    Write a SINGLE line of pandas code that computes the answer to this
    question, assigning the final result to a variable named `result`.

    RULES:
    - Use ONLY the variable `df` (already in scope) and the `pd`/`np` modules.
    - Do NOT import anything. Do NOT define functions. Do NOT use exec/eval.
    - Do NOT read/write files, do NOT use os/sys/subprocess.
    - The code must be exactly one line, assigning to `result`.
    - `result` should be a scalar, a pandas Series, or a pandas DataFrame —
      whichever is most natural for the question.
    - If the question cannot be answered from the given schema, set
      result = "I cannot answer this from the available data."

    Respond with ONLY a JSON object in this exact format, no markdown fences,
    no explanation:
    {{"code": "<the one-line pandas expression>", "explanation": "<one sentence in plain English>"}}
    """).strip()


# ── SAFETY: VALIDATE GENERATED CODE BEFORE EXECUTION ──────────────────────────
FORBIDDEN_PATTERNS = [
    r"\bimport\b", r"\bexec\b", r"\beval\b", r"\bopen\b",
    r"\b__\w+__\b", r"\bos\.", r"\bsys\.", r"\bsubprocess\b",
    r"\bgetattr\b", r"\bsetattr\b", r"\bdelattr\b", r"\bglobals\b",
    r"\blocals\b", r"\bcompile\b", r"\binput\b", r"\bwrite\b",
    r"\bto_csv\b", r"\bto_excel\b", r"\bto_pickle\b", r"\bto_sql\b",
]


def _validate_code(code: str):
    """Returns (is_safe, reason_if_not)."""
    if "\n" in code.strip():
        return False, "Generated code must be a single line."
    if "result" not in code:
        return False, "Generated code does not assign to 'result'."
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return False, f"Generated code contains a forbidden pattern: {pattern}"
    return True, ""


def _execute_safely(code: str, df: pd.DataFrame):
    """Executes the validated one-liner in a restricted namespace.
    Only `df`, `pd`, and `np` are available — no builtins beyond
    what's needed for basic pandas expressions."""
    safe_builtins = {
        "len": len, "min": min, "max": max, "sum": sum,
        "round": round, "abs": abs, "sorted": sorted,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "str": str, "int": int, "float": float, "bool": bool,
        "range": range, "enumerate": enumerate, "zip": zip,
    }
    namespace = {
        "df": df, "pd": pd, "np": np,
        "__builtins__": safe_builtins,
    }
    exec(code, namespace)
    return namespace.get("result")


# ── GEMINI CALL ────────────────────────────────────────────────────────────────
def _ask_gemini(question: str, df: pd.DataFrame) -> dict:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in .streamlit/secrets.toml"}

    schema_summary = _build_schema_summary(df)
    prompt = _build_prompt(schema_summary, question)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        raw_text = response.text.strip()
        # strip markdown fences if the model added them anyway
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text).strip()
        parsed = json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        return {"error": "Gemini did not return valid JSON. Try rephrasing your question."}
    except Exception as e:
        return {"error": f"Gemini API error: {e}"}


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
def show():
    _inject_css()
    st.title("💬 Ask Your Data")
    st.caption("Phase 11.5 — Natural Language Q&A (Gemini 2.5 Flash)")

    df: pd.DataFrame = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a file first.")
        return

    if not st.secrets.get("GEMINI_API_KEY"):
        st.error(
            "❌ GEMINI_API_KEY not configured. "
            "Add it to `.streamlit/secrets.toml` to use this feature."
        )
        st.code('GEMINI_API_KEY = "your-key-here"', language="toml")
        return

    st.info(
        "ℹ️ This sends only your dataset's **schema and summary statistics** "
        "to Gemini — never your raw data. Gemini writes a pandas query, "
        "which runs locally on your machine."
    )

    if "nlpqa_history" not in st.session_state:
        st.session_state["nlpqa_history"] = []

    # ── Display chat history ──
    for entry in st.session_state["nlpqa_history"]:
        st.markdown(
            f'<div class="chat-bubble-user">🧑 {entry["question"]}</div>',
            unsafe_allow_html=True
        )
        if entry.get("error"):
            st.markdown(
                f'<div class="chat-bubble-answer">❌ {entry["error"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-answer">🤖 {entry["explanation"]}</div>',
                unsafe_allow_html=True
            )
            if entry.get("result_display") is not None:
                st.write(entry["result_display"])
            st.markdown(
                f'<div class="code-caption">Code run: <code>{entry["code"]}</code></div>',
                unsafe_allow_html=True
            )
        st.divider()

    # ── Input ──
    question = st.chat_input("Ask a question about your data...")

    if question:
        with st.spinner("Thinking..."):
            response = _ask_gemini(question, df)

        entry = {"question": question}

        if "error" in response:
            entry["error"] = response["error"]
        else:
            code = response.get("code", "")
            explanation = response.get("explanation", "")
            entry["code"] = code
            entry["explanation"] = explanation

            is_safe, reason = _validate_code(code)
            if not is_safe:
                entry["error"] = f"Generated code rejected for safety: {reason}"
            else:
                try:
                    result = _execute_safely(code, df)
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        entry["result_display"] = result
                    else:
                        entry["result_display"] = str(result)
                except Exception as e:
                    entry["error"] = f"Code execution failed: {e}"

        st.session_state["nlpqa_history"].append(entry)
        st.rerun()

    if st.session_state["nlpqa_history"]:
        if st.button("🗑️ Clear conversation"):
            st.session_state["nlpqa_history"] = []
            st.rerun()