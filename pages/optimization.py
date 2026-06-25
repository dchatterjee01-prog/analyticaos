"""
pages/optimization.py
AnalyticaOS - Optimization Engine (Stage D, Step Group 1)
Linear Programming via PuLP.
Tab 1: Budget Allocation Across Channels.
Tab 2: Resource Allocation Across Tasks.
"""

import streamlit as st
import pandas as pd
import pulp


def show():
    st.title("Optimization Engine")
    st.caption("Linear programming for resource and budget allocation decisions.")

    tab_budget, tab_resource = st.tabs(
        ["💰 Budget Allocation", "⚙️ Resource Allocation"]
    )

    with tab_budget:
        _show_budget_allocation()

    with tab_resource:
        _show_resource_allocation()


# ---------------------------------------------------------------------------
# TAB 1: Budget Allocation (Step 1)
# ---------------------------------------------------------------------------
def _show_budget_allocation():
    st.subheader("Use Case: Budget Allocation Across Channels")
    st.markdown(
        "Allocate a fixed total budget across marketing channels to "
        "**maximize total expected return**, subject to per-channel "
        "minimum and maximum spend limits."
    )

    st.markdown("### Step 1: Define Channels")

    if "opt_channels" not in st.session_state:
        st.session_state.opt_channels = pd.DataFrame(
            {
                "Channel": ["Search Ads", "Social Ads", "Email", "Affiliate"],
                "Return per $ (ROI)": [1.8, 1.4, 2.5, 1.1],
                "Min Spend ($)": [500, 0, 0, 0],
                "Max Spend ($)": [10000, 8000, 5000, 6000],
            }
        )

    edited_df = st.data_editor(
        st.session_state.opt_channels,
        num_rows="dynamic",
        width="stretch",
        key="channel_editor",
    )
    st.session_state.opt_channels = edited_df

    st.markdown("### Step 2: Set Total Budget")
    total_budget = st.number_input(
        "Total budget to allocate ($)",
        min_value=0.0,
        value=20000.0,
        step=500.0,
        key="budget_total_input",
    )

    st.markdown("### Step 3: Solve")
    solve_clicked = st.button(
        "Run Optimization", type="primary", key="budget_solve_btn"
    )

    if solve_clicked:
        df = st.session_state.opt_channels.dropna()

        if df.empty:
            st.error("Add at least one channel before solving.")
            return

        invalid_rows = df[df["Min Spend ($)"] > df["Max Spend ($)"]]
        if not invalid_rows.empty:
            st.error(
                "Min Spend exceeds Max Spend for: "
                + ", ".join(invalid_rows["Channel"].tolist())
            )
            return

        if df["Min Spend ($)"].sum() > total_budget:
            st.error(
                f"Sum of minimum spends (${df['Min Spend ($)'].sum():,.0f}) "
                f"exceeds total budget (${total_budget:,.0f}). Lower a "
                "minimum or raise the budget."
            )
            return

        prob = pulp.LpProblem("Budget_Allocation", pulp.LpMaximize)
        channel_vars = {}
        for _, row in df.iterrows():
            name = row["Channel"]
            channel_vars[name] = pulp.LpVariable(
                f"spend_{name}".replace(" ", "_"),
                lowBound=float(row["Min Spend ($)"]),
                upBound=float(row["Max Spend ($)"]),
            )

        prob += pulp.lpSum(
            channel_vars[row["Channel"]] * float(row["Return per $ (ROI)"])
            for _, row in df.iterrows()
        )
        prob += pulp.lpSum(channel_vars.values()) <= total_budget

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status_str = pulp.LpStatus[status]

        st.markdown("### Results")
        if status_str != "Optimal":
            st.warning(f"Solver status: {status_str}. No optimal solution found.")
        else:
            results = []
            total_spend = 0.0
            total_return = 0.0
            for _, row in df.iterrows():
                name = row["Channel"]
                spend = channel_vars[name].value()
                ret = spend * float(row["Return per $ (ROI)"])
                total_spend += spend
                total_return += ret
                results.append(
                    {
                        "Channel": name,
                        "Allocated Spend ($)": round(spend, 2),
                        "Expected Return ($)": round(ret, 2),
                    }
                )

            st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Budget", f"${total_budget:,.0f}")
            col2.metric("Total Allocated", f"${total_spend:,.0f}")
            col3.metric("Expected Total Return", f"${total_return:,.0f}")
            st.success(f"Solver status: {status_str}")


# ---------------------------------------------------------------------------
# TAB 2: Resource Allocation (Step 2 - NEW)
# ---------------------------------------------------------------------------
def _show_resource_allocation():
    st.subheader("Use Case: Resource Allocation Across Tasks")
    st.markdown(
        "Allocate a fixed pool of resource units (e.g. labor-hours, "
        "machine-hours) across tasks or projects to **maximize total "
        "output**, subject to per-task minimum and maximum usage limits."
    )

    st.markdown("### Step 1: Define Tasks")

    if "opt_tasks" not in st.session_state:
        st.session_state.opt_tasks = pd.DataFrame(
            {
                "Task": ["Project A", "Project B", "Project C", "Project D"],
                "Output per Unit": [12.0, 9.5, 15.0, 7.0],
                "Min Units": [10, 0, 5, 0],
                "Max Units": [80, 60, 40, 50],
            }
        )

    edited_df = st.data_editor(
        st.session_state.opt_tasks,
        num_rows="dynamic",
        width="stretch",
        key="task_editor",
    )
    st.session_state.opt_tasks = edited_df

    st.markdown("### Step 2: Set Total Available Resource Units")
    total_units = st.number_input(
        "Total resource units available (e.g. labor-hours)",
        min_value=0.0,
        value=150.0,
        step=10.0,
        key="resource_total_input",
    )

    st.markdown("### Step 3: Solve")
    solve_clicked = st.button(
        "Run Optimization", type="primary", key="resource_solve_btn"
    )

    if solve_clicked:
        df = st.session_state.opt_tasks.dropna()

        if df.empty:
            st.error("Add at least one task before solving.")
            return

        invalid_rows = df[df["Min Units"] > df["Max Units"]]
        if not invalid_rows.empty:
            st.error(
                "Min Units exceeds Max Units for: "
                + ", ".join(invalid_rows["Task"].tolist())
            )
            return

        if df["Min Units"].sum() > total_units:
            st.error(
                f"Sum of minimum units ({df['Min Units'].sum():,.0f}) "
                f"exceeds total available units ({total_units:,.0f}). "
                "Lower a minimum or raise the total."
            )
            return

        prob = pulp.LpProblem("Resource_Allocation", pulp.LpMaximize)
        task_vars = {}
        for _, row in df.iterrows():
            name = row["Task"]
            task_vars[name] = pulp.LpVariable(
                f"units_{name}".replace(" ", "_"),
                lowBound=float(row["Min Units"]),
                upBound=float(row["Max Units"]),
            )

        prob += pulp.lpSum(
            task_vars[row["Task"]] * float(row["Output per Unit"])
            for _, row in df.iterrows()
        )
        prob += pulp.lpSum(task_vars.values()) <= total_units

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status_str = pulp.LpStatus[status]

        st.markdown("### Results")
        if status_str != "Optimal":
            st.warning(f"Solver status: {status_str}. No optimal solution found.")
        else:
            results = []
            total_used = 0.0
            total_output = 0.0
            for _, row in df.iterrows():
                name = row["Task"]
                units = task_vars[name].value()
                out = units * float(row["Output per Unit"])
                total_used += units
                total_output += out
                results.append(
                    {
                        "Task": name,
                        "Allocated Units": round(units, 2),
                        "Expected Output": round(out, 2),
                    }
                )

            st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Units Available", f"{total_units:,.0f}")
            col2.metric("Total Units Allocated", f"{total_used:,.0f}")
            col3.metric("Expected Total Output", f"{total_output:,.1f}")
            st.success(f"Solver status: {status_str}")