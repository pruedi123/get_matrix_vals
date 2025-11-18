from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = [12 * year for year in range(1, 41)]
SOURCE_XLSX = Path(__file__).with_name("global_mo_returns_factors.xlsx")
SOURCE_SHEET = "factors_mo"
OUTPUT_XLSX = SOURCE_XLSX.with_name(f"{SOURCE_XLSX.stem}_rolling_windows.xlsx")
SUMMARY_SHEET = "rolling_minima"
MATRIX_SHEET = "minima_matrix"
PATH_SHEET = "allocation_path"
DEFAULT_GOAL_AMOUNT = 50000
DEFAULT_BEGIN_YEAR = 0
DEFAULT_END_YEAR = 29
CPI_SOURCE_XLSX = Path(__file__).with_name("cpi_mo_factors.xlsx")
CPI_SHEET = 0
CPI_PERCENTILE_DEFAULT = 50
CPI_OUTPUT_SHEET = "cpi_inverse_percentile"


def load_factors(path: Path) -> pd.DataFrame:
    """Load the monthly factor data and ensure clean column names."""
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    df = pd.read_excel(path, sheet_name=SOURCE_SHEET)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the worksheet.")

    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_factor_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col != "Date"]


def load_cpi_factors(path: Path, sheet: str | int = CPI_SHEET) -> pd.DataFrame:
    """Load monthly CPI factors."""
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the CPI worksheet.")
    factor_cols = get_factor_cols(df)
    if len(factor_cols) != 1:
        raise ValueError(f"Expected exactly one CPI factor column, found {factor_cols}.")

    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def apply_expense(df: pd.DataFrame, annual_bps: float) -> pd.DataFrame:
    """Reduce monthly factors by the chosen expense in basis points."""
    if annual_bps <= 0:
        return df

    monthly_rate = (annual_bps / 10000) / 12
    factor_cols = get_factor_cols(df)
    adjusted = df.copy()
    adjusted.loc[:, factor_cols] *= (1 - monthly_rate)
    return adjusted


def compute_rolling_products(
    df: pd.DataFrame, factor_cols: list[str], window: int
) -> pd.DataFrame:
    """Return the rolling window products for every LBM column using log-sums."""
    log_df = df[["Date"] + factor_cols].copy()
    log_df.loc[:, factor_cols] = np.log(log_df[factor_cols])

    rolling_sum = (
        log_df.set_index("Date")[factor_cols]
        .rolling(window=window, min_periods=window)
        .sum()
        .dropna(how="all")
    )
    rolling_prod = np.exp(rolling_sum)
    return rolling_prod.reset_index()


def calculate_window_summaries(
    df: pd.DataFrame, windows: list[int]
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Compute rolling products and minima for each requested window size."""
    factor_cols = get_factor_cols(df)
    if not factor_cols:
        raise ValueError("No LBM columns were found in the worksheet.")

    window_results: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict[str, float]] = []

    for window in windows:
        rolling = compute_rolling_products(df, factor_cols, window)
        if rolling.empty:
            continue

        window_results[window] = rolling
        minima = rolling[factor_cols].min()
        summary_rows.append(
            {
                "WindowYears": window // 12,
                "WindowMonths": window,
                **minima.to_dict(),
            }
        )

    if not window_results:
        raise ValueError("No rolling windows could be calculated with the data provided.")

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("WindowMonths")
        .reset_index(drop=True)
    )
    ordered_cols = ["WindowYears", "WindowMonths"] + factor_cols
    summary_df = summary_df[ordered_cols]

    return window_results, summary_df


def get_equity_percent(allocation: str) -> float:
    """Extract the equity allocation percentage from a column name."""
    cleaned = allocation.replace(" ", "")
    if cleaned.endswith("F"):
        return 0.0

    match = re.search(r"(\d+)", cleaned)
    if not match:
        raise ValueError(f"Could not parse equity percent from '{allocation}'.")
    return float(match.group(1))


def run_pipeline(
    df: pd.DataFrame, windows: list[int]
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper to compute rolling windows and derived summary tables."""
    window_results, summary_df = calculate_window_summaries(df, windows)
    matrix_df = build_minima_matrix(summary_df)
    path_df = compute_allocation_path(matrix_df)
    return window_results, summary_df, matrix_df, path_df


def build_minima_matrix(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Reformat minima so allocations are rows and years are columns."""
    matrix = summary_df.drop(columns=["WindowMonths"]).set_index("WindowYears").T
    matrix = matrix.replace(0, np.nan)
    matrix = 1 / matrix
    matrix.index.name = "Allocation"
    matrix.columns = [f"Year {int(col)}" for col in matrix.columns]
    matrix.insert(0, "Year 0", 1.0)
    return matrix


def compute_allocation_path(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """Determine year-by-year allocations given the monotonic equity constraint."""
    equity_map = {allocation: get_equity_percent(allocation) for allocation in matrix_df.index}
    records: list[dict[str, object]] = []
    start_allocation = next((alloc for alloc, pct in equity_map.items() if pct == 0), None)
    prev_allocation: str | None = start_allocation
    prev_equity = equity_map.get(start_allocation, float("-inf"))

    for col in sorted(matrix_df.columns, key=_parse_year_number):
        year_number = _parse_year_number(col)
        col_series = matrix_df[col]
        non_na = col_series.dropna()
        chosen_allocation: str | None = None
        chosen_value = np.nan

        if year_number == 0 and start_allocation:
            chosen_allocation = start_allocation
            chosen_value = col_series.get(chosen_allocation, np.nan)
            prev_equity = equity_map.get(chosen_allocation, prev_equity)
        elif not non_na.empty:
            min_allocation = non_na.idxmin()
            min_equity = equity_map[min_allocation]

            if prev_allocation is None or min_equity >= prev_equity:
                chosen_allocation = min_allocation
                chosen_value = non_na[min_allocation]
                prev_equity = min_equity
            else:
                chosen_allocation = prev_allocation
                chosen_value = col_series.get(chosen_allocation, np.nan)
        elif prev_allocation is not None:
            chosen_allocation = prev_allocation
            chosen_value = col_series.get(chosen_allocation, np.nan)

        prev_allocation = chosen_allocation

        records.append(
            {
                "Year": year_number,
                "Allocation": chosen_allocation,
                "EquityPercent": equity_map.get(chosen_allocation, np.nan)
                if chosen_allocation
                else np.nan,
                "Value": chosen_value,
            }
        )

    df = pd.DataFrame(records)
    ordered_cols = ["Year", "Allocation", "EquityPercent", "Value"]
    df = df[ordered_cols]
    return df


def calculate_goal_projection(
    path_df: pd.DataFrame, goal_amount: float, start_year: int, end_year: int
) -> tuple[pd.DataFrame, float, float, float, float]:
    if start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year.")

    subset = path_df[
        (path_df["Year"] >= start_year) & (path_df["Year"] <= end_year)
    ].copy()
    total_factor = subset["Value"].sum()
    projected_amount = total_factor * goal_amount
    avg_equity = subset["EquityPercent"].mean()
    bond_allocation = 100 - avg_equity
    return subset, total_factor, projected_amount, avg_equity, bond_allocation


def _parse_year_number(label: str) -> int:
    match = re.search(r"(\d+)", label)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unable to parse year number from '{label}'.")


def calculate_cpi_inverse_percentiles(
    df: pd.DataFrame, windows: list[int], percentile: float = CPI_PERCENTILE_DEFAULT
) -> pd.DataFrame:
    """Percentile of compounded CPI (price level) and its inverse (real $1) for each window."""
    factor_cols = get_factor_cols(df)
    if len(factor_cols) != 1:
        raise ValueError(f"Expected exactly one CPI factor column, found {factor_cols}.")
    factor_col = factor_cols[0]

    log_series = np.log(df[factor_col])
    records: list[dict[str, float]] = []

    for window in windows:
        rolling_sum = (
            log_series.rolling(window=window, min_periods=window)
            .sum()
            .dropna()
        )
        if rolling_sum.empty:
            continue
        compounded = np.exp(rolling_sum)
        percentile_compounded = float(np.percentile(compounded, percentile))
        real_value = 1 / percentile_compounded if percentile_compounded != 0 else np.nan
        records.append(
            {
                "WindowYears": window // 12,
                "WindowMonths": window,
                "Percentile": percentile,
                "Real Ending Value": percentile_compounded,
                "Increase In Cost Factor": real_value,
            }
        )

    if not records:
        raise ValueError("No CPI percentile values could be calculated with the data provided.")

    df_out = (
        pd.DataFrame(records)
        .sort_values("WindowMonths")
        .reset_index(drop=True)
    )
    ordered_cols = [
        "WindowYears",
        "WindowMonths",
        "Percentile",
        "Real Ending Value",
        "Increase In Cost Factor",
    ]
    return df_out[ordered_cols]


def write_outputs(
    window_results: dict[int, pd.DataFrame],
    summary_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    path_df: pd.DataFrame,
    cpi_percentiles_df: pd.DataFrame | None = None,
) -> None:
    """Persist each rolling window plus the minima summary to a workbook."""
    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        for window in sorted(window_results):
            sheet_name = f"roll_{window}m"
            window_results[window].to_excel(writer, sheet_name=sheet_name, index=False)
        summary_df.to_excel(writer, sheet_name=SUMMARY_SHEET, index=False)
        matrix_df.reset_index().to_excel(writer, sheet_name=MATRIX_SHEET, index=False)
        path_df.to_excel(writer, sheet_name=PATH_SHEET, index=False)
        if cpi_percentiles_df is not None:
            cpi_percentiles_df.to_excel(writer, sheet_name=CPI_OUTPUT_SHEET, index=False)


def render_streamlit(windows: list[int]) -> None:
    import streamlit as st
    import altair as alt

    st.set_page_config(page_title="Rolling Factor Minima", layout="wide")
    st.title("Rolling Factor Minimum Matrix")
    st.caption(
        "Products of monthly factors by allocation, using rolling periods from 1 to 40 years."
    )

    @st.cache_data(show_spinner=False)
    def cached_load() -> pd.DataFrame:
        return load_factors(SOURCE_XLSX)

    @st.cache_data(show_spinner=False)
    def cached_pipeline(expense_bps: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = cached_load()
        adjusted_df = apply_expense(df, expense_bps)
        _, summary_df, matrix_df, path_df = run_pipeline(adjusted_df, windows)
        return summary_df, matrix_df, path_df

    @st.cache_data(show_spinner=False)
    def cached_load_cpi() -> pd.DataFrame:
        return load_cpi_factors(CPI_SOURCE_XLSX)

    @st.cache_data(show_spinner=False)
    def cached_cpi_percentiles(percentile: float, version: int = 2) -> pd.DataFrame:
        """Cached CPI percentiles; version bump forces refresh when schema changes."""
        cpi_df = cached_load_cpi()
        return calculate_cpi_inverse_percentiles(cpi_df, windows, percentile=percentile)

    st.sidebar.header("Controls")
    annual_bps = st.sidebar.slider(
        "Annual Expense (basis points)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )
    goal_amount = st.sidebar.number_input(
        "Goal Amount ($)",
        min_value=0.0,
        value=float(DEFAULT_GOAL_AMOUNT),
        step=0.1,
    )
    begin_year = st.sidebar.slider(
        "Begin Year",
        min_value=0,
        max_value=40,
        value=DEFAULT_BEGIN_YEAR,
        step=1,
    )
    end_year = st.sidebar.slider(
        "End Year",
        min_value=0,
        max_value=40,
        value=DEFAULT_END_YEAR,
        step=1,
    )
    cpi_percentile = st.sidebar.slider(
        "CPI Percentile (real $ value)",
        min_value=0,
        max_value=100,
        value=int(CPI_PERCENTILE_DEFAULT),
        step=5,
    )
    income_amount = st.sidebar.number_input(
        "Income Stream (annual $)",
        min_value=0.0,
        value=10000.0,
        step=100.0,
    )
    cola_percent = st.sidebar.slider(
        "Annual COLA (%)",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
    )

    summary_df, matrix_df, path_df = cached_pipeline(int(annual_bps))
    cpi_percentiles_df = cached_cpi_percentiles(float(cpi_percentile), version=3)
    # Backward-compat for old cache/columns
    if "Increase In Cost Factor" not in cpi_percentiles_df.columns:
        if "RealDollarValue" in cpi_percentiles_df.columns:
            cpi_percentiles_df = cpi_percentiles_df.rename(columns={"RealDollarValue": "Increase In Cost Factor"})
        elif "InverseFactor" in cpi_percentiles_df.columns:
            cpi_percentiles_df = cpi_percentiles_df.rename(columns={"InverseFactor": "Increase In Cost Factor"})
        elif "CompoundedCPI" in cpi_percentiles_df.columns:
            cpi_percentiles_df["Increase In Cost Factor"] = 1 / cpi_percentiles_df["CompoundedCPI"]
    if "Real Ending Value" not in cpi_percentiles_df.columns:
        if "PriceLevelFactor" in cpi_percentiles_df.columns:
            cpi_percentiles_df = cpi_percentiles_df.rename(columns={"PriceLevelFactor": "Real Ending Value"})
        elif "CompoundedCPI" in cpi_percentiles_df.columns:
            cpi_percentiles_df = cpi_percentiles_df.rename(columns={"CompoundedCPI": "Real Ending Value"})

    st.markdown(f"**Selected expense:** {annual_bps} bps (applied monthly).")

    st.subheader("Allocation × Years Minimum Matrix")
    st.dataframe(
        matrix_df,
        height=min(700, 40 + 28 * len(matrix_df)),
        use_container_width=True,
    )

    display_path = path_df.copy()
    display_path["Value x Goal"] = display_path["Value"] * goal_amount
    display_path["Equity $"] = display_path["Value x Goal"] * (
        display_path["EquityPercent"] / 100.0
    )
    display_path["Fixed Income $"] = display_path["Value x Goal"] - display_path["Equity $"]

    st.subheader("Allocation Path (Monotonic Equity)")
    display_path_formatted = display_path.copy()
    display_path_formatted["Value"] = display_path_formatted["Value"].map(lambda x: f"{x:.4f}")
    display_path_formatted["Value x Goal"] = display_path["Value x Goal"].round(0).map(
        lambda x: f"${x:,.0f}"
    )
    st.dataframe(
        display_path_formatted.set_index("Year").T,
        height=min(400, 40 + 20 * len(display_path_formatted.columns)),
        use_container_width=True,
    )
    st.line_chart(
        path_df.set_index("Year")["EquityPercent"],
        height=200,
    )

    if end_year < begin_year:
        st.error("End Year must be greater than or equal to Begin Year.")
    else:
        subset, total_factor, projected_amount, avg_equity, bond_alloc = calculate_goal_projection(
            path_df, goal_amount, int(begin_year), int(end_year)
        )
        subset_display = subset.copy()
        subset_display["Value x Goal"] = subset_display["Value"] * goal_amount
        subset_display["Equity $"] = subset_display["Value x Goal"] * (
            subset_display["EquityPercent"] / 100.0
        )
        subset_display["Fixed Income $"] = subset_display["Value x Goal"] - subset_display["Equity $"]
        total_equity_dollars = subset_display["Equity $"].sum()
        total_fixed_dollars = subset_display["Fixed Income $"].sum()
        total_goal_cost = subset_display["Value x Goal"].sum()
        equity_share = (total_equity_dollars / total_goal_cost * 100) if total_goal_cost else 0.0
        fixed_share = (total_fixed_dollars / total_goal_cost * 100) if total_goal_cost else 0.0
        st.subheader("Goal Projection")
        st.write(
            f"Using years {int(begin_year)}–{int(end_year)} "
            f"({len(subset)} periods) and goal ${goal_amount:,.2f}:"
        )
        st.write(f"- Sum of Value row: ${total_factor:,.2f}")
        st.write(f"- Projected result: ${projected_amount:,.0f}")
        st.write(f"- Total equity dollars: ${total_equity_dollars:,.0f} ({equity_share:.2f}%)")
        st.write(f"- Total fixed-income dollars: ${total_fixed_dollars:,.0f} ({fixed_share:.2f}%)")
        st.dataframe(
            subset_display.set_index("Year").T,
            height=min(400, 40 + 20 * len(subset_display.columns)),
            use_container_width=True,
        )

    st.subheader("Underlying Window Minima")
    st.dataframe(summary_df, height=min(700, 40 + 20 * len(summary_df)), use_container_width=True)

    st.subheader(
        f"CPI price level percentile (p{int(cpi_percentile)}) "
        "(Real Ending Value) and real value of $1 (Increase In Cost Factor)"
    )
    cpi_display = cpi_percentiles_df.copy()
    # Place Real Ending Value as the last column
    reordered_cols = [
        "WindowYears",
        "WindowMonths",
        "Percentile",
        "Increase In Cost Factor",
        "Real Ending Value",
    ]
    cpi_display = cpi_display[reordered_cols]
    st.dataframe(cpi_display, height=min(700, 40 + 20 * len(cpi_display)), use_container_width=True)

    chart_choice = st.radio(
        "CPI chart series",
        options=["Increase In Cost Factor", "Real Ending Value", "Both"],
        index=0,
        horizontal=True,
    )
    if chart_choice == "Both":
        chart_df = cpi_display.set_index("WindowYears")[["Increase In Cost Factor", "Real Ending Value"]]
    else:
        chart_df = cpi_display.set_index("WindowYears")[[chart_choice]]
    st.line_chart(chart_df, height=240)

    st.subheader("Indexed income stream (nominal + COLA, then deflated by CPI percentile)")
    cola_rate = cola_percent / 100.0
    income_rows: list[dict[str, float]] = []
    for _, row in cpi_display.iterrows():
        year = int(row["WindowYears"])
        nominal_income = income_amount * ((1 + cola_rate) ** year)
        real_income = nominal_income * row["Real Ending Value"]
        income_rows.append(
            {
                "Year": year,
                "Nominal Income": nominal_income,
                "Real Income (at CPI percentile)": real_income,
            }
        )
    income_df = pd.DataFrame(income_rows)
    income_display = income_df.copy()
    for col in ["Nominal Income", "Real Income (at CPI percentile)"]:
        income_display[col] = income_display[col].round(0).map(lambda x: f"${x:,.0f}")
    st.dataframe(
        income_display,
        height=min(700, 40 + 20 * len(income_display)),
        use_container_width=True,
    )
    # Diverging bridge: nominal bar with inflation drag overlay (negative delta).
    delta_df = income_df.copy()
    delta_df["Inflation Drag"] = delta_df["Real Income (at CPI percentile)"] - delta_df["Nominal Income"]
    chart_data = delta_df.melt(
        id_vars=["Year", "Real Income (at CPI percentile)"],
        value_vars=["Nominal Income", "Inflation Drag"],
        var_name="Series",
        value_name="Value",
    )
    bar_chart = (
        alt.Chart(chart_data)
        .mark_bar(size=12)
        .encode(
            x=alt.X(
                "Year:O",
                title="Year",
                scale=alt.Scale(paddingInner=0.1, paddingOuter=0.05),
            ),
            y=alt.Y("Value:Q", title="Income ($)"),
            color=alt.Color(
                "Series:N",
                title="Component",
                scale=alt.Scale(
                    domain=["Nominal Income", "Inflation Drag"],
                    range=["#cfcfcf", "#d62728"],
                ),
            ),
            tooltip=[
                "Year:O",
                "Series:N",
                alt.Tooltip("Value:Q", format=",.0f", title="Component $"),
                alt.Tooltip("Real Income (at CPI percentile):Q", format=",.0f", title="Real Income"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(bar_chart, use_container_width=True)


def running_in_streamlit() -> bool:
    if os.environ.get("STREAMLIT_SERVER_ENABLED") == "1":
        return True
    if os.environ.get("STREAMLIT_RUNTIME") == "1":
        return True

    if "streamlit" in sys.modules:
        try:
            from streamlit.runtime.scriptrunner import script_run_context as st_ctx

            if st_ctx.get_script_run_ctx() is not None:
                return True
        except Exception:
            return True

    return False


def main() -> None:
    if running_in_streamlit():
        render_streamlit(WINDOWS)
        return

    base_df = load_factors(SOURCE_XLSX)
    cpi_df = load_cpi_factors(CPI_SOURCE_XLSX)
    window_results, summary_df, matrix_df, path_df = run_pipeline(base_df, WINDOWS)
    cpi_percentiles_df = calculate_cpi_inverse_percentiles(
        cpi_df, WINDOWS, percentile=CPI_PERCENTILE_DEFAULT
    )
    write_outputs(window_results, summary_df, matrix_df, path_df, cpi_percentiles_df)

    preview_window = next(iter(sorted(window_results)))
    preview_df = window_results[preview_window].head()

    print(f"Calculated rolling products for {len(window_results)} window sizes.")
    print(f"Workbook saved to {OUTPUT_XLSX} with {len(window_results)} detail sheets.")
    print(f"Minimum factors by allocation written to '{SUMMARY_SHEET}'.")
    print(f"Allocation × years matrix written to '{MATRIX_SHEET}'.")
    print(f"Preview of {preview_window}-month results:")
    print(preview_df)
    print("Minima preview:")
    print(summary_df.head())
    print("Inverse minima matrix preview (1/min values):")
    print(matrix_df.head())
    print("Allocation path preview:")
    print(path_df.head().T)
    print(f"CPI inverse percentile preview (p{CPI_PERCENTILE_DEFAULT}):")
    print(cpi_percentiles_df.head())
    subset, total_factor, projected_amount, avg_equity, bond_alloc = calculate_goal_projection(
        path_df, DEFAULT_GOAL_AMOUNT, DEFAULT_BEGIN_YEAR, DEFAULT_END_YEAR
    )
    print(
        f"Goal projection (years {DEFAULT_BEGIN_YEAR}-{DEFAULT_END_YEAR}, "
        f"goal ${DEFAULT_GOAL_AMOUNT:,.2f}): sum ${total_factor:,.2f}, "
        f"projected ${projected_amount:,.0f}"
    )


if __name__ == "__main__":
    main()
