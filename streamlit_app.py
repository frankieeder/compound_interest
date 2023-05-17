import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go


def make_returns_array_np(return_per_period, num_periods) -> np.array:
    return (1 + return_per_period) ** np.arange(num_periods)


def make_contributions_array_np(contribution_per_period, num_periods) -> np.array:
    return np.array([contribution_per_period] * num_periods)


def make_contributions_schedule_ndarray_np(
    returns, contributions, num_periods_to_contribute=0
) -> np.ndarray:
    outer_product = np.outer(returns, contributions)
    shifted = outer_product / returns
    upper_only = np.tril(shifted)
    growth_per_contribution = upper_only
    if num_periods_to_contribute >= 0:
        growth_per_contribution = growth_per_contribution[:, :num_periods_to_contribute]
    return growth_per_contribution


def make_investment_schedule_dataframe(
    initial_investment,
    contribution_per_period,
    num_periods,
    num_periods_to_contribute,
    return_per_period,
):
    returns = make_returns_array_np(
        return_per_period=return_per_period, num_periods=num_periods
    )
    contributions = make_contributions_array_np(
        contribution_per_period=contribution_per_period, num_periods=num_periods
    )
    contributions_schedule = make_contributions_schedule_ndarray_np(
        returns=returns,
        contributions=contributions,
        num_periods_to_contribute=num_periods_to_contribute,
    )

    initial_investment_growth = returns * initial_investment

    cols = [f"Contribution {i}" for i in range(contributions_schedule.shape[1])]
    df = pd.DataFrame(contributions_schedule, columns=cols)
    df["Initial Investment"] = initial_investment_growth
    df = df[["Initial Investment"] + cols]
    df = df.reset_index()
    return df


def melt_investment_schedule_for_plotly(investment_schedule: pd.DataFrame):
    melted_df = investment_schedule.melt(id_vars="index")
    unique_vars = melted_df["variable"].unique()
    var_to_id = {v: i for i, v in enumerate(unique_vars)}
    melted_df["var_id"] = melted_df["variable"].replace(var_to_id)
    return melted_df


def plot_investment_schedule(investment_schedule: pd.DataFrame) -> go.Figure:
    melted_df = melt_investment_schedule_for_plotly(investment_schedule)
    fig = px.bar(melted_df, x="index", y="value", color="var_id", hover_name="variable")
    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.25,
    )
    fig.update(layout_coloraxis_showscale=False)
    return fig


if __name__ == "__main__":
    import plotly.express as px

    LABEL_INITIAL_INVESTMENT = "Initial Investment"
    LABEL_CONTRIBUTION = "Contribution per Period"
    LABEL_NUM_PERIODS = "Num Periods to Simulate"
    LABEL_NUM_CONTRIBUTIONS = "Num Periods to Contribute"
    LABEL_PERCENT_RETURN = "% Return per Period"

    DEFAULT_INITIAL_INVESTMENT = 5_000
    DEFAULT_CONTRIBUTION = 32_850
    RETIREMENT_AGE = 65
    COLLEGE_GRADUATION_AGE = 21
    DEFAULT_NUM_PERIODS = RETIREMENT_AGE - COLLEGE_GRADUATION_AGE
    DEFAULT_NUM_CONTRIBUTIONS = 5
    DEFAULT_PERCENT_RETURN = 0.10

    st.markdown("""
    # Compound Interest
    (by Contribution)
    """)

    left, right = st.columns([4, 1])

    with left:
        st.markdown(
            "This app shows the growth of compound interest over time, but"
            " distinguishing between the impact of each individual contribution as it"
            " grows."
        )
        st.markdown(
            "The default settings show the growth of 5 years of contributing the 2023"
            " IRS limit of all common retirement accounts"
            f" (${DEFAULT_CONTRIBUTION:,} incl. 401(k), IRA, & HSA) into a dependable"
            f" index fund (with estimated {DEFAULT_PERCENT_RETURN * 100:.0f}% yearly"
            f" return) upon graduating college (age {COLLEGE_GRADUATION_AGE}). **Note"
            f" that by age {RETIREMENT_AGE}, this leads to quite a healthy retirement,"
            " with only 5 years of investing!**"
        )
        st.markdown(f"""
        Hopefully this tool can offer a more intuitive understanding of:
        - How compound interest grows over time
        - How early contributions are the most meaningful, with continued\
         contributions meeting diminishing returns fairly quickly (change \
         `{LABEL_NUM_CONTRIBUTIONS}`)
        - How % return significantly affects outcome  (change `{LABEL_PERCENT_RETURN}`)
        """)

    with right:
        initial_investment = st.number_input(
            LABEL_INITIAL_INVESTMENT, value=DEFAULT_INITIAL_INVESTMENT
        )
        contribution_per_period = st.number_input(
            LABEL_CONTRIBUTION, value=DEFAULT_CONTRIBUTION
        )
        num_periods = st.number_input(LABEL_NUM_PERIODS, value=DEFAULT_NUM_PERIODS)
        num_periods_to_contribute = st.number_input(
            LABEL_NUM_CONTRIBUTIONS, value=DEFAULT_NUM_CONTRIBUTIONS
        )
        return_per_period = st.number_input(
            LABEL_PERCENT_RETURN, value=DEFAULT_PERCENT_RETURN
        )

    investment_schedule = make_investment_schedule_dataframe(
        initial_investment,
        contribution_per_period,
        num_periods,
        num_periods_to_contribute,
        return_per_period,
    )
    fig = plot_investment_schedule(investment_schedule)
    st.plotly_chart(fig, use_container_width=True)
