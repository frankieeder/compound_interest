import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go


def make_returns_array_np(return_per_period, num_periods) -> np.array:
    return (1 + return_per_period) ** np.arange(num_periods)


def make_contributions_array_np(contribution_per_period, num_periods) -> np.array:
    return np.array([contribution_per_period] * num_periods)


def make_contributions_schedule_ndarray_np(returns, contributions, num_periods_to_contribute=0) -> np.ndarray:
    outer_product = np.outer(returns, contributions)
    shifted = outer_product / returns
    upper_only = np.tril(shifted)
    growth_per_contribution = upper_only
    if num_periods_to_contribute > 0:
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
        return_per_period=return_per_period,
        num_periods=num_periods
    )
    contributions = make_contributions_array_np(
        contribution_per_period=contribution_per_period,
        num_periods=num_periods
    )
    contributions_schedule = make_contributions_schedule_ndarray_np(
        returns=returns,
        contributions=contributions,
        num_periods_to_contribute=num_periods_to_contribute
    )

    initial_investment_growth = returns * initial_investment

    cols = [f"Contribution {i}" for i in range(contributions_schedule.shape[1])]
    df = pd.DataFrame(contributions_schedule, columns=cols)
    df['Initial Investment'] = initial_investment_growth
    df = df[['Initial Investment'] + cols]
    df = df.reset_index()
    return df


def melt_investment_schedule_for_plotly(investment_schedule: pd.DataFrame):
    melted_df = investment_schedule.melt(id_vars='index')
    unique_vars = melted_df['variable'].unique()
    var_to_id = {v: i for i, v in enumerate(unique_vars)}
    melted_df['var_id'] = melted_df['variable'].replace(var_to_id)
    return melted_df


def plot_investment_schedule(investment_schedule: pd.DataFrame) -> go.Figure:
    melted_df = melt_investment_schedule_for_plotly(investment_schedule)
    fig = px.bar(melted_df, x='index', y='value', color='var_id', hover_name='variable')
    fig.update_traces(
        marker_line_color='white',
        marker_line_width=0.25,
    )
    return fig


if __name__ == '__main__':
    import plotly.express as px

    initial_investment = st.number_input('Initial Investment', value=10_000)
    contribution_per_period = st.number_input('Contribution per Period', value=32_850)
    num_periods = st.number_input('Num Periods to Simulate', value=47)
    num_periods_to_contribute = st.number_input('Num Periods to Contribute', value=5)
    return_per_period = st.number_input('% Return per Period', value=0.10)

    investment_schedule = make_investment_schedule_dataframe(
            initial_investment,
            contribution_per_period,
            num_periods,
            num_periods_to_contribute,
            return_per_period,
    )
    fig = plot_investment_schedule(investment_schedule)
    st.plotly_chart(fig)

