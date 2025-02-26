#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from rapidfuzz import process

# Fields of interest for the Sankey graph
fields_of_interest = {
    "Total Revenue": 0,
    "Cost of Goods Sold": 0,
    "Gross Profit": 0,
    "Selling General and Administrative": 0,
    "Depreciation & Amortization": 0,
    "Research and Development": 0,
    "Other Operating Expenses": 0,
    "Net Interest Expense": 0,
    "Income Tax": 0,
    "Net Income": 0
}

# Function to perform fuzzy lookup and match fields, now more lenient
def fuzzy_lookup(field, available_fields, threshold=60):  # Lower threshold to 60 for more leniency
    match, score = process.extractOne(field, available_fields, score_cutoff=threshold)
    if match:
        return match
    return None

# Function to get matched fields and track unmatched ones
def get_value_safely(df, key, matched, unmatched):
    try:
        # Get available fields from the dataframe
        available_fields = df.index.tolist()
        # Use fuzzy lookup to get the closest match
        matched_field = fuzzy_lookup(key, available_fields)
        if matched_field:
            matched.append((key, matched_field))  # Track matched field
            return df.loc[matched_field].iloc[0]
        unmatched.append(key)  # Track unmatched field
        return 0
    except Exception:
        unmatched.append(key)
        return 0

def format_value(value, unit):
    if unit == 'Billions':
        return f"${value/1e9:.1f}B"
    elif unit == 'Millions':
        return f"${value/1e6:.1f}M"
    else:  # Thousands
        return f"${value/1e3:.1f}K"

def plot_sankey(ticker, report_type, year, quarter, unit):
    try:
        stock = yf.Ticker(ticker)
        
        if report_type == 'Annual':
            financials = stock.financials
            financials = financials.loc[:, financials.columns.year == year]
        else:  # Quarterly
            financials = stock.quarterly_financials
            financials = financials.loc[:, (financials.columns.year == year) & (financials.columns.quarter == quarter)]
        
        if financials.empty:
            st.error(f"No financial data available for {ticker} in the selected period")
            return

        # Track matched and unmatched fields
        matched = []
        unmatched = []

        # Retrieve values safely using the fuzzy lookup
        total_revenue = get_value_safely(financials, 'Total Revenue', matched, unmatched)
        cost_of_revenue = get_value_safely(financials, 'Cost of Goods Sold', matched, unmatched)
        gross_profit = total_revenue - cost_of_revenue
        operating_expense = get_value_safely(financials, 'Total Operating Expenses', matched, unmatched)
        net_income = get_value_safely(financials, 'Net Income', matched, unmatched)
        rnd = get_value_safely(financials, 'Research and Development', matched, unmatched)
        sga = get_value_safely(financials, 'Selling General and Administrative', matched, unmatched)
        
        # Ensure no value exceeds total revenue
        cost_of_revenue = min(cost_of_revenue, total_revenue)
        gross_profit = max(0, total_revenue - cost_of_revenue)
        operating_expense = min(operating_expense, gross_profit)
        operating_income = max(0, gross_profit - operating_expense)
        net_income = min(net_income, operating_income)
        
        # Calculate other values
        other_expenses = operating_expense - min(rnd, operating_expense) - min(sga, operating_expense - rnd)
        other_expenses = max(0, other_expenses)
        non_operating = net_income - operating_income

        # Prepare labels and values
        labels = [
            f"Revenue\n{format_value(total_revenue, unit)}",
            f"Cost of Goods Sold\n{format_value(cost_of_revenue, unit)}",
            f"Gross Profit\n{format_value(gross_profit, unit)}",
            f"Operating Expenses\n{format_value(operating_expense, unit)}",
            f"Operating {'Loss' if operating_income < 0 else 'Income'}\n{format_value(abs(operating_income), unit)}",
            f"Net {'Loss' if net_income < 0 else 'Income'}\n{format_value(abs(net_income), unit)}",
            f"R&D\n{format_value(rnd, unit)}",
            f"SG&A\n{format_value(sga, unit)}",
            f"Other Expenses\n{format_value(other_expenses, unit)}",
            f"Non-Operating\n{format_value(abs(non_operating), unit)}"
        ]
        
        # Handle profitable vs unprofitable cases
        if operating_income >= 0:
            source = [0, 0, 2, 2, 4, 3, 3, 3, 4]
            target = [2, 1, 4, 3, 5, 6, 7, 8, 9]
            values = [
                gross_profit, cost_of_revenue, operating_income, operating_expense, 
                net_income, rnd, sga, other_expenses, non_operating
            ]
        else:
            source = [0, 0, 2, 3, 3, 3, 4, 4]
            target = [2, 1, 3, 6, 7, 8, 5, 9]
            values = [
                gross_profit, cost_of_revenue, operating_expense, 
                rnd, sga, other_expenses, abs(net_income), abs(non_operating)
            ]
            # Add operating loss as a separate flow
            source.append(3)
            target.append(4)
            values.append(abs(operating_income))

        # Ensure all values are positive for Sankey diagram
        values = [abs(v) for v in values]
        
        # Create the Sankey Diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
              pad=15,
              thickness=20,
              line=dict(color="black", width=0.5),
              label=labels,
              color=["#87CEEB", "#FFB6C1", "#98FB98", "#FFB6C1", "#32CD32", "#006400", 
                     "#FF69B4", "#FF69B4", "#FF69B4", "#FF0000"]
            ),
            link=dict(
              source=source,
              target=target,
              value=values
          ))])

        # Update layout for better readability
        title = f"Financial Breakdown for {ticker} ({report_type} {year}"
        title += f" Q{quarter}" if report_type == 'Quarterly' else ")"
        fig.update_layout(
            title_text=title,
            font=dict(size=14, color="black"),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Display the diagram
        st.plotly_chart(fig, use_container_width=True)

        # Create a DataFrame for matched and unmatched fields
        matched_df = pd.DataFrame(matched, columns=['Expected Field', 'Matched Field'])
        unmatched_df = pd.DataFrame(unmatched, columns=['Unmatched Field'])
        
        # Display matched and unmatched fields
        st.write("Matched Fields:")
        st.dataframe(matched_df)
        
        st.write("Unmatched Fields:")
        st.dataframe(unmatched_df)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Streamlit app section
st.title('Financial Breakdown Sankey Diagram')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL):', 'AAPL')

# Select report type
report_type = st.selectbox('Select Report Type', ['Annual', 'Quarterly'])

# Get current year and quarter
current_year = datetime.now().year
current_quarter = (datetime.now().month - 1) // 3 + 1

# Select year
year = st.selectbox('Select Year', range(current_year, current_year-5, -1))

quarter = None
if report_type == 'Quarterly':
    # Select quarter for quarterly reports
    quarter = st.selectbox('Select Quarter', [1, 2, 3, 4], index=current_quarter-1)

# Select unit for financial figures
unit = st.selectbox('Select Unit', ['Billions', 'Millions', 'Thousands'])

if st.button('Generate Sankey Diagram'):
    st.write("Button clicked! Generating Sankey Diagram...")
    try:
        plot_sankey(ticker, report_type, year, quarter, unit)
        st.success("Sankey diagram generated successfully!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

st.write("App is running. If you can see this, Streamlit is working correctly.")

