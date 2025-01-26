import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define Agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True, company_info=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# APIAgent: Fetch live and historical stock data
class APIAgent:
    def fetch_live_data(self, ticker: str):
        stock = yf.Ticker(ticker)
        live_data = stock.history(period="1d")
        return live_data

    def fetch_historical_data(self, ticker: str, period: str = "1y"):
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period=period)
        return historical_data

# DataProcessingAgent: Process and summarize trends in stock data
class DataProcessingAgent:
    def summarize_trends(self, historical_data: pd.DataFrame):
        summary = {
            "mean_price": historical_data["Close"].mean(),
            "max_price": historical_data["Close"].max(),
            "min_price": historical_data["Close"].min(),
        }
        return summary

# UIDisplayAgent: Visualize and save plots
class UIDisplayAgent:
    def visualize_data(self, historical_data, stock_symbol):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name=f'{stock_symbol} Close'))
        fig.update_layout(title=f"Stock Prices for {stock_symbol} Over Time", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)  # Display the graph in Streamlit

    def save_plot(self, historical_data, stock_symbol, filename="static/plot.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(historical_data.index, historical_data['Close'], label=f"{stock_symbol} Close Price")
        plt.title(f"Stock Prices for {stock_symbol} Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(filename)  # Save the plot as a PNG file

# Utility function to extract and render tables
def extract_table_from_markdown(markdown_content):
    """
    Extracts markdown table from the agent's response and converts it into a DataFrame.
    """
    try:
        lines = [line.strip() for line in markdown_content.split("\n") if line.strip()]
        if "| " in lines[0] and "---" in lines[1]:  # Table detected
            header = lines[0].strip("| ").split(" | ")
            data = [line.strip("| ").split(" | ") for line in lines[2:]]
            return pd.DataFrame(data, columns=header)
    except Exception as e:
        st.warning(f"Error parsing table: {e}")
    return None

# Streamlit UI
st.title("AI Agent Playground")
st.sidebar.title("Agent Selector")

# Sidebar radio button to select model
agent_option = st.sidebar.radio(
    "Select the model(s) to run:",
    options=["Web Search Agent", "Finance AI Agent", "Both Agents"],
    index=2  # Default to "Both Agents"
)

# Select period for the stock graph
period = st.sidebar.selectbox(
    "Select period for the stock graph:",
    options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    index=2  # Default to 1 month
)

# Query input and submit button
query = st.text_area("Enter your query:", "", height=100)
submit = st.button("Run Agent")

# Handle query submission
if submit and query.strip():
    st.subheader("Agent Responses")

    # Define function to process agent responses
    def process_agent_response(agent, agent_name):
        with st.spinner(f"Running {agent_name}..."):
            try:
                # Execute the agent and get the response
                response = agent.run(query)
                
                # Check if the response is a function and execute it
                if callable(response):
                    result = response()  # Execute the function call
                    response_content = str(result)
                else:
                    response_content = response.content if hasattr(response, "content") else str(response)

                # Check for tables
                table = extract_table_from_markdown(response_content)
                if table is not None:
                    st.write(f"**{agent_name} Response (Table):**")
                    st.table(table)  # Display the table if found
                else:
                    st.write(f"**{agent_name} Response:**")
                    st.markdown(response_content, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred with {agent_name}: {e}")
    
    # Show responses based on selected models
    if agent_option == "Web Search Agent" or agent_option == "Both Agents":
        process_agent_response(web_search_agent, "Web Search Agent")
    
    if agent_option == "Finance AI Agent" or agent_option == "Both Agents":
        process_agent_response(finance_agent, "Finance AI Agent")

    # Extract stock symbols from query (check for "compare" and multiple symbols)
    if "compare" in query.lower():
        # Assume two stock symbols are mentioned in the query, e.g., "compare NVDA and AAPL"
        symbols = [word.upper() for word in query.split() if word.isupper()]
        
        if len(symbols) >= 2:  # If two symbols are mentioned
            st.subheader(f"Comparing Analyst Recommendations for {symbols[0]} and {symbols[1]}")

            # Instantiate the agents
            api_agent = APIAgent()
            data_processing_agent = DataProcessingAgent()
            ui_display_agent = UIDisplayAgent()

            # Fetch and process historical data for both stocks
            for symbol in symbols:
                # Fetch historical stock data for both symbols
                historical_data = api_agent.fetch_historical_data(symbol, period=period)

                # Process the data to get summary statistics
                summary = data_processing_agent.summarize_trends(historical_data)
                st.write(f"Summary of trends for {symbol}:", summary)

                # Visualize the data using Plotly
                ui_display_agent.visualize_data(historical_data, symbol)

                # Save the plot as an image file (optional)
                ui_display_agent.save_plot(historical_data, symbol, filename=f"static/{symbol}_plot.png")
                st.write(f"Plot saved as static/{symbol}_plot.png")  # If you're using Streamlit to display

        else:
            st.warning("Please provide exactly two stock symbols for comparison (e.g., NVDA and AAPL).")
    
    else:
        symbols = [word.upper() for word in query.split() if word.isupper()]

        # Instantiate the agents
        api_agent = APIAgent()
        data_processing_agent = DataProcessingAgent()
        ui_display_agent = UIDisplayAgent()

        # Fetch and process historical data for both stocks
        for symbol in symbols:
            st.subheader(f" Analyst Recommendations for {symbol}")
            # Fetch historical stock data for both symbols
            historical_data = api_agent.fetch_historical_data(symbol, period=period)

            # Process the data to get summary statistics
            summary = data_processing_agent.summarize_trends(historical_data)
            st.write(f"Summary of trends for {symbol}:")
            st.write(f"Mean Price:", summary['mean_price'])
            st.write(f"Max Price:", summary['max_price'])
            st.write(f"Min Price:", summary['min_price'])

            # Visualize the data using Plotly
            ui_display_agent.visualize_data(historical_data, symbol)

            # Save the plot as an image file (optional)
            ui_display_agent.save_plot(historical_data, symbol, filename=f"static/{symbol}_plot.png")
            st.write(f"Plot saved as static/{symbol}_plot.png")  # If you're using Streamlit to display

else:
    if submit:
        st.warning("Please enter a query.")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    1. Use the radio button to select one or both agents.
    2. Select the period for the stock graph in the sidebar.
    3. Enter your query in the text area.
    4. Click "Run Agent" to see the responses from the selected agents and the live stock graph.
    5. To compare two stocks, use a query like: "Compare NVDA and AAPL".
    """
)
