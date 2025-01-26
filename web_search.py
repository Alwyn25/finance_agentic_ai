from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")


web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,

)
web_search_agent.print_response("Whats happening in France for NVDA?", stream=True)

finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, 
                      analyst_recommendations=True, 
                      stock_fundamentals=True,
                      company_news=True,
                      company_info=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)
finance_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)
