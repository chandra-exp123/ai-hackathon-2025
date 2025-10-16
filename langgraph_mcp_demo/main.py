# main.py
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

async def main():
    # Use Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Point to your local math MCP server
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["server.py"],  
                "transport": "stdio",
            }
        }
    )

    # Fetch tools exposed by the MCP server
    tools = await client.get_tools()

    # Create the agent using model + MCP tools
    agent = create_react_agent(
        model=model,
        tools=tools,
    )

    # Ask a math question using the MCP tool
    math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

    for m in math_response["messages"]:
        m.pretty_print()

    #print("🤖 Agent response:", math_response["messages"][-1].content)

# Run async entry point
if __name__ == "__main__":
    asyncio.run(main())
