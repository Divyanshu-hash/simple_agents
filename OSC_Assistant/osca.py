import asyncio
from langgraph_cua import create_cua
from dotenv import load_dotenv

load_dotenv()

# Create the CUA graph
cua_graph = create_cua(
    recursion_limit=120,
    timeout_hours=1,
)

messages = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant with access to a real browser. "
            "You can click, scroll, read GitHub pages, and analyze content."
        ),
    },
    {
        "role": "user",
        "content": (
            "I want to contribute to the LangGraph.js project. "
            "Please:\n"
            "1. Open the GitHub repository\n"
            "2. Read the README\n"
            "3. Check Issues and Pull Requests\n"
            "4. Find beginner-friendly issues\n"
            "5. Create a step-by-step contribution plan"
        ),
    },
]

async def main():
    stream = cua_graph.astream(
        {"messages": messages},
        stream_mode="updates"
    )

    async for update in stream:
        # VM creation info
        if "create_vm_instance" in update:
            stream_url = update["create_vm_instance"]["stream_url"]
            print("🔴 Watch agent live at:", stream_url)

        # Final response
        if "final" in update:
            print("\n✅ CONTRIBUTION PLAN\n")
            print(update["final"]["messages"][-1]["content"])

if __name__ == "__main__":
    asyncio.run(main())
