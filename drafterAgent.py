import asyncio

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os
from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import chainlit as cl


load_dotenv()

# Write a cold pitch email proposing integrating AI agents into a business workflow.

@tool
def update(text:str)->str:
    """Updates the global doc text"""

    global document_text

    document_text = text

    return f"Document has been updated successfully! The current content is:\n{document_text}"

@tool
def drafter(filename:str)->str:
    """Writes the new content to the file"""

    global document_text

    if not filename.endswith('.txt'):
        filename += ".txt"

    with open(filename,'w',  encoding='utf-8' ) as w:
        w.write(document_text)
    return f"File updated successfully to {filename}"


tools = [update, drafter]

llm = ChatGroq(model="openai/gpt-oss-20b").bind_tools(tools)
print("Model initialized!")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

document_text = ""


def Agent(state: AgentState)-> AgentState:
    """The main agent node"""
    global document_text

    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents. Be joyful and have an enthusiastic tone. Use emojis.  

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to  show the current document state after modifications or when necessary.
    

    The current document content is:{document_text}
    """)

    # if not state['messages']:
    #     user_input = input("What do you want help with? ")
    #     user_prompt = HumanMessage(content=user_input)
    # else:
    #     user_input = input("What do you want to change in this document ? ")
    #     user_prompt = HumanMessage(content=user_input)

    # messages = [system_prompt] + list(state['messages'])+  [user_prompt]
    messages = [system_prompt] + state['messages']


    response = llm.invoke(messages)



    # return {'messages': list(state['messages']) + [user_prompt] + [response]}
    return {'messages': [response]}



# def to_proceed(state:AgentState):
#
#     for message in reversed(state['messages']):
#         # ... and checks if this is a ToolMessage resulting from save
#         if (isinstance(message, ToolMessage) and
#                 "saved" in message.content.lower() and
#                 "document" in message.content.lower()):
#             return "end"  # goes to the end edge which leads to the endpoint
#         else:
#             return "continue"
#
#     return "continue"

def to_proceed(state:AgentState):
    if not state['messages'][-1].tool_calls:
        return 'end'
    else:
        return 'continue'


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent_node",Agent)
graph.set_entry_point("agent_node")

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

# graph.add_edge("agent_node", "tool_node")

graph.add_conditional_edges(
    "agent_node",
    to_proceed,
    {
        "end": END,
        "continue": "tool_node",
    }
)

graph.add_edge("tool_node", "agent_node")

app = graph.compile()

@cl.on_chat_start
async def start():
    cl.user_session.set("graph", app)
    cl.user_session.set("document_text", "")
    await cl.Message(content="What do you want help with...").send()

@cl.on_message
async def main(message: cl.Message):
    graph = cl.user_session.get("graph")


    msg = cl.Message(content="")
    await msg.send()

    # Stream the full graph execution

    full_response = ""

    async for event in graph.astream(
            {"messages": [HumanMessage(content=message.content)]},
            config={"configurable": {"thread_id": "default"}},
            version="v2"
    ):

        print("Event received:", event)
        # kind = event["event"]
        if "agent_node" in event:
            messages = event["agent_node"].get("messages", [])
            for m in messages:
                if isinstance(m, AIMessage):
                    full_response += m.content

                    for char in m.content:
                        await msg.stream_token(char)
                        await asyncio.sleep(0.01)


            # Case 2: tool events
        kind = getattr(event, "event", None)
        if kind == "on_tool_start":
            await msg.stream_token(f"\n\n🛠️ Using tool: {getattr(event, 'name', '')}...")
        elif kind == "on_tool_end":
            await msg.stream_token(f"\n✅ Tool finished: {event.data['output'][:200]}...")

        await msg.update()


    # msg.content = full_response
    #
    # # Optional: Add a final "Done!" message
    # await msg.update()




# def run_document_agent():
#     print("\n ===== DRAFTER =====")
#
#     state = {"messages": []}
#
#     for step in app.stream(state, stream_mode="values"):
#         if "messages" in step:
#             print_messages(step["messages"])
#
#     print("\n ===== DRAFTER FINISHED =====")
#
# if __name__ == '__main__':
#     run_document_agent()