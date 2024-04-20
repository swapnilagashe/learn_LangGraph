from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_core.agents import AgentAction, AgentFinish

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from graph_state import AgentState

from util import *
load_dotenv()

def run_agent(data): 
    # returns AgentAction,AgentFinish or None
    agent_outcome = agent_runnable.invoke(data)
    return {"agent_outcome": agent_outcome}

# define function to execute tools
def execute_tools(data):
    # get the most recent agent outcome, key added in run_agent function
    agent_action = data['agent_outcome']
    output = tool_executor.invoke(agent_action)
    return {'intermediate_steps':[(agent_action,str(output))]}
    
# define the logic that will be used to determine which conditional edge to go down
def should_continue(data):
    # if agent outcome is AgentFinish, then we return 'exit' string
    # this will be used when setting up the graph to define the flow
    
    if isinstance(data['agent_outcome'],AgentFinish):
        return "end"
    # otherwise AgentAction is returned, that is 'continue' string
    else:
        return "continue"
    

# configure llm
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = initialize_llm(temperature=0.5, streaming=True)
# define tools to be used
tools = [TavilySearchResults(max_results=1)]

# create runnable
agent_runnable = create_openai_functions_agent(llm,tools,prompt)

tool_executor = ToolExecutor(tools)

# define the graph

# define new graph
workflow = StateGraph(AgentState)

# Define the StateGraph
# Add nodes to the graph
workflow.add_node("agent",run_agent)

#workflow.add_node("agent", run_agent())
workflow.add_node("action", execute_tools)

# set the starting point, this node will be called first
workflow.set_entry_point("agent")

# add a conditional edge
workflow.add_conditional_edges(
    "agent", # define start node, means that these are the edges taken after the 'agent' node is called  
    should_continue, # mapping- str:nodes, decides which node to be called next
    {"continue": "action",
     "end":END} # END is special node that returns to the user
)

# add a normal edge from 'tools' to 'agent'
# means after tools is called, agent is called next
workflow.add_edge('action','agent')

# compile the graph
app = workflow.compile()

# define inputs
inputs= {"input":"what is the weather in Pune", "chat_history":[]}

# print output from each node
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("--------------------------------")
    