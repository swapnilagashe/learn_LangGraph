from typing import TypedDict, Annotated, List,Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # the input message to the conversation
    input: str
    # the list of previous messages in the conversation
    chat_history: list[BaseMessage]
    
    # the outcome of a given call to the agent
    # needs None as valid type since that is how it will start
    agent_outcome: Union[AgentAction, AgentFinish, None]
    
    # list of actions are corresponding observations
    # here we annotate this with operator.add to indicate that
    # operations to this state should be added to the existing values and not override it
    # (by default the updates will override hence mentioned to add explicitly)
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]
    