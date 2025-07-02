from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import litellm
from google.adk.tools.tool_context import ToolContext
from .sub_agents.explainer_agent.agent import explainer_agent

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types # For types.Content
from typing import Optional
from google.adk.models import LlmResponse, LlmRequest

# litellm._turn_on_debug()


def add_reminder(reminder: str, tool_context: ToolContext) -> dict:
    """Add a new reminder to the user's reminder list.

    Args:
        reminder: The reminder text to add
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(f"--- Tool: add_reminder called for '{reminder}' ---")

    # Get current reminders from state
    reminders = tool_context.state.get("reminders", [])

    # Add the new reminder
    reminders.append(reminder)

    # Update state with the new list of reminders
    tool_context.state["reminders"] = reminders

    return {
        "action": "add_reminder",
        "reminder": reminder,
        "message": f"Added reminder: {reminder}",
    }


def view_reminders(tool_context: ToolContext) -> dict:
    """View all current reminders.

    Args:
        tool_context: Context for accessing session state

    Returns:
        The list of reminders
    """
    print("--- Tool: view_reminders called ---")

    # Get reminders from state
    reminders = tool_context.state.get("reminders", [])

    return {"action": "view_reminders", "reminders": reminders, "count": len(reminders)}


def update_reminder(index: int, updated_text: str, tool_context: ToolContext) -> dict:
    """Update an existing reminder.

    Args:
        index: The 1-based index of the reminder to update
        updated_text: The new text for the reminder
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(
        f"--- Tool: update_reminder called for index {index} with '{updated_text}' ---"
    )

    # Get current reminders from state
    reminders = tool_context.state.get("reminders", [])

    # Check if the index is valid
    if not reminders or index < 1 or index > len(reminders):
        return {
            "action": "update_reminder",
            "status": "error",
            "message": f"Could not find reminder at position {index}. Currently there are {len(reminders)} reminders.",
        }

    # Update the reminder (adjusting for 0-based indices)
    old_reminder = reminders[index - 1]
    reminders[index - 1] = updated_text

    # Update state with the modified list
    tool_context.state["reminders"] = reminders

    return {
        "action": "update_reminder",
        "index": index,
        "old_text": old_reminder,
        "updated_text": updated_text,
        "message": f"Updated reminder {index} from '{old_reminder}' to '{updated_text}'",
    }


def delete_reminder(index: int, tool_context: ToolContext) -> dict:
    """Delete a reminder.

    Args:
        index: The 1-based index of the reminder to delete
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(f"--- Tool: delete_reminder called for index {index} ---")

    # Get current reminders from state
    reminders = tool_context.state.get("reminders", [])

    # Check if the index is valid
    if not reminders or index < 1 or index > len(reminders):
        return {
            "action": "delete_reminder",
            "status": "error",
            "message": f"Could not find reminder at position {index}. Currently there are {len(reminders)} reminders.",
        }

    # Remove the reminder (adjusting for 0-based indices)
    deleted_reminder = reminders.pop(index - 1)

    # Update state with the modified list
    tool_context.state["reminders"] = reminders

    return {
        "action": "delete_reminder",
        "index": index,
        "deleted_reminder": deleted_reminder,
        "message": f"Deleted reminder {index}: '{deleted_reminder}'",
    }


def update_user_name(name: str, tool_context: ToolContext) -> dict:
    """Update the user's name.

    Args:
        name: The new name for the user
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(f"--- Tool: update_user_name called with '{name}' ---")

    # Get current name from state
    old_name = tool_context.state.get("user_name", "")

    # Update the name in state
    tool_context.state["user_name"] = name

    return {
        "action": "update_user_name",
        "old_name": old_name,
        "new_name": name,
        "message": f"Updated your name to: {name}",
    }



root_agent = Agent(
    name="root_agent",
    model="gemini-2.0-flash",
    description="""
    A root coordinator agent responsible for 
    1. setting user's name in {user_name} state if not already set
    2. managing user interaction flow and forwarding valid user queries to the explainer agent.
    
    INPUT: {user_name}

    FUNCTIONAL OVERVIEW:
    - Handles user identification (collecting and remembering user name).
    - Acts as a smart query router between the user and the explainer agent.
    - Ensures only valid queries are routed to the explainer agent.

    BEHAVIOR RULES:

    1. User Name Handling:
       - If the user's name is not available in user_name state:
         - When the user asks "Do you know my name" or similar, respond by asking them to introduce themselves.
         - When the user provides their name, store it using the update_user_name tool.
         - YOU MUST NOT forward any name-related messages to the explainer agent.
         - After storing the name, politely ask the user to send their actual query.
       - If the user's name is already known, greet them using their name and say "PLEASE ENTER YOUR QUERY NOW {user_name}".

    2. Query Routing:
       - Forward queries to explainer agent ONLY AFTER the user's name is known.
       - Do not forward any of these messages to explainer agent:
         - "Do you know my name"
         - "What's my name"
         - "I am [name]"
         - "My name is [name]"
         - Any other name-related messages
       - For all other messages (once name is known), forward them exactly as-is to the explainer agent.

    3. Communication Style:
       - Be brief, polite, and professional.
       - For name-related messages, respond directly without involving explainer agent.
       - For forwarded queries, acknowledge that the query is being forwarded.

    4. Edge Case Handling:
       - If name is not set and query is not name-related, first ask for name before proceeding.
       - Never forward name-related messages to explainer agent.
    """,
    instruction="""
    Follow these steps exactly:
    1. Check if user_name is set in state
    2. If not set:
       - If message is name-related (asking if you know their name, providing name):
         * Handle it directly (ask for name or store name)
         * DO NOT FORWARD to explainer agent
       - If message is not name-related:
         * First ask user to introduce themselves
    3. If name is set:
       - Forward all non-name-related messages to explainer agent
       - For name-related messages, respond directly (e.g., "Yes, your name is [name]")
    """,
    tools=[update_user_name],
    sub_agents=[explainer_agent],
)

