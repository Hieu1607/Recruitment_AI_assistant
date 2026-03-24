# LangGraph nodes execution


def router_node(state):
    # Call LLM to decide route
    return {"route": "DSL"}


def dsl_node(state):
    # Call DSL tool
    return {"dsl_result": [...]}


def llm_node(state):
    # Call LLM tool
    return {"llm_result": "answer"}


def merge_node(state):
    # Combine results
    return {"final_answer": "..."}
