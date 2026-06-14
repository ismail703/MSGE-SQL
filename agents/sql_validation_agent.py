from states import SqlValidationState, SemanticCheckResult
import sqlite3
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import get_text2sql_debugger_system_prompt, get_text2sql_semantic_system_prompt, get_text2sql_semantic_user_prompt
from langgraph.graph import StateGraph, END, START 
from typing import Literal
from models import llm, qwen, DB_PATH, llama
import random


def syntax_checker(state: SqlValidationState):
    """
    Node: Attempt to execute the SQL. If it fails, ask the LLM to fix it.
    """
    current_sql = state["sql_candidate"]
    retries = state.get("retry_count", 0)
    
    print(f"  [INFO] Checking Syntax (Attempt {retries + 1})")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(current_sql)
        result_data = cursor.fetchall()
        conn.close()

        print("   [Success] SQL executed successfully.")
        return {
            "query_result": str(result_data),
            "is_sql_modified": False,
            "retry_count": 0
        }

    except Exception as e:
        error_msg = str(e)
        print(f"   [Error] SQL failed: {error_msg}")
            
        if retries >= 3:
            print("   [Fail] Max retries reached.")
            return {
                "is_sql_modified": False,
                "query_result": f"Error: Failed after 3 attempts. Last error: {error_msg}",
                "retry_count": 0
            }
    
        db_results = state.get("db_results", {})
        full_context = "\n\n".join([
            db_results.get("db_schema", ""),
            db_results.get("evidence", ""),
            db_results.get("values", ""),
            db_results.get("examples", "")
        ]).strip()

        system_prompt = get_text2sql_debugger_system_prompt(full_context)
    
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original Query: {current_sql}\nSQLite Error: {error_msg}")
        ]
    
        response = llama.invoke(messages)
        fixed_sql = response.content.replace("```sql", "").replace("```", "").strip()
        print("Fixed SQL: ", fixed_sql)

        return {
            "sql_candidate": fixed_sql,
            "is_sql_modified": True,
            "retry_count": retries + 1
        }

def should_continue_syntax(state: SqlValidationState) -> Literal["syntax_checker", "semantic_checker"]:
    if state.get("is_sql_modified", False):
        return "syntax_checker"
    else:
        return "semantic_checker"        

def semantic_checker(state: SqlValidationState):
    """
    Audit the SQL logic against the user's original intent.
    """
    print("  [INFO] Semantic Logic Review ")
    current_sql = state["sql_candidate"]
    original_question = state["question"]

    rand_num = random.random()

    if rand_num > 0.6:
        structured_llm = llm.with_structured_output(SemanticCheckResult)    
    elif rand_num > 0.3:
        structured_llm = llama.with_structured_output(SemanticCheckResult)
    else:
        structured_llm = qwen.with_structured_output(SemanticCheckResult)
    
    db_results = state.get("db_results", {})
    full_context = "\n\n".join([
        db_results.get("db_schema", ""),
        db_results.get("evidence", ""),
        db_results.get("values", ""),
        db_results.get("examples", "")
    ]).strip()

    system_prompt = get_text2sql_semantic_system_prompt(full_context)
    user_prompt = get_text2sql_semantic_user_prompt(original_question, current_sql)

    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    print(f"   [Reasoning]: {result.reasoning}")

    if result.is_semantically_correct:
        print("   [Success] Logic is sound.")
        return {"is_sql_modified": False}
    else:       
        print("   [Warning] Logic error detected. Updating SQL.")
        print(f"  [INFO] Corrected SQL: {result.corrected_sql}")
        return {
            "sql_candidate": result.corrected_sql,
            "is_sql_modified": True,
            "retry_count": 0
        }

def check_semantic_modification(state: SqlValidationState) -> Literal["syntax_checker", "format_result"]:
    if state.get("is_sql_modified", False):
        print("   >> Looping back to Syntax Checker")
        return "syntax_checker"
    else:
        print("   >> Proceeding to Finish")
        return "format_result"

workflow = StateGraph(SqlValidationState)

workflow.add_node("syntax_checker", syntax_checker)
workflow.add_node("semantic_checker", semantic_checker)

workflow.add_edge(START, "syntax_checker")

workflow.add_conditional_edges(
    "syntax_checker",
    should_continue_syntax,
    {
        "syntax_checker": "syntax_checker",
        "semantic_checker": "semantic_checker"
    }
)

workflow.add_conditional_edges(
    "semantic_checker",
    check_semantic_modification,
    {
        "syntax_checker": "syntax_checker",
        "format_result": END
    }
)

sql_validation_agent = workflow.compile()