from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START 
from langgraph.types import Send

from states import AgentState, SqlValidationState, VectorDBQueries, QuerySkeleton, CCDnPPlan
from agents.sql_validation_agent import sql_validation_agent
from prompts import (
    SKELETON_SYSTEM_PROMPT,
    get_skeleton_sql_generation_prompt,       
    TEXT2SQL_DECOMPOSITION_SYSTEM_PROMPT,
    get_text2sql_generation_prompt,
    TEXT2SQL_FORMAT_SYSTEM_PROMPT,
    get_text2sql_format_user_prompt,
    CC_DNP_SYSTEM_PROMPT,
    get_cc_dnp_sql_generation_prompt
)
from models import (
    llm, qwen, coll_schema, coll_examples, coll_evidence, coll_values, llama
)


def generate_vect_db_query(state: AgentState):
    print(f"  [INFO] Generating Vector DB Queries... ")
    
    structured_llm = qwen.with_structured_output(VectorDBQueries)
    
    queries_object = structured_llm.invoke([
        SystemMessage(content=TEXT2SQL_DECOMPOSITION_SYSTEM_PROMPT),
        HumanMessage(content=state['question'])
    ])
    
    print("   schema:   ", queries_object.schema_query)
    print("   evidence: ", queries_object.knowledge_query)
    print("   value:    ", queries_object.value_query)
    print("   example:  ", queries_object.example_query)

    return {
        "vect_queries": {
            "schema": queries_object.schema_query,
            "evidence": queries_object.knowledge_query,
            "value": queries_object.value_query,
            "example": queries_object.example_query
        }
    }

def retrieve_schema(state: AgentState):
    queries = state['vect_queries']
    print(f"  [INFO] Retrieving Schema & Metadata...")

    res_schema = []
    for q_text in queries['schema']:
        results = coll_schema.query(query_texts=[q_text], n_results=1)
        
        if results['documents'] and results['documents'][0]:
            doc_content = "\n".join(results['documents'][0])
            res_schema.append(doc_content)
    
    final_schema = "\n---\n".join(list(set(res_schema)))
    return {"db_results": {"db_schema": final_schema}}

def retrieve_examples(state: AgentState):
    queries = state['vect_queries']
    print(f"  [INFO] Retrieving Examples...")

    search_terms = queries.get("example", state['question'])
    if isinstance(search_terms, list):
        search_term = search_terms[0]
    else:
        search_term = search_terms

    res_examples = coll_examples.query(query_texts=[search_term], n_results=2)
    
    examples_list = []
    if res_examples['metadatas'] and res_examples['metadatas'][0]:
        for meta, doc_text in zip(res_examples['metadatas'][0], res_examples['documents'][0]):
            sql_code = meta.get('query', 'No SQL found')
            examples_list.append(f"Question: {doc_text}\nSQL: {sql_code}")

    examples_txt = "\n---\n".join(examples_list)
    if not examples_txt:
        examples_txt = "No relevant SQL examples found."

    return {"db_results": {"examples": examples_txt}}

def retrieve_evidence(state: AgentState):
    queries = state['vect_queries']
    print(f"  [INFO] Retrieving Evidence...")
    
    search_terms = queries['evidence']
    unique_docs = set()
    
    for term in search_terms:
        res_evidence = coll_evidence.query(query_texts=[term], n_results=4)
        if res_evidence['documents'] and res_evidence['documents'][0]:
            for doc in res_evidence['documents'][0]:
                unique_docs.add(doc)

    evidence_txt = "\n\n".join(list(unique_docs))
    return {"db_results": {"evidence": evidence_txt}}

def retrieve_values(state: AgentState):
    queries = state['vect_queries']
    print(f"  [INFO] Retrieving Values...")
    
    search_terms = queries["value"]
    unique_value_mappings = set()
    
    for term in search_terms:
        res_values = coll_values.query(query_texts=[term], n_results=7)

        if res_values.get('metadatas') and res_values['metadatas'][0]:
            for meta in res_values['metadatas'][0]:
                val = meta.get('value', 'Unknown')
                col = meta.get('column_name', 'Unknown')
                tbl = meta.get('table_name', 'Unknown')
                unique_value_mappings.add(f"Found Value: '{val}' in Table: {tbl}, Column: {col}")

    values_txt = "\n".join(list(unique_value_mappings))
    if not values_txt:
        values_txt = "No specific categorical value matches found."

    return {"db_results": {"values": values_txt}}

def skeleton_generator(state: AgentState):
    print("  [INFO] Generating SQL Skeleton (schema-blind)...")

    structured_qwen = qwen.with_structured_output(QuerySkeleton)
    db_results = state.get("db_results", {})

    full_context = "\n\n".join([
        db_results.get("db_schema", ""),
        db_results.get("evidence", ""),
        db_results.get("values", "")
    ]).strip()

    print("   Full Context for Skeleton Generation:\n", full_context)

    skeleton_obj: QuerySkeleton = structured_qwen.invoke([
        SystemMessage(content=SKELETON_SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ])

    print("   Operations:  ", skeleton_obj.operations)
    print("   Filters:     ", skeleton_obj.filters)
    print("   Aggregations:", skeleton_obj.aggregations)
    print("   Skeleton SQL:\n", skeleton_obj.skeleton_sql)

    prompt = get_skeleton_sql_generation_prompt(skeleton_obj.skeleton_sql, full_context, state['question'])
    response = llm.invoke([prompt])
    cleaned_sql = response.content.replace("```sql", "").replace("```", "").strip()

    return {"sql_candidate": [cleaned_sql]}

def icl_generator(state: AgentState):
    """
    Generate the SQL query using the retrieved context.
    """
    print("  [INFO] Generating SQL   ")

    db_results = state.get("db_results", {})
    full_context = "\n\n".join([
        db_results.get("db_schema", ""),
        db_results.get("evidence", ""),
        db_results.get("values", ""),
        db_results.get("examples", "")
    ]).strip()
    
    prompt = get_text2sql_generation_prompt(full_context, state['question'])
    response = llm.invoke([prompt])
    cleaned_sql = response.content.replace("```sql", "").replace("```", "").strip()

    print("Generated SQL: ", cleaned_sql)
    return {"sql_candidate": [cleaned_sql]}

def cc_dnp_generator(state: AgentState):
    """
    Clause-by-Clause Divide-and-Prompt generator.
    The planner creates an ordered SQL generation plan.
    The code LLM turns the plan into final SQL.
    """
    print("  [INFO] Generating SQL using CC-DnP...")

    db_results = state.get("db_results", {})

    full_context = "\n\n".join([
        db_results.get("db_schema", ""),
        db_results.get("evidence", ""),
        db_results.get("values", ""),
        db_results.get("examples", "")
    ]).strip()

    print("   Full Context for CC-DnP Generation:\n", full_context)

    structured_qwen = qwen.with_structured_output(CCDnPPlan)

    plan_obj: CCDnPPlan = structured_qwen.invoke([
        SystemMessage(content=CC_DNP_SYSTEM_PROMPT),
        HumanMessage(content=f"""
            User question:
            {state['question']}

            Retrieved database context:
            {full_context}
            """)
                ])

    print("   Reasoning: ", plan_obj.reasoning)
    print("   Clause Order: ", plan_obj.clause_generation_order)
    print("   Nested SQL Needed: ", plan_obj.nested_sql_needed)
    print("   Final Plan:\n", plan_obj.final_plan)

    plan_text = f"""
        Reasoning:
        {plan_obj.reasoning}

        Clause generation order:
        {plan_obj.clause_generation_order}

        Nested SQL needed:
        {plan_obj.nested_sql_needed}

        Final plan:
        {plan_obj.final_plan}
    """

    prompt = get_cc_dnp_sql_generation_prompt(
        plan=plan_text,
        db_context=full_context,
        question=state["question"]
    )

    response = llm.invoke([prompt])
    cleaned_sql = response.content.replace("```sql", "").replace("```", "").strip()

    print("Generated CC-DnP SQL: ", cleaned_sql)

    return {"sql_candidate": [cleaned_sql]}

def run_sql_validation_agent(state: SqlValidationState):
    final_validation_state = sql_validation_agent.invoke(state)
    final_sql = final_validation_state.get("sql_candidate", "")
    final_query_result = final_validation_state.get("query_result", "")

    return {
        "validation_results": [
            {
                "sql_candidate": final_sql,
                "query_result": final_query_result,
                "is_sql_modified": final_validation_state.get("is_sql_modified", False),
            }
        ]
    }

def sql_validation(state: AgentState):
    sql_candidates = state.get("sql_candidate", [])

    if not sql_candidates:
        return []

    return [
        Send(
            "sql_validation_agent",
            {
                "question": state["question"],
                "db_results": state.get("db_results", {}),
                "sql_candidate": candidate,
                "is_sql_modified": False,
                "retry_count": state.get("retry_count", 0),
            },
        )
        for candidate in sql_candidates
    ]

def format_result(state: AgentState):
    """
    Convert raw database results into a natural language response.
    """
    print("  [Node] Formatting Final Response")

    validation_results = state.get("validation_results", [])
    if validation_results:
        selected_result = next(
            (result for result in validation_results if "Error:" not in result.get("query_result", "")),
            validation_results[-1],
        )
        raw_result = selected_result.get("query_result", "")
    else:
        raw_result = state.get("query_result", "")

    if "Error:" in raw_result:
        answer = f"I'm sorry, I encountered an issue: {raw_result}"
    else:
        user_question, raw_data = state["question"], raw_result
        
        user_message = get_text2sql_format_user_prompt(user_question, raw_data)
        response = llm.invoke([
            SystemMessage(content=TEXT2SQL_FORMAT_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        answer = response.content
        print("Final Answer: ", answer)

    return {
        "formatted_result": answer,
        "data_results": [answer]
    }      


workflow = StateGraph(AgentState)

workflow.add_node("generate_vect_db_query", generate_vect_db_query)
workflow.add_node("schema_db", retrieve_schema)
workflow.add_node("example_db", retrieve_examples)
workflow.add_node("evidence_db", retrieve_evidence)
workflow.add_node("cell_value_db", retrieve_values)

workflow.add_node("skeleton_generator", skeleton_generator)
workflow.add_node("icl_generator", icl_generator)
workflow.add_node("cc_dnp_generator", cc_dnp_generator)
workflow.add_node("sql_validation_agent", run_sql_validation_agent)
# workflow.add_node("format_result", format_result)

workflow.add_edge(START, "generate_vect_db_query")

workflow.add_edge("generate_vect_db_query", "schema_db")
workflow.add_edge("generate_vect_db_query", "example_db")
workflow.add_edge("generate_vect_db_query", "evidence_db")
workflow.add_edge("generate_vect_db_query", "cell_value_db")

workflow.add_edge("schema_db", "skeleton_generator")
# workflow.add_edge("example_db", "skeleton_generator")
workflow.add_edge("evidence_db", "skeleton_generator")
workflow.add_edge("cell_value_db", "skeleton_generator")
workflow.add_conditional_edges("skeleton_generator", sql_validation, ["sql_validation_agent"])
workflow.add_edge("sql_validation_agent", END)

workflow.add_edge("schema_db", "icl_generator")
workflow.add_edge("example_db", "icl_generator")
workflow.add_edge("evidence_db", "icl_generator")
workflow.add_edge("cell_value_db", "icl_generator")
workflow.add_conditional_edges("icl_generator", sql_validation, ["sql_validation_agent"])
workflow.add_edge("sql_validation_agent", END)

workflow.add_edge("schema_db", "cc_dnp_generator")
# workflow.add_edge("example_db", "cc_dnp_generator")
# workflow.add_edge("evidence_db", "cc_dnp_generator")
workflow.add_edge("cell_value_db", "cc_dnp_generator")
workflow.add_conditional_edges("cc_dnp_generator", sql_validation, ["sql_validation_agent"])
workflow.add_edge("sql_validation_agent", END)

# workflow.add_edge("sql_validation_agent", "format_result")

# workflow.add_edge("format_result", END)
text2sql_agent = workflow.compile()
