import operator
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    if dict1 is None:
        dict1 = {}
    if dict2 is None:
        dict2 = {}

    return {**dict1, **dict2}

# ==========================================
# Text2SQL Agent States
# ==========================================

class VectorDBQueries(BaseModel):
    """Output model for generating targeted queries for each Vector DB"""
    schema_query: List[str] = Field(description="List of queries to find similar SQL patterns")
    knowledge_query: List[str] = Field(description="List of queries for domain rules")
    value_query: List[str] = Field(description="List of queries for specific data values")
    example_query: List[str] = Field(description="List of queries to find similar SQL patterns")

class SemanticCheckResult(BaseModel): 
    reasoning: str = Field(description="Explanation of why the SQL is correct or incorrect based on the user question.") 
    is_semantically_correct: bool = Field(description="True if the SQL perfectly matches the user intent. False if logic needs fixing.") 
    corrected_sql: str = Field(description="The fixed SQL query if incorrect. If correct, return the original SQL.") 

class QuerySkeleton(BaseModel):
    operations: List[str] = Field(
        description="High-level SQL operations required: WITH, SELECT, JOIN, GROUP BY, HAVING, ORDER BY, subquery, CTE, etc."
    )
    filters: List[str] = Field(
        description="Filtering conditions in plain English (no table/column names yet). E.g. 'filter by last 30 days', 'only active customers'."
    )
    aggregations: List[str] = Field(
        description="Aggregation intent in plain English. E.g. 'count per region', 'sum of revenue'."
    )
    output_columns: List[str] = Field(
        description="What columns/metrics the user wants returned, described in plain English."
    )
    skeleton_sql: str = Field(
        description=(
            "A draft SQL skeleton using placeholder names (e.g. <table_name>, <column_name>) "
            "that captures the full clause structure: SELECT ... FROM <table> JOIN <table> ON ... "
            "WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT ... "
            "Do NOT use any real table or column names."
        )
    )

class CCDnPPlan(BaseModel):
    from_clause: str = Field(description="FROM and JOIN clauses to use.")
    where_clause: str = Field(
        description="WHERE filters, including date and categorical filters.")
    group_by_clause: str = Field(
        description="GROUP BY clause if needed, otherwise empty.")
    having_clause: str = Field(
        description="HAVING clause if needed, otherwise empty.")
    order_by_clause: str = Field(
        description="ORDER BY clause if needed, otherwise empty.")
    select_clause: str = Field(description="SELECT clause generated last.")
    reasoning: str = Field(
        description="Brief explanation of the clause-by-clause plan.")


class AgentState(TypedDict):
    question: str                                    
    vect_queries: dict                             
    db_results: Annotated[dict, merge_dicts]      
    sql_candidate: Annotated[List[str], operator.add]                       
    formatted_result: str
    data_results: Annotated[List[str], operator.add]
    validation_results: Annotated[List[dict], operator.add]


class SqlValidationState(TypedDict):
    question: str
    db_results: dict
    sql_candidate: str
    is_sql_modified: bool
    query_result: str
    retry_count: int