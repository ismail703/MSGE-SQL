# MSGE-SQL (Multi-Strategy Generation and Evaluation for SQL)

**Project Overview**
MSGE-SQL is a context-enriched, multi-strategy framework for generating robust SQL queries from natural language questions. The system leverages Large Language Models (LLMs) to generate multiple diverse SQL candidates using different strategies, enriches them with relevant schema and domain knowledge, and employs intelligent validation and selection mechanisms to ensure the generated queries are both syntactically correct and semantically aligned with user intent.

**Purpose**: Implement a Text-to-SQL pipeline that decomposes user questions, retrieves grounding information from vector DBs, and generates/validates executable SQL queries using multiple complementary strategies.

![Multi-agent architecture](assets/text2sql_agent.png)

**Key Features**:
- **Context-Aware Generation**: Provides LLMs with rich contextual information including database schemas (DDL and Markdown formats), question-SQL examples, categorical database values, and domain-specific knowledge.
- **Multi-Strategy SQL Generation**: Employs three parallel approaches:
  - In-Context Learning (ICL) with few-shot examples
  - Clause-by-Clause Divide-and-Prompt (CC-DnP) for complex queries
  - Skeleton-based method with reasoning models
- **Intelligent Validation**: Two-stage validation loop that checks for both syntactic errors and semantic alignment with user intent.
- **LLM-as-a-Judge Selection**: Evaluates and selects the optimal SQL query from multiple candidates.

**Repository Structure**
- **`models.py`**: LLM and vector DB client configuration.
- **`prompts.py`**: System and user prompt templates used by the LLMs.
- **`states.py`**: Typed state definitions for agents and merge rules.
- **`text2sql.py`**: Main workflow graph that wires retrieval, generation, and validation nodes.
- **`sql_validation_agent.py`**: Subgraph that runs `syntax_checker` and `semantic_checker` loops.
- **`main.py`**: Small runner that can generate a `text2sql_agent.png` diagram and invoke the workflow.
- **`chroma_db_store/`**: Persistent Chroma DB storage with collections used for grounding.

**Getting Started**

**Data Storage & Embedding**:
- All database metadata, schemas, and domain knowledge are embedded and stored in a **SQLite-backed vector database** (Chroma DB).
- The system maintains separate collections for:
  - **Database Schema**: Markdown-formatted schema representations
  - **Question-SQL Examples**: Masked question-SQL pairs for in-context learning
  - **Categorical Values**: Valid database values to prevent exact data mismatches
  - **Domain Knowledge**: Business rules, guidelines, and domain-specific constraints
- During the offline preprocessing phase, all textual information is embedded into vector representations and indexed for efficient semantic retrieval during inference.
- SQLite provides persistent storage, ensuring the embeddings and schema information persist across sessions and are readily available for querying.

**Running the Workflow**:
1. Ensure your LLM and vector DB credentials are configured in `models.py`.
2. Run the workflow locally via `python main.py` to:
   - Execute a sample Text-to-SQL query
   - Generate and display the agent architecture diagram (`text2sql_agent.png`)
   - Inspect the validation and selection pipeline in action.
3. Provide a natural language question, and the system will:
   - Retrieve relevant context from the SQLite-backed vector DB
   - Generate multiple SQL candidates using different strategies
   - Validate each candidate for syntax and semantic correctness
   - Select and return the optimal SQL query

**Multi-Agent Architecture**
- **Overview**: The system builds retrieval queries from the user question, retrieves schema/evidence/values/examples from vector stores, then runs up to three parallel SQL generation strategies (skeleton-based, in-context learning, clause-by-clause). Each generated SQL candidate is validated by the `sql_validation_agent` which runs syntax checks and a semantic audit.

**Runtime notes & token limits**
- The `semantic_checker` uses the `llm` from `models.py` (by default set to `openai/gpt-oss-120b` in this workspace). That model and your API plan have a tokens-per-minute (TPM) limit; running multiple generation branches in parallel (and having large prompts / long `full_context`) can exceed that limit and produce 429 rate-limit errors.
- Short-term mitigations:
  - Reduce parallel fan-out in `text2sql.py` so only one generator runs per request.
  - Use a smaller/cheaper model for semantic checks in `models.py`.
  - Trim `full_context` passed into semantic prompts (`prompts.py`) to fewer lines.
  
**Contributing**
- Run the workflow locally via `python main.py` and inspect the saved diagram.
- Please open issues for edge cases (e.g., inconsistent state types) and include the traceback and input question.

**License & Credits**
- Project created for a Master-level Text2SQL research exercise. Reuse and modification are allowed for research and educational purposes.
