from agents.text2sql import text2sql_agent
import pprint

initial_state = {
    "question": "What is the total revenue of *2 recharge types in January 2026?",
}

img_bytes = text2sql_agent.get_graph(xray=1).draw_mermaid_png()

with open("assets/text2sql_agent.png", "wb") as f:
    f.write(img_bytes)

print("🚀 Starting Text-to-SQL Workflow...")

try:
    # for event in text2sql_agent.stream(initial_state, {"recursion_limit": 15}):
    #     for node_name, state_update in event.items():
    #         print("-" * 40)
    #         pprint.pprint(state_update)
    #         print("-" * 40)
    final_state = text2sql_agent.invoke(initial_state, {"recursion_limit": 15})
    
    print("\n✅ Workflow completed successfully!")
    print("Final Validation Results:")
    
    pprint.pprint(final_state.get('validation_results', 'No validation results generated.'))
    print("\nFinal Result:")
    pprint.pprint(final_state.get('final_result', 'No final result generated.'))

except Exception as e:
    print(f"\n❌ Workflow failed or hit recursion limit: {e}")