"""
Full LangGraph Agentic AI Architecture
DeBERTa fine-tuned on Bitext Customer Support → Llama 3.2-3B-Instruct → SQL
"""

# ─────────────────────────────────────────────
# DEPENDENCIES
# pip install langgraph langchain langchain-huggingface
# pip install langchain-community sqlalchemy sqlparse
# pip install langchain-ollama transformers torch
#
# OPTION A (Ollama — recommended):
#   1. Install Ollama from https://ollama.com
#   2. Run: ollama pull llama3.2:3b
#
# OPTION B (HuggingFace — needs GPU or patience on CPU):
#   1. pip install accelerate bitsandbytes
#   2. huggingface-cli login
# ─────────────────────────────────────────────

from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.utilities import SQLDatabase
from transformers import pipeline as hf_pipeline, AutoConfig
import sqlparse


# ─────────────────────────────────────────────
# PATHS — update to match your machine
# ─────────────────────────────────────────────

DEBERTA_PATH = r"C:\Users\User\Desktop\Customer_service_chatbot\out\final_model"


# ─────────────────────────────────────────────
# BITEXT INTENT TAXONOMY
# Exact 27 intents from the Bitext GitHub README:
# https://github.com/bitext/customer-support-llm-chatbot-training-dataset
#
# 10 categories → 27 intents:
#   ACCOUNT:           create_account, delete_account, edit_account,
#                      recover_password, registration_problems, switch_account
#   CANCELLATION_FEE:  check_cancellation_fee
#   CONTACT:           contact_customer_service, contact_human_agent
#   DELIVERY:          delivery_options, delivery_period
#   FEEDBACK:          complaint, review
#   INVOICE:           check_invoice, get_invoice
#   ORDER:             cancel_order, change_order, place_order, track_order
#   PAYMENT:           check_payment_methods, payment_issue
#   REFUND:            check_refund_policy, get_refund, track_refund
#   SHIPPING_ADDRESS:  change_shipping_address, set_up_shipping_address
#   SUBSCRIPTION:      newsletter_subscription
# ─────────────────────────────────────────────

# Intents that should trigger a DB query
# (anything needing order/invoice/refund/payment lookup)
SQL_INTENTS = {
    "track_order",              # ORDER
    "check_invoice",            # INVOICE
    "get_invoice",              # INVOICE
    "check_refund_policy",      # REFUND
    "track_refund",             # REFUND
    "check_payment_methods",    # PAYMENT
    "check_cancellation_fee",   # CANCELLATION_FEE
    "delivery_options",         # DELIVERY
    "delivery_period",          # DELIVERY
    "review",                   # FEEDBACK — look up past order to review
}

# Intents that need a direct conversational response (no DB needed)
CHAT_INTENTS = {
    "cancel_order",             # ORDER
    "change_order",             # ORDER
    "place_order",              # ORDER
    "get_refund",               # REFUND
    "payment_issue",            # PAYMENT
    "complaint",                # FEEDBACK
    "contact_customer_service", # CONTACT
    "contact_human_agent",      # CONTACT
    "create_account",           # ACCOUNT
    "delete_account",           # ACCOUNT
    "edit_account",             # ACCOUNT
    "recover_password",         # ACCOUNT
    "registration_problems",    # ACCOUNT
    "switch_account",           # ACCOUNT
    "change_shipping_address",  # SHIPPING_ADDRESS
    "set_up_shipping_address",  # SHIPPING_ADDRESS
    "newsletter_subscription",  # SUBSCRIPTION
}

# Intents (or fallback labels) that fall outside what this agent handles.
# None of the 27 Bitext intents are out of scope today — this exists as a
# hook for a future "unknown"/low-confidence fallback label if you add one.
OUT_OF_SCOPE_INTENTS = set()

# Friendly display names passed to Llama as intent context
INTENT_DESCRIPTIONS = {
    "track_order":              "tracking an order",
    "check_invoice":            "checking an invoice",
    "get_invoice":              "retrieving an invoice",
    "check_refund_policy":      "checking the refund policy",
    "track_refund":             "tracking a refund",
    "check_payment_methods":    "checking available payment methods",
    "check_cancellation_fee":   "checking the cancellation fee",
    "delivery_options":         "asking about delivery options",
    "delivery_period":          "asking about delivery timeframes",
    "review":                   "reviewing a past order",
    "cancel_order":             "cancelling an order",
    "change_order":             "changing an existing order",
    "place_order":              "placing a new order",
    "get_refund":               "requesting a refund",
    "payment_issue":            "resolving a payment issue",
    "complaint":                "making a complaint",
    "contact_customer_service": "contacting customer service",
    "contact_human_agent":      "requesting a human agent",
    "create_account":           "creating a new account",
    "delete_account":           "deleting an account",
    "edit_account":             "editing account details",
    "recover_password":         "recovering a password",
    "registration_problems":    "resolving registration issues",
    "switch_account":           "switching between accounts",
    "change_shipping_address":  "changing a shipping address",
    "set_up_shipping_address":  "setting up a shipping address",
    "newsletter_subscription":  "managing newsletter subscription",
}


def get_intent_description(intent: str) -> str:
    return INTENT_DESCRIPTIONS.get(intent, intent.replace("_", " "))


# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages:       Annotated[list, add_messages]
    user_input:     str
    intent:         str
    confidence:     float
    sql_query:      Optional[str]
    sql_result:     Optional[str]
    sql_error:      Optional[str]
    needs_sql:      bool
    retry_count:    int
    final_response: str


# ─────────────────────────────────────────────
# 2. MODEL LOADING
# ─────────────────────────────────────────────

def load_deberta():
    """Load the Bitext fine-tuned DeBERTa classifier from local path."""
    print(f"[DeBERTa] Loading from: {DEBERTA_PATH}")

    config = AutoConfig.from_pretrained(DEBERTA_PATH)
    print(f"[DeBERTa] Labels: {config.id2label}")

    classifier = hf_pipeline(
        "text-classification",
        model=DEBERTA_PATH,
        tokenizer=DEBERTA_PATH,
        device=-1,      # CPU; set to 0 for GPU
        top_k=None      # return all label scores
    )
    print("[DeBERTa] Ready.")
    return classifier


def load_llama_ollama():
    """
    OPTION A — Llama 3.2 3B via Ollama (recommended).
    Run first: ollama pull llama3.2:3b
    """
    from langchain_ollama import ChatOllama
    print("[Llama] Loading via Ollama (llama3.2:3b)...")
    llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
    print("[Llama] Ready.")
    return llm


def load_llama_huggingface():
    """
    OPTION B — Llama 3.2 3B from HuggingFace (needs GPU).
    Requires: huggingface-cli login + Meta licence accepted on HF Hub.
    """
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline as hf_pipe
    print("[Llama] Loading meta-llama/Llama-3.2-3B-Instruct from HuggingFace...")
    pipe = hf_pipe(
        "text-generation",
        model="meta-llama/Llama-3.2-3B-Instruct",
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
        device_map="auto",
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("[Llama] Ready.")
    return llm


def load_db(db_uri: str = "sqlite:///customer_service.db"):
    """Connect to customer service SQL database."""
    return SQLDatabase.from_uri(db_uri)


# ─────────────────────────────────────────────
# 3. GRAPH NODES
# ─────────────────────────────────────────────

def node_classify_intent(state: AgentState, deberta) -> AgentState:
    """
    Node 1: DeBERTa classifies user message into one of 27 Bitext intents.
    """
    all_scores = deberta(state["user_input"])[0]
    all_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)

    top        = all_scores[0]
    intent     = top["label"]
    confidence = top["score"]
    needs_sql  = intent in SQL_INTENTS and confidence > 0.7

    print(f"[DeBERTa] Intent: {intent} ({confidence:.2%}) | SQL: {needs_sql}")
    top3 = [(s["label"], f"{s['score']:.2%}") for s in all_scores[:3]]
    print(f"[DeBERTa] Top 3: {top3}")

    return {
        **state,
        "intent":     intent,
        "confidence": confidence,
        "needs_sql":  needs_sql,
        "messages":   state["messages"] + [
            SystemMessage(content=f"Intent: {intent} ({confidence:.2%})")
        ]
    }


def node_generate_sql(state: AgentState, llm, db: SQLDatabase) -> AgentState:
    """
    Node 2: Llama generates SQL based on the detected customer intent.
    The intent is passed as extra context so Llama knows what to query.
    """
    schema      = db.get_table_info()
    intent_desc = get_intent_description(state["intent"])

    prompt = f"""You are a SQL expert for a customer service system.
The customer is {intent_desc}. Write a SQL SELECT query to retrieve the relevant data.

Rules:
- Return ONLY the raw SQL query
- No markdown, no backticks, no explanation
- SELECT queries only

Database schema:
{schema}

Customer message: {state["user_input"]}

SQL:"""

    response  = llm.invoke(prompt)
    sql_query = response.content if hasattr(response, "content") else str(response)
    sql_query = sql_query.strip().split(";")[0] + ";"

    print(f"[Llama SQL] {sql_query}")

    return {
        **state,
        "sql_query": sql_query,
        "sql_error": None
    }


def node_validate_sql(state: AgentState) -> AgentState:
    """
    Node 3: Block any destructive SQL operations.
    """
    sql          = state.get("sql_query", "")
    forbidden_ops = {"DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT"}
    parsed       = sqlparse.parse(sql)

    for statement in parsed:
        for token in statement.flatten():
            if token.ttype is sqlparse.tokens.Keyword.DML:
                if token.value.upper() in forbidden_ops:
                    print(f"[SQL Validator] BLOCKED: {token.value}")
                    return {
                        **state,
                        "sql_error": f"Blocked: {token.value} operation not permitted.",
                        "sql_query": None
                    }

    print("[SQL Validator] Passed.")
    return state


def node_execute_sql(state: AgentState, db: SQLDatabase) -> AgentState:
    """
    Node 4: Execute validated SQL and return results.
    """
    if state.get("sql_error"):
        return state

    try:
        result = db.run(state["sql_query"])
        print(f"[SQL Executor] Result: {str(result)[:200]}")
        return {
            **state,
            "sql_result": result,
            "sql_error":  None
        }
    except Exception as e:
        print(f"[SQL Executor] Error: {e}")
        return {
            **state,
            "sql_result":  None,
            "sql_error":   str(e),
            "retry_count": state.get("retry_count", 0) + 1
        }


def node_generate_response(state: AgentState, llm) -> AgentState:
    """
    Node 5: Llama generates a customer-service-appropriate response.
    Prompt is tailored to the specific Bitext intent.
    """
    intent      = state["intent"]
    intent_desc = get_intent_description(intent)
    sql_result  = state.get("sql_result")
    sql_error   = state.get("sql_error")

    if sql_result:
        # Intent needed DB data — summarise results in a helpful way
        prompt = f"""You are a friendly and professional customer service assistant.
The customer is {intent_desc}. You retrieved the following data from the database.
Summarise it clearly and helpfully. Do not show raw SQL or technical details.

Customer message: {state["user_input"]}
Database results: {sql_result}

Response:"""

    elif sql_error:
        # DB lookup failed — apologise and offer next steps
        prompt = f"""You are a friendly customer service assistant.
The customer is {intent_desc} but a system error occurred when looking up their data.
Apologise briefly and offer to help them another way or escalate to a human agent.

Customer message: {state["user_input"]}
Error: {sql_error}

Response:"""

    elif intent in OUT_OF_SCOPE_INTENTS:
        # Out of scope — politely redirect
        prompt = f"""You are a customer service assistant.
The customer has asked something outside the scope of what you can help with.
Politely explain what you can help with (orders, refunds, account, payments, delivery)
and invite them to ask something within scope.

Customer message: {state["user_input"]}

Response:"""

    else:
        # General customer service intent — conversational response
        prompt = f"""You are a friendly and professional customer service assistant.
The customer is {intent_desc}. Respond helpfully and empathetically.
Keep the response concise and actionable.

Customer message: {state["user_input"]}

Response:"""

    response = llm.invoke(prompt)
    text     = response.content if hasattr(response, "content") else str(response)
    text     = text.strip()

    print(f"[Llama Response] {text[:120]}...")

    return {
        **state,
        "final_response": text,
        "messages":       state["messages"] + [AIMessage(content=text)]
    }


def node_handle_error(state: AgentState) -> AgentState:
    """
    Node 6: Fallback when DeBERTa confidence is too low to route safely.
    """
    response = (
        "I'm sorry, I didn't quite understand your request. "
        "Could you rephrase it? I can help with orders, refunds, "
        "account settings, payments, and delivery queries."
    )
    return {
        **state,
        "final_response": response,
        "messages":       state["messages"] + [AIMessage(content=response)]
    }


# ─────────────────────────────────────────────
# 4. ROUTING FUNCTIONS
# ─────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """
    Three routes based on Bitext intent:
      - Low confidence            → handle_error
      - SQL intent (order/invoice) → generate_sql
      - Everything else           → generate_response (direct Llama reply)
    """
    if state["confidence"] < 0.5:
        return "handle_error"
    if state["needs_sql"]:
        return "generate_sql"
    return "generate_response"


def route_after_execute(state: AgentState) -> str:
    """Retry SQL up to 2 times on error, then fall through to response."""
    if state.get("sql_error") and state.get("retry_count", 0) < 2:
        return "generate_sql"
    return "generate_response"


# ─────────────────────────────────────────────
# 5. GRAPH BUILDER
# ─────────────────────────────────────────────

def build_agent_graph(deberta, llm, db: SQLDatabase):
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent",   lambda s: node_classify_intent(s, deberta))
    graph.add_node("generate_sql",      lambda s: node_generate_sql(s, llm, db))
    graph.add_node("validate_sql",      node_validate_sql)
    graph.add_node("execute_sql",       lambda s: node_execute_sql(s, db))
    graph.add_node("generate_response", lambda s: node_generate_response(s, llm))
    graph.add_node("handle_error",      node_handle_error)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "generate_sql":      "generate_sql",
            "generate_response": "generate_response",
            "handle_error":      "handle_error"
        }
    )

    graph.add_edge("generate_sql", "validate_sql")
    graph.add_edge("validate_sql", "execute_sql")

    graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "generate_sql":      "generate_sql",
            "generate_response": "generate_response"
        }
    )

    graph.add_edge("generate_response", END)
    graph.add_edge("handle_error",      END)

    return graph.compile()


# ─────────────────────────────────────────────
# 6. ENTRYPOINT
# ─────────────────────────────────────────────

def run_agent(user_input: str, graph, conversation_history: list = []):
    initial_state: AgentState = {
        "messages":       conversation_history + [HumanMessage(content=user_input)],
        "user_input":     user_input,
        "intent":         "",
        "confidence":     0.0,
        "sql_query":      None,
        "sql_result":     None,
        "sql_error":      None,
        "needs_sql":      False,
        "retry_count":    0,
        "final_response": ""
    }
    final_state = graph.invoke(initial_state)
    return final_state["final_response"], final_state["messages"]


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading models...")

    deberta = load_deberta()

    # ── Choose ONE: ───────────────────────────
    llm = load_llama_ollama()           # OPTION A: Ollama (recommended)
    # llm = load_llama_huggingface()    # OPTION B: HuggingFace (needs GPU)
    # ─────────────────────────────────────────

    db = load_db("sqlite:///customer_service.db")

    graph = build_agent_graph(deberta, llm, db)

    history = []
    print("\nCustomer Service Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("Customer: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        response, history = run_agent(user_input, graph, history)
        print(f"Agent: {response}\n")
