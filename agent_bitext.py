"""
Full LangGraph Agentic AI Architecture
DeBERTa fine-tuned on Bitext Customer Support → Llama 3.2-3B-Instruct → SQL
"""

# ─────────────────────────────────────────────
# DEPENDENCIES
# pip install langgraph langchain langchain-huggingface
# pip install langchain-community sqlalchemy sqlparse
# pip install langchain-ollama transformers torch
# pip install pyenchant   ← needed for gibberish/pinyin detection
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
# DB SCHEMA
# Exact schema from your customer_service.db.
# Used as fallback context when no SQL template exists.
# ─────────────────────────────────────────────

DB_SCHEMA = """
Table: customers
  - customer_id    INTEGER PRIMARY KEY
  - full_name      TEXT
  - email          TEXT

Table: orders
  - order_id       INTEGER PRIMARY KEY
  - customer_id    INTEGER   REFERENCES customers(customer_id)
  - order_date     TEXT
  - status         TEXT
  - total_amount   REAL

Table: order_items
  - item_id        INTEGER PRIMARY KEY
  - order_id       INTEGER   REFERENCES orders(order_id)
  - product_name   TEXT
  - quantity       INTEGER
  - unit_price     REAL

Table: invoices
  - invoice_id     INTEGER PRIMARY KEY
  - order_id       INTEGER   REFERENCES orders(order_id)
  - issued_date    TEXT
  - amount         REAL
  - status         TEXT

Table: payment_methods
  - method_id      INTEGER PRIMARY KEY
  - method_name    TEXT
  - is_active      INTEGER

Table: payments
  - payment_id     INTEGER PRIMARY KEY
  - order_id       INTEGER   REFERENCES orders(order_id)
  - method_id      INTEGER   REFERENCES payment_methods(method_id)
  - amount         REAL
  - status         TEXT
  - payment_date   TEXT

Table: refunds
  - refund_id        INTEGER PRIMARY KEY
  - order_id         INTEGER   REFERENCES orders(order_id)
  - amount           REAL
  - status           TEXT
  - requested_date   TEXT
  - processed_date   TEXT

Table: refund_policy
  - policy_id      INTEGER PRIMARY KEY
  - category       TEXT
  - window_days    INTEGER
  - description    TEXT

Table: deliveries
  - delivery_id          INTEGER PRIMARY KEY
  - order_id             INTEGER   REFERENCES orders(order_id)
  - carrier              TEXT
  - status               TEXT
  - estimated_delivery   TEXT
  - actual_delivery      TEXT

Table: delivery_options
  - option_id        INTEGER PRIMARY KEY
  - region           TEXT
  - method           TEXT
  - cost             REAL
  - estimated_days   TEXT

Table: cancellation_policy
  - policy_id      INTEGER PRIMARY KEY
  - order_status   TEXT
  - fee_percent    REAL
  - description    TEXT

Table: reviews
  - review_id      INTEGER PRIMARY KEY
  - order_id       INTEGER   REFERENCES orders(order_id)
  - customer_id    INTEGER   REFERENCES customers(customer_id)
  - rating         INTEGER
  - comment        TEXT
  - review_date    TEXT

IMPORTANT RULES FOR SQL GENERATION:
- Use EXACT column names above — do not invent columns
- customers uses full_name NOT name
- invoices uses issued_date NOT issued_at, paid_at does NOT exist
- refunds uses requested_date NOT requested_at, processed_date NOT completed_at
- payments links to payment method via method_id → payment_methods.method_id
- deliveries has NO customer_id — always join through orders
- delivery_options has NO order_id — it stores general options, not per-order
- Always use explicit table aliases (e.g. o, c, d) when joining
- Only generate SELECT queries, never INSERT/UPDATE/DELETE/DROP
"""


# ─────────────────────────────────────────────
# SQL TEMPLATES
# Pre-written, verified queries for every SQL intent.
# These completely bypass Llama SQL generation —
# no hallucinated columns, no failed JOINs, no retries.
# ─────────────────────────────────────────────

SQL_TEMPLATES = {
    "track_order": """
        SELECT o.order_id, o.order_date, o.status, o.total_amount,
               d.carrier, d.estimated_delivery, d.actual_delivery, d.status AS delivery_status
        FROM orders o
        LEFT JOIN deliveries d ON o.order_id = d.order_id
        ORDER BY o.order_date DESC
        LIMIT 10;
    """,

    "check_invoice": """
        SELECT i.invoice_id, o.order_id, o.order_date,
               i.issued_date, i.amount, i.status
        FROM invoices i
        JOIN orders o ON i.order_id = o.order_id
        ORDER BY i.issued_date DESC
        LIMIT 10;
    """,

    "get_invoice": """
        SELECT i.invoice_id, o.order_id, o.order_date,
               i.issued_date, i.amount, i.status
        FROM invoices i
        JOIN orders o ON i.order_id = o.order_id
        ORDER BY i.issued_date DESC
        LIMIT 10;
    """,

    "track_refund": """
        SELECT r.refund_id, o.order_id, o.order_date,
               r.amount, r.status, r.requested_date, r.processed_date
        FROM refunds r
        JOIN orders o ON r.order_id = o.order_id
        ORDER BY r.requested_date DESC
        LIMIT 10;
    """,

    "check_refund_policy": """
        SELECT policy_id, category, window_days, description
        FROM refund_policy
        ORDER BY policy_id;
    """,

    "check_payment_methods": """
        SELECT method_name
        FROM payment_methods
        WHERE is_active = 1
        ORDER BY method_name;
    """,

    "check_cancellation_fee": """
        SELECT order_status, fee_percent, description
        FROM cancellation_policy
        ORDER BY fee_percent;
    """,

    "delivery_options": """
        SELECT region, method, cost, estimated_days
        FROM delivery_options
        ORDER BY cost;
    """,

    "delivery_period": """
        SELECT o.order_id, o.order_date, o.status,
               d.carrier, d.estimated_delivery, d.actual_delivery, d.status AS delivery_status
        FROM deliveries d
        JOIN orders o ON d.order_id = o.order_id
        ORDER BY o.order_date DESC
        LIMIT 10;
    """,

    "review": """
        SELECT r.review_id, o.order_id, o.order_date,
               c.full_name, r.rating, r.comment, r.review_date
        FROM reviews r
        JOIN orders o ON r.order_id = o.order_id
        JOIN customers c ON r.customer_id = c.customer_id
        ORDER BY r.review_date DESC
        LIMIT 10;
    """,

    # Product price / info — triggered by "how much is X" keyword override
    "review_order": """
        SELECT oi.product_name, oi.unit_price, oi.quantity,
               o.status, o.order_date
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        ORDER BY o.order_date DESC
        LIMIT 20;
    """,
}


# ─────────────────────────────────────────────
# BITEXT INTENT TAXONOMY
# Exact 27 intents from the Bitext GitHub README:
# https://github.com/bitext/customer-support-llm-chatbot-training-dataset
# ─────────────────────────────────────────────

SQL_INTENTS = {
    "track_order",
    "check_invoice",
    "get_invoice",
    "check_refund_policy",
    "track_refund",
    "check_payment_methods",
    "check_cancellation_fee",
    "delivery_options",
    "delivery_period",
    "review",
    "review_order",         # product price / info lookup
}

CHAT_INTENTS = {
    "cancel_order",
    "change_order",
    "place_order",
    "get_refund",
    "payment_issue",
    "complaint",
    "contact_customer_service",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "edit_account",
    "recover_password",
    "registration_problems",
    "switch_account",
    "change_shipping_address",
    "set_up_shipping_address",
    "newsletter_subscription",
}

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


def clean_response(text: str) -> str:
    """
    Post-process Llama output to strip prompt leaking.
    Llama 3.2-3B sometimes repeats the prompt or adds
    'Example:', 'Note:', 'Please respond' etc. after its answer.
    This cuts the response at the first sign of leaking.
    """
    # Phrases that signal Llama has started repeating/leaking the prompt
    stop_phrases = [
        "\nPlease respond",
        "\nExample of",
        "\nHere is the response",
        "\n(Note:",
        "\nNote:",
        "\nCustomer message:",
        "\nCustomer question:",
        "\nDatabase results:",
        "\nYour response",
        "\nResponse:",
        "\nA good response",
    ]
    for phrase in stop_phrases:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx].strip()

    # Also strip if Llama wraps in quotes — unwrap it
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    return text.strip()


# ─────────────────────────────────────────────
# KEYWORD OVERRIDE
# Temporary safety net while DeBERTa is retrained on Bitext data.
# Runs AFTER input validation but BEFORE DeBERTa — if a keyword
# pattern clearly matches an intent, DeBERTa is skipped entirely.
#
# Priority order inside each intent: more specific phrases first.
# ─────────────────────────────────────────────

KEYWORD_OVERRIDES: dict[str, list[str]] = {
    "track_order": [
        "where is my order", "track my order", "order status",
        "when will my order", "has my order", "where is my",
        "track order", "order tracking", "shipping status",
    ],
    "check_invoice": [
        "check my invoice", "see my invoice", "view invoice",
        "invoice status", "show invoice",
    ],
    "get_invoice": [
        "get my invoice", "send me invoice", "download invoice",
        "email me invoice", "i need my invoice",
    ],
    "check_refund_policy": [
        "refund policy", "return policy", "what is your refund",
        "can i return", "how do i return",
    ],
    "track_refund": [
        "where is my refund", "refund status", "track my refund",
        "when will i get my refund", "has my refund",
    ],
    "check_payment_methods": [
        "payment methods", "how can i pay", "payment options",
        "do you accept", "ways to pay",
    ],
    "check_cancellation_fee": [
        "cancellation fee", "cancel fee", "how much to cancel",
        "fee to cancel",
    ],
    "delivery_options": [
        "delivery options", "shipping options", "how much is delivery",
        "delivery cost", "shipping cost", "shipping methods",
    ],
    "delivery_period": [
        "how long does delivery", "delivery time", "when will it arrive",
        "estimated delivery", "how many days",
    ],
    "review": [
        "leave a review", "write a review", "rate my order",
        "submit review", "give feedback",
    ],
    "cancel_order": [
        "cancel my order", "cancel order", "i want to cancel",
        "stop my order",
    ],
    "get_refund": [
        "i want a refund", "request a refund", "give me a refund",
        "money back", "refund my order", "refund please",
    ],
    "place_order": [
        "place an order", "i want to order", "how do i order",
        "buy", "purchase",
    ],
    "complaint": [
        "i want to complain", "make a complaint", "this is unacceptable",
        "i am unhappy", "i am disappointed", "terrible service",
        "awful", "worst",
    ],
    "contact_human_agent": [
        "speak to a human", "talk to a person", "human agent",
        "real person", "live agent", "speak to an agent",
    ],
    "contact_customer_service": [
        "contact customer service", "customer support", "help desk",
        "reach support",
    ],
    "recover_password": [
        "forgot my password", "reset my password", "recover password",
        "can't log in", "cannot log in", "lost my password",
    ],
    "payment_issue": [
        "payment failed", "payment not working", "couldn't pay",
        "charge failed", "billing problem", "payment problem",
    ],
    # Product price / info lookup — routes to SQL to check order_items
    "review_order": [
        "how much is", "what is the price", "price of",
        "cost of", "how much does", "how much do",
    ],
}


def keyword_override(text: str) -> Optional[str]:
    """
    Check if input clearly matches a known intent by keyword patterns.
    Returns the matched intent string, or None if no match.
    More specific phrases are listed first inside each intent.
    """
    text_lower = text.lower()
    for intent, keywords in KEYWORD_OVERRIDES.items():
        if any(kw in text_lower for kw in keywords):
            print(f"[Keyword Override] Matched intent: {intent!r}")
            return intent
    return None


# ─────────────────────────────────────────────
# INPUT VALIDATION — TWO-LAYER DETECTION
# Layer 1: non-ASCII check   — catches CJK/Arabic scripts (我问为什么)
# Layer 2a: enchant check    — catches romanised non-English via dictionary
# Layer 2b: pinyin syllables — fallback when enchant is not installed,
#                              catches pinyin specifically (wo bu zhi dao…)
#
# pip install pyenchant   (optional but recommended for Layer 2a)
# Windows enchant C lib:  https://pyenchant.github.io/pyenchant/install.html
# ─────────────────────────────────────────────

# Attempt to load pyenchant once at startup
try:
    import enchant as _enchant
    _EN_DICT    = _enchant.Dict("en_US")
    _ENCHANT_OK = True
    print("[Input Guard] pyenchant loaded — dictionary check active.")
except Exception:
    _EN_DICT    = None
    _ENCHANT_OK = False
    print("[Input Guard] pyenchant not found — using pinyin syllable fallback.")

# Comprehensive pinyin syllable list used when enchant is unavailable.
# Covers all standard Mandarin pinyin initials + finals combinations.
_PINYIN_SYLLABLES = {
    "a","o","e","ai","ei","ao","ou","an","en","ang","eng","er",
    "ba","bo","bai","bei","bao","ban","ben","bang","beng","bi","bie","biao",
    "bian","bin","bing","bu",
    "pa","po","pai","pei","pao","pou","pan","pen","pang","peng","pi","pie",
    "piao","pian","pin","ping","pu",
    "ma","mo","me","mai","mei","mao","mou","man","men","mang","meng","mi",
    "mie","miao","miu","mian","min","ming","mu",
    "fa","fo","fei","fou","fan","fen","fang","feng","fu",
    "da","de","dai","dei","dao","dou","dan","den","dang","deng","di","die",
    "diao","diu","dian","ding","dong","dou","du","duan","dui","dun","duo",
    "ta","te","tai","tao","tou","tan","tang","teng","ti","tie","tiao","tian",
    "ting","tong","tu","tuan","tui","tun","tuo",
    "na","ne","nai","nei","nao","nou","nan","nen","nang","neng","ni","nie",
    "niao","niu","nian","nin","niang","ning","nong","nu","nuan","nun","nuo","nv",
    "la","le","lai","lei","lao","lou","lan","lang","leng","li","lia","lie",
    "liao","liu","lian","lin","liang","ling","long","lu","luan","lun","luo","lv",
    "ga","ge","gai","gei","gao","gou","gan","gen","gang","geng","gong","gu",
    "gua","guai","guan","guang","gui","gun","guo",
    "ka","ke","kai","kei","kao","kou","kan","ken","kang","keng","kong","ku",
    "kua","kuai","kuan","kuang","kui","kun","kuo",
    "ha","he","hai","hei","hao","hou","han","hen","hang","heng","hong","hu",
    "hua","huai","huan","huang","hui","hun","huo",
    "ji","jia","jie","jiao","jiu","jian","jin","jiang","jing","jiong","ju",
    "juan","jun","jue",
    "qi","qia","qie","qiao","qiu","qian","qin","qiang","qing","qiong","qu",
    "quan","qun","que",
    "xi","xia","xie","xiao","xiu","xian","xin","xiang","xing","xiong","xu",
    "xuan","xun","xue",
    "zhi","zha","zhe","zhai","zhei","zhao","zhou","zhan","zhen","zhang","zheng",
    "zhong","zhu","zhua","zhuai","zhuan","zhuang","zhui","zhun","zhuo",
    "chi","cha","che","chai","chao","chou","chan","chen","chang","cheng","chong",
    "chu","chua","chuai","chuan","chuang","chui","chun","chuo",
    "shi","sha","she","shai","shei","shao","shou","shan","shen","shang","sheng",
    "shu","shua","shuai","shuan","shuang","shui","shun","shuo",
    "ri","re","rao","rou","ran","ren","rang","reng","rong","ru","rua","ruan",
    "rui","run","ruo",
    "zi","za","ze","zai","zei","zao","zou","zan","zen","zang","zeng","zong",
    "zu","zuan","zui","zun","zuo",
    "ci","ca","ce","cai","cao","cou","can","cen","cang","ceng","cong","cu",
    "cuan","cui","cun","cuo",
    "si","sa","se","sai","sao","sou","san","sen","sang","seng","song","su",
    "suan","sui","sun","suo",
    "yi","ya","ye","yai","yao","you","yan","yin","yang","ying","yong","yu",
    "yuan","yun","yue",
    "wa","wo","wai","wei","wan","wen","wang","weng","wu",
    # common standalone syllables often seen in pinyin text
    "ni","ta","wo","bu","shi","de","le","ge","ne","ma","ba","na","zai","lai",
    "dao","shuo","kan","gei","rang","bei","mei","men","hai","dui","hen","zhen",
    "hao","xie","xia","wei","qing","neng","yao","hui","xin","ai","kuai","shen",
    "mo","dao","jiu","zhe","na","lao","yong","cheng","ming","xian","zen","me",
}


def _is_pinyin(text: str) -> bool:
    """Return True if >= 50% of words are recognised pinyin syllables."""
    words = text.lower().split()
    if len(words) < 2:
        return False
    hits  = sum(1 for w in words if w in _PINYIN_SYLLABLES)
    ratio = hits / len(words)
    print(f"[Input Guard] Pinyin syllable check: {hits}/{len(words)} = {ratio:.0%}")
    return ratio >= 0.5


def is_valid_input(text: str) -> tuple[bool, str]:
    """
    Validate user input before passing to DeBERTa.

    Returns:
        (True,  "ok")          — valid English, proceed to classification
        (False, "non_english") — non-ASCII script (CJK, Arabic, Cyrillic…)
        (False, "gibberish")   — ASCII but not English (pinyin, random text…)
    """
    # ── Layer 1: non-ASCII script ─────────────
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False, "non_english"

    # ── Layer 2a: enchant dictionary check ───
    if _ENCHANT_OK:
        words = [w.strip(".,!?;:'\"()[]") for w in text.lower().split()]
        words = [w for w in words if len(w) > 2]
        if len(words) >= 2:
            valid = sum(1 for w in words if _EN_DICT.check(w))
            ratio = valid / len(words)
            print(f"[Input Guard] Dictionary check: {valid}/{len(words)} = {ratio:.0%}")
            if ratio < 0.4:
                return False, "gibberish"
        return True, "ok"

    # ── Layer 2b: pinyin syllable fallback ───
    if _is_pinyin(text):
        return False, "gibberish"

    return True, "ok"


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
    used_template:  bool           # tracks whether a template was used
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
        device=-1,                          # CPU; set to 0 for GPU
        top_k=None,                         # return all label scores
        clean_up_tokenization_spaces=False  # suppress BPE tokenizer warning
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
        max_new_tokens=512,     # controls output length — max_length not needed
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
    Two-layer input validation runs first:
      - non-ASCII script  → "non_english"
      - romanised pinyin / gibberish → "gibberish"
    Both are routed to handle_error without touching DeBERTa.
    """
    # ── Two-layer input validation ────────────
    valid, reason = is_valid_input(state["user_input"])
    if not valid:
        label = "non_english" if reason == "non_english" else "gibberish"
        print(f"[Input Guard] Rejected — reason: {label}")
        return {
            **state,
            "intent":        label,
            "confidence":    1.0,
            "needs_sql":     False,
            "used_template": False,
            "messages":      state["messages"] + [
                SystemMessage(content=f"Intent: {label} (input validation failed)")
            ]
        }
    # ─────────────────────────────────────────

    # ── Keyword override — runs before DeBERTa ─
    override = keyword_override(state["user_input"])
    if override:
        needs_sql = override in SQL_INTENTS
        return {
            **state,
            "intent":        override,
            "confidence":    1.0,
            "needs_sql":     needs_sql,
            "used_template": False,
            "messages":      state["messages"] + [
                SystemMessage(content=f"Intent: {override} (keyword override)")
            ]
        }
    # ─────────────────────────────────────────

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
        "intent":        intent,
        "confidence":    confidence,
        "needs_sql":     needs_sql,
        "used_template": False,
        "messages":      state["messages"] + [
            SystemMessage(content=f"Intent: {intent} ({confidence:.2%})")
        ]
    }


def node_generate_sql(state: AgentState, llm, db: SQLDatabase) -> AgentState:
    """
    Node 2: Fetch SQL query for the detected intent.

    Priority:
      1. SQL_TEMPLATES — pre-written verified queries (no Llama needed)
      2. Llama fallback — only if no template exists for this intent
    """
    intent = state["intent"]

    # ── Priority 1: use pre-written template ──
    if intent in SQL_TEMPLATES:
        sql_query = SQL_TEMPLATES[intent].strip()
        print(f"[SQL Template] Intent '{intent}' → using verified template.")
        print(f"[SQL Template] {sql_query[:120]}...")
        return {
            **state,
            "sql_query":     sql_query,
            "sql_error":     None,
            "used_template": True,
        }

    # ── Priority 2: Llama fallback for unknown intents ──
    print(f"[SQL Llama] No template for '{intent}' — asking Llama to generate SQL.")
    intent_desc = get_intent_description(intent)

    prompt = f"""You are a SQL expert for a customer service database.
The customer is {intent_desc}. Write a valid SQL SELECT query.

Rules:
- Return ONLY the raw SQL query, no explanation, no backticks
- Use ONLY the exact table and column names listed in the schema
- Always use explicit table aliases when joining tables
- SELECT queries only

Schema:
{DB_SCHEMA}

Customer message: {state["user_input"]}

SQL:"""

    response  = llm.invoke(prompt)
    sql_query = response.content if hasattr(response, "content") else str(response)
    sql_query = sql_query.strip().split(";")[0] + ";"

    print(f"[SQL Llama] {sql_query[:120]}...")

    return {
        **state,
        "sql_query":     sql_query,
        "sql_error":     None,
        "used_template": False,
    }


def node_validate_sql(state: AgentState) -> AgentState:
    """
    Node 3: Block destructive SQL operations.
    Templates are trusted but still validated as a safety net.
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
    Node 5: Llama generates the final customer-facing response.
    """
    intent      = state["intent"]
    intent_desc = get_intent_description(intent)
    sql_result  = state.get("sql_result")
    sql_error   = state.get("sql_error")

    if sql_result:
        prompt = f"""You are a customer service assistant. Answer the customer's question using the data below.
Keep your response to 2-3 sentences maximum. Do not repeat instructions. Do not add examples.
All prices are in SGD (Singapore Dollars, $).

Customer question: {state["user_input"]}
Data: {sql_result}

Your response (2-3 sentences only):"""

    elif sql_error:
        prompt = f"""You are a customer service assistant. A system error occurred.
Apologise in 1-2 sentences and offer to escalate to a human agent.

Customer question: {state["user_input"]}

Your response (1-2 sentences only):"""

    elif intent in ("non_english", "gibberish"):
        prompt = f"""You are a customer service assistant.
Tell the customer in 1 sentence that you only support English.

Your response (1 sentence only):"""

    else:
        prompt = f"""You are a customer service assistant helping with {intent_desc}.
Reply helpfully in 2-3 sentences. Be friendly and concise.

Customer message: {state["user_input"]}

Your response (2-3 sentences only):"""

    response = llm.invoke(prompt)
    text     = response.content if hasattr(response, "content") else str(response)
    text     = clean_response(text)     # strip prompt leaking / repetition

    print(f"[Llama Response] {text[:120]}...")

    return {
        **state,
        "final_response": text,
        "messages":       state["messages"] + [AIMessage(content=text)]
    }


def node_handle_error(state: AgentState) -> AgentState:
    """
    Node 6: Fallback for invalid input or low-confidence classification.
    Three distinct messages depending on the failure reason.
    """
    if state["intent"] == "non_english":
        # Non-ASCII script (Chinese characters, Arabic, etc.)
        response = (
            "I'm sorry, I currently only support English. "
            "Please rephrase your question in English and I'll be happy to help!"
        )
    elif state["intent"] == "gibberish":
        # ASCII but not meaningful English (pinyin, random characters, etc.)
        response = (
            "I'm sorry, I didn't recognise that as a customer service question. "
            "Could you rephrase it in English? I can help with orders, refunds, "
            "account settings, payments, and delivery queries."
        )
    else:
        # Low DeBERTa confidence — valid English but unclear intent
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
    Five routes:
      - Non-English script  → handle_error  (non-ASCII characters)
      - Gibberish/pinyin    → handle_error  (not meaningful English)
      - Low confidence      → handle_error  (DeBERTa unsure)
      - SQL intent          → generate_sql  (template or Llama)
      - Chat intent         → generate_response
    """
    if state["intent"] in ("non_english", "gibberish"):
        return "handle_error"
    if state["confidence"] < 0.5:
        return "handle_error"
    if state["needs_sql"]:
        return "generate_sql"
    return "generate_response"


def route_after_execute(state: AgentState) -> str:
    """
    After SQL execution:
      - Template queries do NOT retry (they are correct by design)
      - Llama-generated queries retry up to 2 times on error
    """
    if state.get("sql_error"):
        if state.get("used_template"):
            # Template failed — something wrong with DB itself, go straight to response
            print("[Router] Template query failed — skipping retry, going to response.")
            return "generate_response"
        if state.get("retry_count", 0) < 2:
            # Llama-generated query failed — retry
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
        "used_template":  False,
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
    # llm = load_llama_ollama()           # OPTION A: Ollama (recommended)
    llm = load_llama_huggingface()    # OPTION B: HuggingFace (needs GPU)
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
