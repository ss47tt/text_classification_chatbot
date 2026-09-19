"""
test_db.py
Sanity-checks customer_service.db directly, without loading DeBERTa
or Llama. Useful for quickly verifying the schema and seed data are
correct before testing the full agent end-to-end.

Run from the same folder as customer_service.db:
    python test_db.py
"""

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///customer_service.db")

print("── Tables ──")
print(db.get_usable_table_names())

print("\n── Full schema (what gets passed to Llama for SQL generation) ──")
print(db.get_table_info())

# One representative query per SQL_INTENT, run directly (bypassing the
# LLM) so you can confirm the underlying data is queryable and correct.
SAMPLE_QUERIES = {
    "track_order":            "SELECT * FROM orders WHERE customer_id = 1;",
    "check_invoice":          "SELECT * FROM invoices WHERE order_id = 101;",
    "get_invoice":            "SELECT * FROM invoices WHERE order_id = 101;",
    "check_refund_policy":    "SELECT * FROM refund_policy;",
    "track_refund":           "SELECT * FROM refunds WHERE order_id = 103;",
    "check_payment_methods":  "SELECT method_name FROM payment_methods WHERE is_active = 1;",
    "check_cancellation_fee": "SELECT * FROM cancellation_policy;",
    "delivery_options":       "SELECT * FROM delivery_options WHERE region = 'Singapore';",
    "delivery_period":        "SELECT estimated_delivery, actual_delivery FROM deliveries WHERE order_id = 102;",
    "review":                 "SELECT * FROM reviews WHERE customer_id = 1;",
}

print("\n── Sample query per SQL intent ──")
for intent, query in SAMPLE_QUERIES.items():
    print(f"\n[{intent}]")
    print(f"  SQL: {query}")
    try:
        result = db.run(query)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  ERROR: {e}")
