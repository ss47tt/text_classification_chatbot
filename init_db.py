"""
init_db.py
Creates and seeds customer_service.db — the SQLite database the
LangGraph SQL nodes in agent_bitext.py (generate_sql / execute_sql)
query against.

Run this ONCE from the same folder as agent_bitext.py:
    python init_db.py

It's safe to re-run: it drops and recreates every table each time,
so you always get a clean, known dataset to test against.

Schema covers every Bitext SQL_INTENT from agent_bitext.py:
    track_order            -> orders, order_items
    check_invoice/get_invoice -> invoices
    check_refund_policy/track_refund -> refunds, refund_policy
    check_payment_methods   -> payments, payment_methods
    check_cancellation_fee  -> cancellation_policy
    delivery_options/delivery_period -> deliveries, delivery_options
    review                  -> reviews
"""

import sqlite3
from datetime import date, timedelta

DB_PATH = "customer_service.db"

SCHEMA = """
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS payment_methods;
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS refund_policy;
DROP TABLE IF EXISTS deliveries;
DROP TABLE IF EXISTS delivery_options;
DROP TABLE IF EXISTS cancellation_policy;
DROP TABLE IF EXISTS reviews;

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    full_name     TEXT NOT NULL,
    email         TEXT NOT NULL
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER NOT NULL,
    order_date    TEXT NOT NULL,
    status        TEXT NOT NULL,          -- placed, shipped, delivered, cancelled
    total_amount  REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    item_id       INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    product_name  TEXT NOT NULL,
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE invoices (
    invoice_id    INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    issued_date   TEXT NOT NULL,
    amount        REAL NOT NULL,
    status        TEXT NOT NULL,          -- paid, unpaid, overdue
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE payment_methods (
    method_id     INTEGER PRIMARY KEY,
    method_name   TEXT NOT NULL,          -- Visa, Mastercard, PayPal, Bank Transfer
    is_active     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE payments (
    payment_id    INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    method_id     INTEGER NOT NULL,
    amount        REAL NOT NULL,
    status        TEXT NOT NULL,          -- success, failed, pending
    payment_date  TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (method_id) REFERENCES payment_methods(method_id)
);

CREATE TABLE refunds (
    refund_id       INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL,
    amount          REAL NOT NULL,
    status          TEXT NOT NULL,        -- requested, processing, completed, rejected
    requested_date  TEXT NOT NULL,
    processed_date  TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE refund_policy (
    policy_id     INTEGER PRIMARY KEY,
    category      TEXT NOT NULL,
    window_days   INTEGER NOT NULL,
    description   TEXT NOT NULL
);

CREATE TABLE deliveries (
    delivery_id         INTEGER PRIMARY KEY,
    order_id             INTEGER NOT NULL,
    carrier               TEXT NOT NULL,
    status                 TEXT NOT NULL,   -- pending, in_transit, delivered
    estimated_delivery   TEXT NOT NULL,
    actual_delivery       TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE delivery_options (
    option_id     INTEGER PRIMARY KEY,
    region        TEXT NOT NULL,
    method        TEXT NOT NULL,          -- standard, express, next_day
    cost          REAL NOT NULL,
    estimated_days TEXT NOT NULL
);

CREATE TABLE cancellation_policy (
    policy_id     INTEGER PRIMARY KEY,
    order_status  TEXT NOT NULL,          -- placed, shipped
    fee_percent   REAL NOT NULL,
    description   TEXT NOT NULL
);

CREATE TABLE reviews (
    review_id     INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL,
    customer_id   INTEGER NOT NULL,
    rating        INTEGER NOT NULL,       -- 1-5
    comment       TEXT,
    review_date   TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""


def seed(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    today = date.today()

    cur.executemany(
        "INSERT INTO customers VALUES (?,?,?)",
        [
            (1, "Alice Tan",   "alice.tan@example.com"),
            (2, "Marcus Lee",  "marcus.lee@example.com"),
            (3, "Priya Nair",  "priya.nair@example.com"),
        ],
    )

    cur.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?)",
        [
            (101, 1, str(today - timedelta(days=10)), "delivered", 89.99),
            (102, 1, str(today - timedelta(days=2)),  "shipped",   45.50),
            (103, 2, str(today - timedelta(days=20)), "cancelled", 120.00),
            (104, 3, str(today - timedelta(days=1)),  "placed",    32.75),
        ],
    )

    cur.executemany(
        "INSERT INTO order_items VALUES (?,?,?,?,?)",
        [
            (1, 101, "Wireless Mouse", 1, 29.99),
            (2, 101, "USB-C Hub",      1, 60.00),
            (3, 102, "Phone Case",     1, 45.50),
            (4, 103, "Desk Lamp",      2, 60.00),
            (5, 104, "Notebook Set",   1, 32.75),
        ],
    )

    cur.executemany(
        "INSERT INTO invoices VALUES (?,?,?,?,?)",
        [
            (1001, 101, str(today - timedelta(days=10)), 89.99,  "paid"),
            (1002, 102, str(today - timedelta(days=2)),  45.50,  "paid"),
            (1003, 103, str(today - timedelta(days=20)), 120.00, "unpaid"),
            (1004, 104, str(today - timedelta(days=1)),  32.75,  "paid"),
        ],
    )

    cur.executemany(
        "INSERT INTO payment_methods VALUES (?,?,?)",
        [
            (1, "Visa",           1),
            (2, "Mastercard",     1),
            (3, "PayPal",         1),
            (4, "Bank Transfer",  1),
            (5, "Amex",           0),
        ],
    )

    cur.executemany(
        "INSERT INTO payments VALUES (?,?,?,?,?,?)",
        [
            (1, 101, 1, 89.99,  "success", str(today - timedelta(days=10))),
            (2, 102, 3, 45.50,  "success", str(today - timedelta(days=2))),
            (3, 103, 2, 120.00, "failed",  str(today - timedelta(days=20))),
            (4, 104, 1, 32.75,  "pending", str(today - timedelta(days=1))),
        ],
    )

    cur.executemany(
        "INSERT INTO refunds VALUES (?,?,?,?,?,?)",
        [
            (1, 103, 120.00, "completed", str(today - timedelta(days=18)), str(today - timedelta(days=15))),
        ],
    )

    cur.executemany(
        "INSERT INTO refund_policy VALUES (?,?,?,?)",
        [
            (1, "standard", 30, "Full refund within 30 days of delivery, item unused."),
            (2, "electronics", 14, "Refund within 14 days, original packaging required."),
            (3, "final_sale", 0, "Final sale items are not eligible for refund."),
        ],
    )

    cur.executemany(
        "INSERT INTO deliveries VALUES (?,?,?,?,?,?)",
        [
            (1, 101, "DHL",       "delivered",  str(today - timedelta(days=8)), str(today - timedelta(days=7))),
            (2, 102, "FedEx",     "in_transit", str(today + timedelta(days=1)), None),
            (3, 103, "UPS",       "delivered",  str(today - timedelta(days=17)), str(today - timedelta(days=16))),
            (4, 104, "SingPost",  "pending",    str(today + timedelta(days=3)), None),
        ],
    )

    cur.executemany(
        "INSERT INTO delivery_options VALUES (?,?,?,?,?)",
        [
            (1, "Singapore",      "standard",  3.99,  "3-5 business days"),
            (2, "Singapore",      "express",   8.99,  "1-2 business days"),
            (3, "Singapore",      "next_day",  14.99, "1 business day"),
            (4, "International",  "standard",  19.99, "7-14 business days"),
        ],
    )

    cur.executemany(
        "INSERT INTO cancellation_policy VALUES (?,?,?,?)",
        [
            (1, "placed",  0.0,  "Free cancellation before the order ships."),
            (2, "shipped", 10.0, "10% restocking fee once the order has shipped."),
        ],
    )

    cur.executemany(
        "INSERT INTO reviews VALUES (?,?,?,?,?,?)",
        [
            (1, 101, 1, 5, "Great product, fast delivery!", str(today - timedelta(days=5))),
            (2, 103, 2, 2, "Order was cancelled, refund took a while.", str(today - timedelta(days=14))),
        ],
    )

    conn.commit()


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    seed(conn)

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Created {DB_PATH} with tables: {tables}")

    conn.close()


if __name__ == "__main__":
    main()
