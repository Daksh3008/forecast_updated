#  db/schema.py
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_prices (
    asset      TEXT NOT NULL,
    date       DATE NOT NULL,
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION,
    volume     DOUBLE PRECISION,
    PRIMARY KEY (asset, date)
);

CREATE INDEX IF NOT EXISTS idx_daily_prices_date
ON daily_prices (date);
"""
