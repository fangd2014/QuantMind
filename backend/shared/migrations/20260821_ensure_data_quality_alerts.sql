CREATE TABLE IF NOT EXISTS data_quality_alerts (
    id              BIGSERIAL PRIMARY KEY,
    alert_type      VARCHAR(32) NOT NULL,
    severity        VARCHAR(16) NOT NULL,
    market          VARCHAR(8),
    field           VARCHAR(48),
    source          VARCHAR(32),
    symbol          VARCHAR(32),
    trade_date      DATE,
    message         TEXT NOT NULL,
    details         JSONB,
    acknowledged    BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_by VARCHAR(64),
    acknowledged_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dqa_created_at
    ON data_quality_alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dqa_unack_severity
    ON data_quality_alerts (acknowledged, severity, created_at DESC)
    WHERE acknowledged = FALSE;
CREATE INDEX IF NOT EXISTS idx_dqa_market_field
    ON data_quality_alerts (market, field, created_at DESC);
