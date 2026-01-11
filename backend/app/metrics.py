from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "icu_requests_total",
    "Total number of inference requests",
    ["endpoint", "method"]
)

REQUEST_LATENCY = Histogram(
    "icu_requests_latency_seconds",
    "Request Latency",
    ["endpoint"]
)

ERROR_COUNT  = Counter(
    "icu_errors_total",
    "Total number of errors",
    ["endpoint"]
)

RISK_SCORE_DIST = Histogram(
    "icu_risk_score",
    "Predicted risk score distribution",
    buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
)