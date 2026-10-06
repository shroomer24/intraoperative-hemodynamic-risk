"""Frozen v0.1 constants from the supplied scientific protocol."""

PROTOCOL_VERSION = "vitaldb-hypotension-v0.1"
TRACKS = {
    "map": "Solar8000/ART_MBP",
    "sbp": "Solar8000/ART_SBP",
    "dbp": "Solar8000/ART_DBP",
    "hr": "Solar8000/HR",
    "spo2": "Solar8000/PLETH_SPO2",
    "etco2": "Solar8000/ETCO2",
}
HISTORY_SECONDS = 300
ANCHOR_STRIDE_SECONDS = 60
HORIZON_SECONDS = 300
CONFIRMATION_SECONDS = 60
FUTURE_SECONDS = 360
MAX_AGE_SECONDS = 10
MAP_THRESHOLD = 65
MAP_TECHNICAL_CEILING = 250
MIN_HISTORY_COVERAGE = 0.90
MIN_SLOPE_OBSERVATIONS = 2
