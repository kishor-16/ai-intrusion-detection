"""
EdgeShield AI — single source of truth for the feature schema.

BOTH the training pipeline (ml-engine/) and the IoT simulator (iot-simulator/)
import this module. The model is trained on exactly FEATURE_COLUMNS in exactly
this order; the simulator emits vectors in exactly this order. Never edit one
side without the other — this file IS the contract (see docs/architecture.md).

Feature names match the published CICIoT2023 CSV columns so the real dataset
drops in with zero code changes.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GLOBAL_SEED = 42

# 46 features, exact CICIoT2023 column order.
FEATURE_COLUMNS = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration",
    "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
    "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

LABEL_COLUMN = "label"

# Binary/categorical indicator columns — never jittered by the simulator and
# never log-transformed by preprocessing. Everything else is a heavy-tailed
# continuous feature that gets signed-log1p compression before scaling.
INDICATOR_COLUMNS = {
    "Protocol Type", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IPv", "LLC",
}

# The 6 EdgeShield classes.
CLASSES = ["Normal", "DDoS", "Brute Force", "Spoofing", "Scanning", "Malware"]

# CICIoT2023 raw label -> EdgeShield class (used by preprocess.py on real data).
RAW_LABEL_MAP = {
    "BenignTraffic": "Normal",
    # --- DDoS / DoS floods ---
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS", "DDoS-SlowLoris": "DDoS",
    "DoS-UDP_Flood": "DDoS", "DoS-SYN_Flood": "DDoS", "DoS-TCP_Flood": "DDoS",
    "DoS-HTTP_Flood": "DDoS",
    # --- Brute force ---
    "DictionaryBruteForce": "Brute Force",
    # --- Spoofing / MITM ---
    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    # --- Reconnaissance / scanning ---
    "Recon-PingSweep": "Scanning", "Recon-OSScan": "Scanning",
    "Recon-PortScan": "Scanning", "Recon-HostDiscovery": "Scanning",
    "VulnerabilityScan": "Scanning",
    # --- Malware / botnet / web exploitation ---
    "Mirai-greeth_flood": "Malware", "Mirai-greip_flood": "Malware",
    "Mirai-udpplain": "Malware", "Backdoor_Malware": "Malware",
    "CommandInjection": "Malware", "SqlInjection": "Malware",
    "XSS": "Malware", "Uploading_Attack": "Malware",
    "Browser_Hijacking": "Malware",
}

# Shared artifact locations (all relative to repo root).
ARTIFACT_DIR = REPO_ROOT / "common" / "artifacts"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
TRANSFORM_PATH = ARTIFACT_DIR / "transform.json"  # log1p columns + post-scale clip
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.json"
PROVENANCE_PATH = ARTIFACT_DIR / "provenance.json"

DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_CSV = DATASET_DIR / "sample" / "sample.csv"
SUBSET_DIR = DATASET_DIR / "subset"
PROCESSED_DIR = DATASET_DIR / "processed"

MODEL_DIR = REPO_ROOT / "models"
FP32_ONNX = MODEL_DIR / "model_fp32.onnx"
INT8_ONNX = MODEL_DIR / "model_int8.onnx"
INT8_META = MODEL_DIR / "int8_meta.json"  # quantization method + input clip value
BEST_MODEL_PT = MODEL_DIR / "best_model.pt"
BASELINE_PT = MODEL_DIR / "baseline_model.pt"
BASELINE_ONNX = MODEL_DIR / "baseline_fp32.onnx"
FL_MODEL_PT = MODEL_DIR / "fl_global_model.pt"
METRICS_JSON = MODEL_DIR / "metrics.json"

BENCH_DIR = REPO_ROOT / "benchmarks"
BENCH_RESULTS = BENCH_DIR / "results.json"

FL_DIR = REPO_ROOT / "federated-learning"
FL_HISTORY = FL_DIR / "history.json"

EXPLAIN_DIR = REPO_ROOT / "ml-inference-service" / "explain"
SHAP_SUMMARY = EXPLAIN_DIR / "shap_summary.json"

N_FEATURES = len(FEATURE_COLUMNS)
N_CLASSES = len(CLASSES)
