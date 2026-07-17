"""
Generate dataset/sample/sample.csv — a SYNTHETIC, clearly-labeled sample dataset
in the exact CICIoT2023 feature schema (common/schema.py).

WHY THIS EXISTS
---------------
CICIoT2023 is ~13 GB. This bundled sample guarantees the full pipeline
(preprocess -> train -> quantize -> benchmarks -> demo) runs end-to-end on a
clean machine with zero downloads. It is statistically realistic (class-
conditional distributions grounded in how each attack behaves on the wire) but
it is SYNTHETIC and is marked as such in common/artifacts/provenance.json,
the /model-info endpoint, and the dashboard badge.

All metrics shown anywhere are still REAL MEASUREMENTS of the model trained on
whatever data preprocess.py used — never hardcoded. To get numbers on the real
dataset, run ml-engine/scripts/download_dataset.py then preprocess.py without
--sample.

Usage: python ml-engine/scripts/make_sample.py [--rows-per-class 2000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.schema import CLASSES, FEATURE_COLUMNS, GLOBAL_SEED, LABEL_COLUMN, SAMPLE_CSV


def _protocol_block(rng, n, http=0.0, https=0.0, dns=0.0, telnet=0.0, smtp=0.0,
                    ssh=0.0, irc=0.0, tcp=0.9, udp=0.1, dhcp=0.0, arp=0.0,
                    icmp=0.0, ipv=1.0, llc=0.0):
    """Bernoulli protocol indicator columns."""
    def b(p):
        return (rng.random(n) < p).astype(float)
    return {
        "HTTP": b(http), "HTTPS": b(https), "DNS": b(dns), "Telnet": b(telnet),
        "SMTP": b(smtp), "SSH": b(ssh), "IRC": b(irc), "TCP": b(tcp),
        "UDP": b(udp), "DHCP": b(dhcp), "ARP": b(arp), "ICMP": b(icmp),
        "IPv": b(ipv), "LLC": b(llc),
    }


def _flags(rng, n, fin=0.05, syn=0.1, rst=0.02, psh=0.3, ack=0.7, ece=0.0, cwr=0.0):
    def b(p):
        return (rng.random(n) < p).astype(float)
    return {
        "fin_flag_number": b(fin), "syn_flag_number": b(syn),
        "rst_flag_number": b(rst), "psh_flag_number": b(psh),
        "ack_flag_number": b(ack), "ece_flag_number": b(ece),
        "cwr_flag_number": b(cwr),
    }


def _counts(rng, n, ack=8, syn=1, fin=1, urg=0, rst=0):
    def pois(lam):
        return rng.poisson(lam, n).astype(float)
    return {
        "ack_count": pois(ack), "syn_count": pois(syn), "fin_count": pois(fin),
        "urg_count": pois(urg), "rst_count": pois(rst),
    }


def _sizes(rng, n, avg_mean, avg_sigma, std_scale, number_lam):
    """Packet-size aggregate columns, kept internally consistent the way
    CICIoT2023's flow aggregator derives them (Magnitude, Radius, etc. are
    functions of the running packet-size statistics)."""
    avg = rng.lognormal(np.log(avg_mean), avg_sigma, n)
    std = np.abs(rng.normal(0, std_scale, n)) * avg * 0.35
    number = np.maximum(1, rng.poisson(number_lam, n)).astype(float)
    mn = np.maximum(20.0, avg - 1.6 * std)
    mx = avg + 1.9 * std + rng.exponential(15, n)
    tot_sum = avg * number * rng.uniform(0.92, 1.08, n)
    tot_size = avg * rng.uniform(0.95, 1.05, n)
    variance = std ** 2
    return {
        "Tot sum": tot_sum, "Min": mn, "Max": mx, "AVG": avg, "Std": std,
        "Tot size": tot_size, "Number": number,
        "Magnitue": np.sqrt(2.0 * avg) * rng.uniform(0.97, 1.03, n),
        "Radius": np.sqrt(2.0 * variance) * rng.uniform(0.97, 1.03, n),
        "Covariance": variance * rng.uniform(0.8, 1.2, n),
        "Variance": np.clip(variance / (avg ** 2 + 1e-6), 0, 1),
        "Weight": number * rng.uniform(38.0, 42.0, n),
    }


def gen_class(rng, label, n):
    """Class-conditional flow generator. Parameters follow the on-the-wire
    behaviour of each traffic class (see docs/implementation.md)."""
    if label == "Normal":
        d = {
            "flow_duration": rng.exponential(8.0, n) + 0.05,
            "Header_Length": rng.normal(20000, 6000, n).clip(64, None),
            "Protocol Type": rng.choice([6.0, 17.0], n, p=[0.8, 0.2]),
            "Duration": rng.uniform(50, 120, n),
            "Rate": rng.lognormal(np.log(45), 0.8, n),
            "Srate": rng.lognormal(np.log(40), 0.8, n),
            "Drate": rng.lognormal(np.log(6), 1.0, n),
            "IAT": rng.lognormal(np.log(0.08), 0.9, n),
        }
        d.update(_flags(rng, n, fin=0.06, syn=0.08, rst=0.01, psh=0.35, ack=0.75))
        d.update(_counts(rng, n, ack=9, syn=1, fin=1, urg=0, rst=0))
        d.update(_protocol_block(rng, n, http=0.25, https=0.45, dns=0.15, tcp=0.82, udp=0.18))
        d.update(_sizes(rng, n, avg_mean=420, avg_sigma=0.55, std_scale=1.0, number_lam=9))
    elif label == "DDoS":
        d = {
            "flow_duration": rng.exponential(0.6, n) + 0.001,
            "Header_Length": rng.normal(60000, 18000, n).clip(64, None),
            "Protocol Type": rng.choice([6.0, 17.0, 1.0], n, p=[0.55, 0.3, 0.15]),
            "Duration": rng.uniform(40, 70, n),
            "Rate": rng.lognormal(np.log(4200), 0.5, n),
            "Srate": rng.lognormal(np.log(4100), 0.5, n),
            "Drate": rng.lognormal(np.log(1.5), 1.0, n),
            "IAT": rng.lognormal(np.log(0.0004), 0.7, n),
        }
        d.update(_flags(rng, n, fin=0.02, syn=0.85, rst=0.08, psh=0.05, ack=0.15))
        d.update(_counts(rng, n, ack=1, syn=14, fin=0, urg=0, rst=1))
        d.update(_protocol_block(rng, n, tcp=0.6, udp=0.3, icmp=0.15))
        d.update(_sizes(rng, n, avg_mean=90, avg_sigma=0.25, std_scale=0.3, number_lam=60))
    elif label == "Brute Force":
        d = {
            "flow_duration": rng.exponential(2.5, n) + 0.05,
            "Header_Length": rng.normal(9000, 2500, n).clip(64, None),
            "Protocol Type": np.full(n, 6.0),
            "Duration": rng.uniform(50, 100, n),
            "Rate": rng.lognormal(np.log(160), 0.5, n),
            "Srate": rng.lognormal(np.log(150), 0.5, n),
            "Drate": rng.lognormal(np.log(20), 0.8, n),
            "IAT": rng.lognormal(np.log(0.02), 0.7, n),
        }
        d.update(_flags(rng, n, fin=0.25, syn=0.35, rst=0.20, psh=0.55, ack=0.65))
        d.update(_counts(rng, n, ack=6, syn=4, fin=3, urg=0, rst=2))
        d.update(_protocol_block(rng, n, ssh=0.55, telnet=0.35, tcp=1.0, udp=0.0))
        d.update(_sizes(rng, n, avg_mean=150, avg_sigma=0.35, std_scale=0.5, number_lam=14))
    elif label == "Spoofing":
        d = {
            "flow_duration": rng.exponential(5.0, n) + 0.05,
            "Header_Length": rng.normal(3500, 1200, n).clip(28, None),
            "Protocol Type": rng.choice([1.0, 6.0, 17.0], n, p=[0.25, 0.35, 0.4]),
            "Duration": rng.uniform(30, 90, n),
            "Rate": rng.lognormal(np.log(22), 0.7, n),
            "Srate": rng.lognormal(np.log(20), 0.7, n),
            "Drate": rng.lognormal(np.log(15), 0.8, n),
            "IAT": rng.lognormal(np.log(0.15), 0.8, n),
        }
        d.update(_flags(rng, n, fin=0.03, syn=0.05, rst=0.02, psh=0.15, ack=0.35))
        d.update(_counts(rng, n, ack=2, syn=0, fin=0, urg=0, rst=0))
        d.update(_protocol_block(rng, n, dns=0.35, arp=0.55, tcp=0.3, udp=0.4, icmp=0.2, llc=0.3))
        d.update(_sizes(rng, n, avg_mean=110, avg_sigma=0.3, std_scale=0.4, number_lam=6))
    elif label == "Scanning":
        d = {
            "flow_duration": rng.exponential(0.15, n) + 0.001,
            "Header_Length": rng.normal(2500, 800, n).clip(40, None),
            "Protocol Type": np.full(n, 6.0),
            "Duration": rng.uniform(20, 60, n),
            "Rate": rng.lognormal(np.log(320), 0.6, n),
            "Srate": rng.lognormal(np.log(300), 0.6, n),
            "Drate": rng.lognormal(np.log(3), 1.0, n),
            "IAT": rng.lognormal(np.log(0.003), 0.8, n),
        }
        d.update(_flags(rng, n, fin=0.30, syn=0.70, rst=0.45, psh=0.02, ack=0.10))
        d.update(_counts(rng, n, ack=1, syn=9, fin=4, urg=1, rst=7))
        d.update(_protocol_block(rng, n, tcp=0.92, udp=0.08, icmp=0.1))
        d.update(_sizes(rng, n, avg_mean=52, avg_sigma=0.15, std_scale=0.15, number_lam=30))
    elif label == "Malware":
        d = {
            "flow_duration": rng.exponential(12.0, n) + 0.1,
            "Header_Length": rng.normal(14000, 5000, n).clip(64, None),
            "Protocol Type": rng.choice([6.0, 17.0], n, p=[0.7, 0.3]),
            "Duration": rng.uniform(40, 110, n),
            "Rate": rng.lognormal(np.log(85), 0.9, n),
            "Srate": rng.lognormal(np.log(70), 0.9, n),
            "Drate": rng.lognormal(np.log(30), 0.9, n),
            # beaconing: periodic IAT, low jitter
            "IAT": np.abs(rng.normal(1.0, 0.08, n)) * rng.choice([0.5, 1.0, 2.0], n),
        }
        d.update(_flags(rng, n, fin=0.10, syn=0.20, rst=0.05, psh=0.45, ack=0.60, ece=0.02, cwr=0.02))
        d.update(_counts(rng, n, ack=7, syn=2, fin=1, urg=2, rst=1))
        d.update(_protocol_block(rng, n, http=0.35, dns=0.40, irc=0.25, tcp=0.75, udp=0.25))
        d.update(_sizes(rng, n, avg_mean=260, avg_sigma=0.7, std_scale=0.9, number_lam=12))
    else:
        raise ValueError(label)

    df = pd.DataFrame({c: d[c] for c in FEATURE_COLUMNS})
    df[LABEL_COLUMN] = label
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-class", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    frames = [gen_class(rng, c, args.rows_per_class) for c in CLASSES]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_CSV, index=False)
    print(f"[make_sample] wrote {len(df):,} rows x {len(df.columns)} cols -> {SAMPLE_CSV}")
    print(f"[make_sample] class counts:\n{df[LABEL_COLUMN].value_counts().to_string()}")
    print("[make_sample] NOTE: this is a SYNTHETIC sample for pipeline bring-up; "
          "run download_dataset.py for the real CICIoT2023.")


if __name__ == "__main__":
    main()
