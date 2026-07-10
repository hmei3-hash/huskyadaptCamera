"""
Serial Latency Collector & Analyzer
====================================
Collects [LATENCY] lines from ESP32 serial, computes stats,
and prints a comparison table for your report.

Usage:
  1. Flash baseline, run:
     python latency_test.py --port COM3 --label baseline --samples 15

  2. Flash full version, run:
     python latency_test.py --port COM3 --label full --samples 15

  3. Compare results:
     python latency_test.py --compare

Data is saved to latency_baseline.csv / latency_full.csv
"""

import argparse
import re
import sys
import os
import csv
import time

def parse_latency_line(line):
    """Parse a [LATENCY] line and return dict of values."""
    result = {}

    m = re.search(r'dist=(\d+)cm', line)
    if m: result['dist_cm'] = int(m.group(1))

    m = re.search(r'proc=(\d+)us', line)
    if m: result['proc_us'] = int(m.group(1))

    m = re.search(r'e2e=(\d+)us', line)
    if m: result['e2e_us'] = int(m.group(1))

    m = re.search(r'mode=(\w)', line)
    if m: result['mode'] = m.group(1)

    m = re.search(r'agree=(\d+)', line)
    if m: result['agree'] = int(m.group(1))

    return result if 'proc_us' in result else None


def collect(port, baud, label, num_samples):
    """Collect latency samples from serial port."""
    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed.")
        print("  pip install pyserial")
        sys.exit(1)

    filename = f"latency_{label}.csv"
    samples = []

    print(f"Connecting to {port} @ {baud}...")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)  # wait for ESP32 reset
    ser.flushInput()

    print(f"Collecting {num_samples} trigger events for '{label}'...")
    print("Move your hand in and out of the sensor to generate triggers.\n")

    while len(samples) < num_samples:
        raw = ser.readline()
        try:
            line = raw.decode('utf-8', errors='ignore').strip()
        except:
            continue

        if not line:
            continue

        # Print all serial output so user can see what's happening
        print(f"  {line}")

        if '[LATENCY]' in line:
            parsed = parse_latency_line(line)
            if parsed:
                samples.append(parsed)
                n = len(samples)
                print(f"  >>> Sample {n}/{num_samples}: "
                      f"proc={parsed['proc_us']}us  "
                      f"e2e={parsed['e2e_us']}us  "
                      f"dist={parsed['dist_cm']}cm")

    ser.close()

    # Save to CSV
    fieldnames = ['sample', 'dist_cm', 'proc_us', 'e2e_us', 'mode', 'agree']
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, s in enumerate(samples, 1):
            row = {'sample': i}
            row.update(s)
            w.writerow(row)

    print(f"\nSaved {len(samples)} samples to {filename}")
    print_stats(label, samples)


def print_stats(label, samples):
    """Print statistics for a set of samples."""
    proc_vals = sorted([s['proc_us'] for s in samples])
    e2e_vals  = sorted([s['e2e_us'] for s in samples])

    def median(vals):
        n = len(vals)
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    def mean(vals):
        return sum(vals) / len(vals)

    print(f"\n{'=' * 50}")
    print(f"  Stats: {label}  ({len(samples)} samples)")
    print(f"{'=' * 50}")
    print(f"  Processing latency (echo → trigger):")
    print(f"    Min:    {min(proc_vals):>8} us")
    print(f"    Max:    {max(proc_vals):>8} us")
    print(f"    Mean:   {mean(proc_vals):>8.1f} us")
    print(f"    Median: {median(proc_vals):>8.1f} us")
    print(f"")
    print(f"  End-to-end latency (ping → trigger):")
    print(f"    Min:    {min(e2e_vals):>8} us")
    print(f"    Max:    {max(e2e_vals):>8} us")
    print(f"    Mean:   {mean(e2e_vals):>8.1f} us")
    print(f"    Median: {median(e2e_vals):>8.1f} us")
    print(f"{'=' * 50}\n")


def load_csv(filename):
    """Load samples from CSV file."""
    samples = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'dist_cm': int(row['dist_cm']),
                'proc_us': int(row['proc_us']),
                'e2e_us':  int(row['e2e_us']),
                'mode':    row.get('mode', '-'),
                'agree':   int(row['agree']) if row.get('agree') else 0,
            })
    return samples


def compare():
    """Load both CSVs and print comparison table."""
    base_file = "latency_baseline.csv"
    full_file = "latency_full.csv"

    if not os.path.exists(base_file):
        print(f"ERROR: {base_file} not found. Run with --label baseline first.")
        sys.exit(1)
    if not os.path.exists(full_file):
        print(f"ERROR: {full_file} not found. Run with --label full first.")
        sys.exit(1)

    base = load_csv(base_file)
    full = load_csv(full_file)

    def median(vals):
        vals = sorted(vals)
        n = len(vals)
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    def mean(vals):
        return sum(vals) / len(vals)

    bp = [s['proc_us'] for s in base]
    be = [s['e2e_us']  for s in base]
    fp = [s['proc_us'] for s in full]
    fe = [s['e2e_us']  for s in full]

    print()
    print("=" * 65)
    print("  LATENCY COMPARISON: Baseline vs IMU Fusion (Full)")
    print("=" * 65)
    print()
    print(f"  {'Metric':<30} {'Baseline':>12} {'Full':>12} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Proc latency - median (us)':<30} {median(bp):>12.1f} {median(fp):>12.1f} {median(fp)-median(bp):>+10.1f}")
    print(f"  {'Proc latency - mean (us)':<30} {mean(bp):>12.1f} {mean(fp):>12.1f} {mean(fp)-mean(bp):>+10.1f}")
    print(f"  {'Proc latency - min (us)':<30} {min(bp):>12} {min(fp):>12} {min(fp)-min(bp):>+10}")
    print(f"  {'Proc latency - max (us)':<30} {max(bp):>12} {max(fp):>12} {max(fp)-max(bp):>+10}")
    print()
    print(f"  {'E2E latency - median (us)':<30} {median(be):>12.1f} {median(fe):>12.1f} {median(fe)-median(be):>+10.1f}")
    print(f"  {'E2E latency - mean (us)':<30} {mean(be):>12.1f} {mean(fe):>12.1f} {mean(fe)-mean(be):>+10.1f}")
    print(f"  {'E2E latency - min (us)':<30} {min(be):>12} {min(fe):>12} {min(fe)-min(be):>+10}")
    print(f"  {'E2E latency - max (us)':<30} {max(be):>12} {max(fe):>12} {max(fe)-max(be):>+10}")
    print()
    print(f"  {'Samples':<30} {len(base):>12} {len(full):>12}")
    print("=" * 65)

    # Note about state confirmation overhead
    print()
    print("  NOTE: Full version requires STATE_CONFIRM_N (3) consecutive")
    print("  readings before triggering. At PING_INTERVAL_MS=40ms, this")
    print("  adds ~80-120ms of inherent detection delay that is NOT")
    print("  captured by e2e (which only times the final ping cycle).")
    print("  True user-perceived delay = e2e + (STATE_CONFIRM_N - 1) * 40ms")
    print()

    # Save comparison to text file
    # (user can copy into report)
    with open("latency_comparison.txt", "w") as f:
        f.write("| Metric | Baseline | Full | Delta |\n")
        f.write("|--------|----------|------|-------|\n")
        f.write(f"| Proc median (us) | {median(bp):.1f} | {median(fp):.1f} | {median(fp)-median(bp):+.1f} |\n")
        f.write(f"| Proc mean (us) | {mean(bp):.1f} | {mean(fp):.1f} | {mean(fp)-mean(bp):+.1f} |\n")
        f.write(f"| E2E median (us) | {median(be):.1f} | {median(fe):.1f} | {median(fe)-median(be):+.1f} |\n")
        f.write(f"| E2E mean (us) | {mean(be):.1f} | {mean(fe):.1f} | {mean(fe)-mean(be):+.1f} |\n")
        f.write(f"| Samples | {len(base)} | {len(full)} | |\n")
    print("  Markdown table saved to latency_comparison.txt")


def main():
    p = argparse.ArgumentParser(description="ESP32 Latency Tester")
    p.add_argument("--port", type=str, default=None,
                   help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--label", type=str, default="baseline",
                   help="Test label: 'baseline' or 'full'")
    p.add_argument("--samples", type=int, default=15,
                   help="Number of trigger events to collect")
    p.add_argument("--compare", action="store_true",
                   help="Compare baseline vs full from saved CSVs")
    args = p.parse_args()

    if args.compare:
        compare()
    else:
        if not args.port:
            print("ERROR: --port is required for data collection")
            print("  Example: python latency_test.py --port COM3 --label baseline")
            sys.exit(1)
        collect(args.port, args.baud, args.label, args.samples)


if __name__ == "__main__":
    main()
