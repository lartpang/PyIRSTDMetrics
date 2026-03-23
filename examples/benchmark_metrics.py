"""
Benchmark script: Evaluate the impact of target count on metric computation efficiency.

Synthesizes samples with varying target counts using fixed random seeds,
times all metrics, and plots target count vs. computation time.

Usage:
    # Run benchmark and save results to benchmark.json
    python benchmark_metrics.py

    # Only plot from existing benchmark.json
    python benchmark_metrics.py --plot-only
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
import py_irstd_metrics

NUM_REPEATS = 5
NUM_BINS = 4
TARGET_COUNTS = [2**i for i in range(8)]
METRIC_GROUPS = {
    "Pixel-wise Metrics": {
        "metrics": ["CMMetrics"],
        "colors": ["#2196F3"],
        "markers": ["o"],
    },
    "Target-wise Metrics (Greedy Matching)": {
        "metrics": ["BasicPDFa", "ShootPDFa"],
        "colors": ["#4CAF50", "#FF9800"],
        "markers": ["s", "^"],
    },
    "Target-wise Metrics (Hungarian Matching)": {
        "metrics": ["OPDCMatching", "DistMatching", "HIoUError"],
        "colors": ["#F44336", "#9C27B0", "#795548"],
        "markers": ["D", "v", "P"],
    },
}
METRIC_LABELS = {
    "CMMetrics": "CMMetrics",
    "BasicPDFa": "Distance-Rule PD and Fa",
    "ShootPDFa": "Shoot-Rule PD and Fa",
    "OPDCMatching": "OPDC-based HIoU",
    "DistMatching": "Distance-based HIoU",
    "HIoUError": "HIoU Error",
}
BENCHMARK_FILE = Path(__file__).parent / "benchmark.json"


def create_all_metrics(num_bins=8, metric_names=METRIC_LABELS.keys()):
    """Create all metric instances, returning a {name: metric_instance} dictionary."""
    metrics = {}

    # Pixel-wise metrics
    if "CMMetrics" in metric_names:
        metrics["CMMetrics"] = py_irstd_metrics.CMMetrics(
            num_bins=num_bins,
            threshold=0.5,
            metric_handlers={
                "iou": py_irstd_metrics.IoUHandler(with_dynamic=False, with_binary=True, sample_based=False),
                "f1": py_irstd_metrics.FmeasureHandler(
                    with_dynamic=False, with_binary=True, sample_based=False, beta=1
                ),
                "pre": py_irstd_metrics.PrecisionHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "rec": py_irstd_metrics.RecallHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "tpr": py_irstd_metrics.TPRHandler(with_dynamic=True, with_binary=False, sample_based=False),
                "fpr": py_irstd_metrics.FPRHandler(with_dynamic=True, with_binary=False, sample_based=False),
            },
        )
    # Target-wise metrics - distance-based PD-Fa
    if "BasicPDFa" in metric_names:
        metrics["BasicPDFa"] = py_irstd_metrics.ProbabilityDetectionAndFalseAlarmRate(
            num_bins=num_bins, distance_threshold=3
        )
    # Target-wise metrics - shooting rule-based PD-Fa
    if "ShootPDFa" in metric_names:
        metrics["ShootPDFa"] = py_irstd_metrics.ShootingRuleBasedProbabilityDetectionAndFalseAlarmRate(
            num_bins=num_bins, box_1_radius=1, box_2_radius=4
        )
    # Target-wise metrics - OPDC matching
    if "OPDCMatching" in metric_names:
        metrics["OPDCMatching"] = py_irstd_metrics.MatchingBasedMetrics(
            num_bins=num_bins,
            matching_method=py_irstd_metrics.OPDCMatching(overlap_threshold=0.5, distance_threshold=3),
        )
    # Target-wise metrics - distance matching
    if "DistMatching" in metric_names:
        metrics["DistMatching"] = py_irstd_metrics.MatchingBasedMetrics(
            num_bins=num_bins, matching_method=py_irstd_metrics.DistanceOnlyMatching(distance_threshold=3)
        )
    # HIoU-based error analysis
    if "HIoUError" in metric_names:
        metrics["HIoUError"] = py_irstd_metrics.HierarchicalIoUBasedErrorAnalysis(
            num_bins=num_bins, overlap_threshold=0.5, distance_threshold=3
        )
    return metrics


def generate_synthetic_sample(num_targets: int, target_radius_range: tuple = (3, 12), image_size: tuple = (512, 512)):
    """Generate a synthetic mask and pred with a specified number of targets.

    To simulate a real scenario:
    - mask: Place num_targets small circular targets
    - pred: Add random offset, scale jitter, and noise to the mask to simulate imperfect prediction

    Args:
        num_targets: Number of targets
        target_radius_range: Target radius range
        image_size: Image size (H, W)

    Returns:
        pred (np.ndarray[float]): 0~1 continuous prediction map
        mask (np.ndarray[bool]): Binary GT mask
    """
    rng = np.random.default_rng(42 + num_targets)

    H, W = image_size
    mask = np.zeros((H, W), dtype=bool)
    pred_float = np.zeros((H, W), dtype=float)

    # Randomly place targets on the image, ensuring no overlap
    margin = max(target_radius_range) + 2
    max_attempts = num_targets * 50

    attempts = 0
    positions = []
    while len(positions) < num_targets and attempts < max_attempts:
        cy = rng.integers(margin, H - margin)
        cx = rng.integers(margin, W - margin)
        r = rng.integers(target_radius_range[0], target_radius_range[1] + 1)

        # Check for overlap with existing targets
        is_valid = True
        for py, px, pr in positions:
            dist = np.sqrt((cy - py) ** 2 + (cx - px) ** 2)
            if dist < r + pr + 4:  # Minimum 4 pixels apart
                is_valid = False
                break

        if is_valid:
            positions.append((cy, cx, r))
        attempts += 1

    # Create coordinate grid
    yy, xx = np.ogrid[:H, :W]

    for cy, cx, r in positions:
        # GT mask: precise circle
        circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
        mask |= circle_mask

        # Pred: add random offset and scale jitter
        offset_y = rng.integers(-2, 3)
        offset_x = rng.integers(-2, 3)
        scale = rng.uniform(0.7, 1.3)
        pred_r = max(1, int(r * scale))
        pred_cy = np.clip(cy + offset_y, pred_r, H - pred_r - 1)
        pred_cx = np.clip(cx + offset_x, pred_r, W - pred_r - 1)

        pred_circle = ((yy - pred_cy) ** 2 + (xx - pred_cx) ** 2).astype(float)
        pred_circle = np.clip(1.0 - pred_circle / (pred_r**2 + 1e-6), 0, 1)
        pred_float = np.maximum(pred_float, pred_circle)

    # Add a small number of FP noise targets (about 20% of the number of targets)
    num_fp = max(1, num_targets // 5)
    for _ in range(num_fp):
        fy = rng.integers(margin, H - margin)
        fx = rng.integers(margin, W - margin)
        fr = rng.integers(target_radius_range[0], target_radius_range[1] + 1)
        fp_circle = ((yy - fy) ** 2 + (xx - fx) ** 2).astype(float)
        fp_intensity = rng.uniform(0.3, 0.8)
        fp_circle = np.clip(fp_intensity - fp_circle / (fr**2 + 1e-6), 0, 1)
        pred_float = np.maximum(pred_float, fp_circle)

    # Add slight background noise
    noise = rng.uniform(0, 0.05, size=(H, W))
    pred_float = np.clip(pred_float + noise, 0, 1)
    return pred_float, mask


def benchmark_single(num_targets):
    """Run benchmark for all metrics with a given number of targets.

    Args:
        num_targets: Number of targets

    Returns:
        dict: {metric_name: average_time(seconds)}
    """
    # Generate sample with fixed seed
    pred, mask = generate_synthetic_sample(num_targets)

    timing_results = {}
    for metric_name in METRIC_LABELS.keys():
        times = []
        for _ in range(NUM_REPEATS):
            # Recreate metric instance each time to ensure independence
            all_metrics = create_all_metrics(num_bins=NUM_BINS)
            metric = all_metrics[metric_name]

            start = time.perf_counter()
            metric.update(pred, mask)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Verify that results can be obtained normally
            metric.get()
        timing_results[metric_name] = np.median(times)
    return timing_results


def run_full_benchmark():
    """Run full benchmark.

    Returns:
        dict: {metric_name: [time for each target count]}
    """
    metric_names = list(METRIC_LABELS.keys())
    all_results = {name: [] for name in metric_names}
    for n_tgt in TARGET_COUNTS:
        print(f"  {n_tgt:>4d} targets: ", end=" ", flush=True)
        results = benchmark_single(num_targets=n_tgt)
        for name in metric_names:
            all_results[name].append(results[name])
        print(" | ".join(f"{name}: {results[name] * 1000:.1f}ms" for name in metric_names))
    return all_results


def save_benchmark(all_results, version, json_path=BENCHMARK_FILE):
    """Save benchmark results as a new version entry in the JSON file.

    Args:
        all_results: {metric_name: [time_ms list]}
        version: Version label string
        json_path: Path to the benchmark JSON file
    """
    # Load existing data or start fresh
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    data[version] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_bins": NUM_BINS,
            "num_repeats": NUM_REPEATS,
            "target_counts": TARGET_COUNTS,
            "image_size": [512, 512],
        },
        "results": {name: [t * 1000 for t in times] for name, times in all_results.items()},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {json_path} (version: {version})")


def plot_all_versions(json_path=BENCHMARK_FILE, save_path="benchmark_results.png"):
    """Plot benchmark results from all versions in the JSON file.

    Each version gets its own subplot, allowing easy visual comparison
    across different code versions.

    Args:
        json_path: Path to the benchmark JSON file
        save_path: Output image path
    """
    assert json_path.exists(), f"Error: {json_path} not found. Run benchmark first."
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    versions = list(data.keys())
    num_versions = len(versions)

    fig, axes = plt.subplots(1, num_versions, figsize=(6 * num_versions, 6), squeeze=False)

    for idx, version in enumerate(versions):
        ax = axes[0, idx]
        entry = data[version]
        target_counts = entry["config"]["target_counts"]
        results = entry["results"]

        for group_name, group_cfg in METRIC_GROUPS.items():
            for name, color, marker in zip(group_cfg["metrics"], group_cfg["colors"], group_cfg["markers"]):
                if name not in results:
                    continue
                times_ms = np.array(results[name])
                ax.plot(
                    target_counts,
                    times_ms,
                    color=color,
                    marker=marker,
                    label=METRIC_LABELS.get(name, name),
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Number of Targets", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title(version, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.suptitle(
        f"Metric Computation Time vs. Number of Targets (num_bins={NUM_BINS}, 512×512, repeats={NUM_REPEATS})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyIRSTDMetrics Benchmark")
    parser.add_argument("--version", type=str, default=None, help="Version label for this benchmark run")
    parser.add_argument("--plot", action="store_true", help="Only plot from existing benchmark.json")
    args = parser.parse_args()

    if args.plot:
        if not BENCHMARK_FILE.exists():
            raise FileNotFoundError(f"Error: {BENCHMARK_FILE} not found. Run benchmark first.")
        plot_all_versions(save_path="benchmark_results.png")
    else:
        version = args.version or f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("=" * 70)
        print("PyIRSTDMetrics Benchmark: Target Count vs. Computation Time")
        print(f"Version: {version}")
        print(f"Target counts: {TARGET_COUNTS}")
        print("=" * 70)
        benchmark_single(num_targets=5)

        print("Running benchmark...")
        all_results = run_full_benchmark()

        save_benchmark(all_results, version=version)
