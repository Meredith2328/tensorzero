#!/usr/bin/env python3
"""
Automated Postgres inference load test runner.

Starts Postgres via docker compose, compiles binaries, runs migrations,
starts the gateway, runs the load test for each write-mode variant,
collects JSON reports, and prints a comparison table.

Usage:
    # Run all variants at 500 QPS for 60s:
    python scripts/load-test/run_postgres_load_test.py --rate 500 --duration 60s

    # Run only specific variants:
    python scripts/load-test/run_postgres_load_test.py --rate 500 --duration 60s --variants sync async

    # Skip compilation (if already built):
    python scripts/load-test/run_postgres_load_test.py --rate 500 --duration 60s --skip-compile

    # Skip docker (Postgres already running, env vars already set):
    python scripts/load-test/run_postgres_load_test.py --rate 500 --duration 60s --skip-docker
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Resolve paths relative to the repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CRATES_DIR = REPO_ROOT / "crates"
LOAD_TEST_DIR = CRATES_DIR / "tensorzero-core" / "tests" / "load"
DOCKER_COMPOSE_FILE = SCRIPT_DIR / "docker-compose.yml"

POSTGRES_PORT = 5433
POSTGRES_URL = f"postgres://postgres:postgres@localhost:{POSTGRES_PORT}/tensorzero-load-test"


@dataclass
class Variant:
    name: str
    config: str
    drain_wait_ms: int


VARIANTS = [
    Variant("sync", "tensorzero.postgres-only.sync.toml", 5000),
    Variant("async", "tensorzero.postgres-only.async.toml", 5000),
    Variant("batch-fast", "tensorzero.postgres-only.batch.fast.toml", 8000),
    Variant("batch-balanced", "tensorzero.postgres-only.batch.balanced.toml", 10000),
    Variant("batch-throughput", "tensorzero.postgres-only.batch.throughput.toml", 12000),
]

VARIANT_MAP = {v.name: v for v in VARIANTS}


def log(msg: str) -> None:
    print(f"\033[1;36m==> {msg}\033[0m", flush=True)


def log_error(msg: str) -> None:
    print(f"\033[1;31m==> ERROR: {msg}\033[0m", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------

def postgres_is_reachable() -> bool:
    """Check if Postgres is already accepting connections on the expected port."""
    import socket

    try:
        with socket.create_connection(("localhost", POSTGRES_PORT), timeout=2):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def start_docker(skip: bool) -> bool:
    """Start Postgres via docker compose. Returns True if we started it (and should stop it)."""
    if skip:
        log("Skipping docker compose (--skip-docker)")
        return False

    if postgres_is_reachable():
        log(f"Postgres already reachable on port {POSTGRES_PORT}, skipping docker compose")
        return False

    log("Starting Postgres via docker compose...")
    subprocess.run(
        [
            "docker", "compose",
            "-f", str(DOCKER_COMPOSE_FILE),
            "up", "-d",
            "--build", "--force-recreate", "--remove-orphans",
        ],
        check=True,
    )

    log("Waiting for Postgres to be healthy...")
    deadline = time.time() + 60
    while time.time() < deadline:
        if postgres_is_reachable():
            # Also check pg_isready via docker
            result = subprocess.run(
                [
                    "docker", "compose",
                    "-f", str(DOCKER_COMPOSE_FILE),
                    "ps", "--format", "json",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    for line in result.stdout.strip().splitlines():
                        svc = json.loads(line)
                        if svc.get("Health") == "healthy":
                            log("Postgres is ready.")
                            return True
                except json.JSONDecodeError:
                    pass
        time.sleep(1)

    log_error("Postgres did not become healthy in 60s")
    sys.exit(1)


def stop_docker(we_started_it: bool) -> None:
    if not we_started_it:
        return
    log("Stopping Postgres docker compose...")
    subprocess.run(
        ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "down", "-v"],
        check=False,
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_binaries(skip: bool) -> None:
    if skip:
        log("Skipping compilation (--skip-compile)")
        return

    log("Compiling gateway (release, e2e_tests feature)...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "gateway", "--features", "e2e_tests"],
        cwd=CRATES_DIR,
        check=True,
    )

    log("Compiling load test binary (release)...")
    subprocess.run(
        ["cargo", "build", "--release", "--package", "postgres-inference-load-test"],
        cwd=CRATES_DIR,
        check=True,
    )


def find_binary(name: str) -> Path:
    candidate = CRATES_DIR / "target" / "release" / name
    if candidate.exists():
        return candidate
    log_error(f"Binary not found: {candidate}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

def run_migrations(gateway_bin: Path) -> None:
    log("Running Postgres migrations...")
    result = subprocess.run(
        [str(gateway_bin), "--run-postgres-migrations"],
        cwd=CRATES_DIR,
        env={**os.environ, "TENSORZERO_POSTGRES_URL": POSTGRES_URL},
        timeout=60,
    )
    if result.returncode != 0:
        log_error("Postgres migrations failed")
        sys.exit(1)
    log("Migrations complete.")


# ---------------------------------------------------------------------------
# Gateway management
# ---------------------------------------------------------------------------

def wait_for_gateway(url: str, timeout: int = 30) -> bool:
    health_url = f"{url}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def start_gateway(variant: Variant, gateway_bin: Path, port: int) -> subprocess.Popen:
    config_path = LOAD_TEST_DIR / variant.config
    bind_address = f"0.0.0.0:{port}"
    env = {
        **os.environ,
        "TENSORZERO_POSTGRES_URL": POSTGRES_URL,
        "TENSORZERO_INTERNAL_FLAG_ENABLE_POSTGRES_WRITE": "1",
    }
    log(f"Starting gateway on port {port} with config {variant.config}...")
    proc = subprocess.Popen(
        [str(gateway_bin), "--config-file", str(config_path), "--bind-address", bind_address],
        cwd=CRATES_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


def stop_process(proc: subprocess.Popen, name: str = "process") -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        log(f"  {name} did not exit, sending SIGKILL...")
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Load test execution
# ---------------------------------------------------------------------------

def run_load_test(
    variant: Variant,
    load_test_bin: Path,
    args: argparse.Namespace,
    report_path: Path,
    gateway_url: str,
) -> dict | None:
    cmd = [
        str(load_test_bin),
        "--gateway-url", gateway_url,
        "--function-name", "load_test_chat",
        "-r", str(args.rate),
        "-c", str(args.concurrency),
        "-d", args.duration,
        "--max-tokens", "128",
        "--drain-wait-ms", str(variant.drain_wait_ms),
        "--max-error-rate", str(args.max_error_rate),
        "--benchmark-report-json", str(report_path),
        "--verify-timeout-s", str(args.verify_timeout_s),
    ]
    if args.prompt_chars:
        cmd.extend(["--prompt-chars", str(args.prompt_chars)])
    if args.max_p99_latency_ms:
        cmd.extend(["--max-p99-latency-ms", str(args.max_p99_latency_ms)])

    env = {
        **os.environ,
        "TENSORZERO_POSTGRES_URL": POSTGRES_URL,
        "TENSORZERO_INTERNAL_FLAG_ENABLE_POSTGRES_WRITE": "1",
    }

    log(f"Running load test: {variant.name} @ {args.rate} QPS for {args.duration}")
    result = subprocess.run(cmd, cwd=CRATES_DIR, env=env)

    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    else:
        log_error(f"  Report not generated for {variant.name} (exit code {result.returncode})")
        return None


def run_variant(
    variant: Variant,
    gateway_bin: Path,
    load_test_bin: Path,
    args: argparse.Namespace,
    output_dir: Path,
    port: int,
) -> dict | None:
    gateway_url = f"http://localhost:{port}"
    report_path = output_dir / f"{variant.name}.json"

    gateway_proc = start_gateway(variant, gateway_bin, port)
    try:
        if not wait_for_gateway(gateway_url, timeout=60):
            log_error(f"  Gateway failed to start for {variant.name}")
            stderr = gateway_proc.stderr.read().decode() if gateway_proc.stderr else ""
            if stderr:
                log_error(f"  Gateway stderr:\n{stderr[:2000]}")
            return None

        log(f"  Gateway ready on {gateway_url}")
        report = run_load_test(variant, load_test_bin, args, report_path, gateway_url)
        return report
    finally:
        stop_process(gateway_proc, f"gateway ({variant.name})")
        # Brief pause to release the port
        time.sleep(1)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict[str, dict]) -> None:
    if not results:
        log_error("No results to compare.")
        return

    log("Comparison Table")
    print()

    header = (
        f"{'Variant':<20} {'QPS':>8} {'p99 (ms)':>10} {'Err %':>8} "
        f"{'Chat Rows':>10} {'Model Rows':>11} {'Pass':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, report in results.items():
        qps = report.get("achieved_qps", "?")
        p99 = report.get("p99_latency_ms", "?")
        err = report.get("error_rate", "?")
        chat_rows = report.get("chat_inference_count", "?")
        model_rows = report.get("model_inference_count", "?")
        passed = report.get("all_checks_passed", "?")

        err_str = f"{err:.4f}" if isinstance(err, (int, float)) else str(err)
        qps_str = f"{qps:.1f}" if isinstance(qps, (int, float)) else str(qps)
        p99_str = f"{p99:.1f}" if isinstance(p99, (int, float)) else str(p99)
        pass_str = "PASS" if passed is True else ("FAIL" if passed is False else str(passed))

        print(
            f"{name:<20} {qps_str:>8} {p99_str:>10} {err_str:>8} "
            f"{str(chat_rows):>10} {str(model_rows):>11} {pass_str:>6}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Postgres inference load tests across write-mode variants."
    )
    parser.add_argument(
        "-r", "--rate", type=int, required=True,
        help="Target QPS (requests per second)",
    )
    parser.add_argument(
        "-d", "--duration", type=str, default="60s",
        help="Test duration (e.g., 60s, 120s). Default: 60s",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=16,
        help="Number of concurrent workers. Default: 16",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        choices=list(VARIANT_MAP.keys()),
        help="Variants to run. Default: all",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for JSON reports. Default: /tmp/load-test-results-<timestamp>",
    )
    parser.add_argument(
        "--skip-compile", action="store_true",
        help="Skip cargo compilation (use previously built binaries)",
    )
    parser.add_argument(
        "--skip-docker", action="store_true",
        help="Skip docker compose (Postgres already running)",
    )
    parser.add_argument(
        "--keep-docker", action="store_true",
        help="Don't tear down docker compose after tests",
    )
    parser.add_argument(
        "--max-error-rate", type=float, default=0.01,
        help="Max allowed error rate. Default: 0.01",
    )
    parser.add_argument(
        "--max-p99-latency-ms", type=int, default=None,
        help="Optional p99 latency threshold in ms",
    )
    parser.add_argument(
        "--verify-timeout-s", type=int, default=30,
        help="Max time to wait for DB parity checks. Default: 30",
    )
    parser.add_argument(
        "--prompt-chars", type=int, default=None,
        help="Payload size in characters. Default: use load test default (256)",
    )
    parser.add_argument(
        "--port", type=int, default=3099,
        help="Gateway port (avoids conflicts with dev gateway). Default: 3099",
    )
    args = parser.parse_args()

    # Resolve variants
    variants_to_run = (
        [VARIANT_MAP[v] for v in args.variants] if args.variants else VARIANTS
    )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(f"/tmp/load-test-results-{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Results will be saved to {output_dir}")
    log(f"Variants: {', '.join(v.name for v in variants_to_run)}")
    log(f"Rate: {args.rate} QPS, Duration: {args.duration}, Concurrency: {args.concurrency}")
    print()

    # Start Postgres
    we_started_docker = start_docker(args.skip_docker)

    try:
        # Compile
        compile_binaries(args.skip_compile)

        # Find binaries
        gateway_bin = find_binary("gateway")
        load_test_bin = find_binary("postgres-inference-load-test")
        log(f"Gateway binary: {gateway_bin}")
        log(f"Load test binary: {load_test_bin}")
        print()

        # Run migrations
        run_migrations(gateway_bin)
        print()

        # Run each variant sequentially
        results: dict[str, dict] = {}
        for variant in variants_to_run:
            log(f"--- Variant: {variant.name} ---")
            report = run_variant(
                variant, gateway_bin, load_test_bin, args, output_dir, args.port
            )
            if report:
                results[variant.name] = report
                log(
                    f"  Completed: QPS={report.get('achieved_qps', '?')}, "
                    f"p99={report.get('p99_latency_ms', '?')}ms, "
                    f"errors={report.get('error_rate', '?')}"
                )
            else:
                log_error(f"  Variant {variant.name} failed")
            print()

        # Summary
        print_comparison_table(results)

        # Save combined results
        combined_path = output_dir / "combined.json"
        with open(combined_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"Combined results saved to {combined_path}")

    finally:
        if not args.keep_docker:
            stop_docker(we_started_docker)


if __name__ == "__main__":
    main()
