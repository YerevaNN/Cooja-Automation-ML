#!/usr/bin/env python3
"""
Submit Cooja topology runs via Submitit (Slurm backend) for parameter sweeps.

This script generates parameter combinations and submits them continuously with
real-time progress tracking using multiple TQDM progress bars. Shows progress
by spacing, N values, and seeds with queue status monitoring.

The coordination runs locally (not on cluster) to avoid blocking resources.
Includes queue management to prevent overwhelming Slurm and resumability.

Usage:
python3 scripts/submitit_batch_sweep.py \
  --repo /auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2 \
  --cooja-root /opt/contiki-ng/tools/cooja \
  --n-start 15 --n-end 75 \
  --spacing-start 20 --spacing-end 30 --spacing-step 2 \
  --seed-start 1 --seed-end 30 \
  --partition a100

This will submit all combinations of N (15-75), spacing (20, 22, 24, 26, 28, 30),
and SEED (1-30) with continuous progress tracking and configurable delays between submissions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

import submitit
from tqdm import tqdm


def generate_combinations(
    n_start: int,
    n_end: int,
    n_step: int,
    spacing_start: int,
    spacing_end: int,
    spacing_step: int,
    seed_start: int,
    seed_end: int,
    seed_step: int,
) -> List[Tuple[int, int, int]]:
    """Generate all parameter combinations."""
    n_values = list(range(n_start, n_end + 1, n_step))
    spacing_values = list(range(spacing_start, spacing_end + 1, spacing_step))
    seed_values = list(range(seed_start, seed_end + 1, seed_step))
    
    combinations = list(itertools.product(n_values, spacing_values, seed_values))
    return combinations


def organize_combinations_by_spacing(
    combinations: List[Tuple[int, int, int]]
) -> Dict[int, List[Tuple[int, int]]]:
    """Organize combinations by spacing value."""
    spacing_groups = {}
    for n, spacing, seed in combinations:
        if spacing not in spacing_groups:
            spacing_groups[spacing] = []
        spacing_groups[spacing].append((n, seed))
    
    return spacing_groups


def check_existing_folder(repo: str, topo_type: str, n: int, spacing: float, seed: int) -> tuple[bool, str]:
    """Check if a folder for this exact combination already exists.
    
    Returns:
        tuple: (folder_exists, existing_folder_name)
    """
    repo_path = Path(repo).resolve()
    runs_dir = repo_path / "artifacts"
    
    topology_id = f"{topo_type}_n{n}_s{int(spacing)}_seed{seed}"
    folder_pattern = f"{topology_id}_job*"
    existing_folders = list(runs_dir.glob(folder_pattern))
    
    if existing_folders:
        return True, existing_folders[0].name
    return False, ""


def get_user_job_count() -> int:
    """Get the number of jobs currently in the queue for this user."""
    try:
        result = subprocess.run(['squeue', '-u', os.getenv('USER', ''), '-h'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        else:
            return 0
    except Exception as e:
        return 0


def wait_for_queue_space(max_jobs: int, pbar=None) -> None:
    """Wait until queue has space for more jobs."""
    while True:
        current_jobs = get_user_job_count()
        if current_jobs < max_jobs:
            if pbar:
                pbar.set_postfix({"Queue": f"{current_jobs}/{max_jobs}"})
            return
        
        if pbar:
            pbar.set_postfix({"Queue": f"{current_jobs}/{max_jobs} - Waiting..."})
        time.sleep(1)  # Check every second when queue is full


def load_progress_file(progress_file: Path) -> Dict[str, Any]:
    """Load progress from file."""
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            pass
    
    return {
        "completed_batches": [],
        "submitted_combinations": [],
        "total_combinations": 0,
        "successful_submissions": 0,
        "failed_submissions": 0
    }


def save_progress(progress_file: Path, progress: Dict[str, Any]) -> None:
    """Save progress to file."""
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        pass


def submit_single_combination(
    repo: str,
    cooja_root: str,
    n: int,
    spacing: int,
    seed: int,
    topo_type: str,
    duration_s: int,
    tx_range: float,
    interference_range: float,
    sparse_max_attempts: int,
    java_home: str | None,
    with_db: bool,
    pg_host: str,
    pg_port: int,
    pg_db: str,
    pg_user: str,
    pg_pass: str,
    partition: str,
    individual_cpus: int,
    individual_mem: int,
    individual_timeout: int
) -> bool:
    """Submit a single job combination."""
    repo_path = Path(repo).resolve()
    
    try:
        # Call the single job script for this combination
        cmd = [
            "python3", str(repo_path / "core" / "run" / "submitit_single_job.py"),
            "--repo", repo,
            "--cooja-root", cooja_root,
            "--type", topo_type,
            "--n", str(n),
            "--spacing", str(spacing),
            "--sparse-max-attempts", str(sparse_max_attempts),
            "--seed", str(seed),
            "--duration", str(duration_s),
            "--tx", str(tx_range),
            "--ix", str(interference_range),
            "--partition", partition,
            "--cpus", str(individual_cpus),
            "--mem", str(individual_mem),
            "--timeout", str(individual_timeout),
        ]
        
        if java_home:
            cmd.extend(["--java-home", java_home])
        
        if with_db:
            cmd.extend([
                "--with-db",
                "--pg-host", pg_host,
                "--pg-port", str(pg_port),
                "--pg-db", pg_db,
                "--pg-user", pg_user,
                "--pg-pass", pg_pass,
            ])
        
        # Execute the single job submission
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)
        return result.returncode == 0
        
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Submit Cooja topology runs with continuous progress tracking")
    
    # Required paths
    ap.add_argument("--repo", required=True, help="Repository root path")
    ap.add_argument("--cooja-root", required=True, help="Cooja installation path")
    
    # Parameter ranges
    ap.add_argument("--n-start", type=int, default=15, help="Start N value")
    ap.add_argument("--n-end", type=int, default=75, help="End N value")
    ap.add_argument("--n-step", type=int, default=5, help="N step size")
    ap.add_argument("--spacing-start", type=int, default=20, help="Start spacing value")
    ap.add_argument("--spacing-end", type=int, default=30, help="End spacing value")
    ap.add_argument("--spacing-step", type=int, default=2, help="Spacing step size")
    ap.add_argument("--seed-start", type=int, default=1, help="Start seed value")
    ap.add_argument("--seed-end", type=int, default=30, help="End seed value")
    ap.add_argument("--seed-step", type=int, default=1, help="Seed step size")
    
    # Simulation parameters
    ap.add_argument("--type", default="sparse_grid", help="Topology type")
    ap.add_argument("--duration", type=int, default=300, help="Simulation duration (seconds)")
    ap.add_argument("--tx", type=float, default=50, help="Transmission range")
    ap.add_argument("--ix", type=float, default=100, help="Interference range")
    ap.add_argument("--sparse-max-attempts", type=int, default=100, help="Max attempts for sparse grid")
    ap.add_argument("--java-home", default=os.getenv("JAVA_HOME", None), help="Path to JDK 17+")
    
    # Slurm resources for individual jobs
    ap.add_argument("--partition", default="a100", help="Slurm partition")
    ap.add_argument("--individual-cpus", type=int, default=2, help="CPUs per individual job")
    ap.add_argument("--individual-mem", type=int, default=4, help="Memory in GB per individual job")
    ap.add_argument("--individual-timeout", type=int, default=30, help="Time limit (minutes) per individual job")
    ap.add_argument("--nice", type=int, default=0, help="Nice value")
    ap.add_argument("--account", default=None, help="Optional Slurm account")
    
    # Queue management
    ap.add_argument("--max-jobs", type=int, default=100, help="Maximum jobs in queue at once")
    
    # Database options
    ap.add_argument("--with-db", action="store_true", help="Enable database loading")
    ap.add_argument("--pg-host", default=os.getenv("PGHOST", "localhost"))
    ap.add_argument("--pg-port", type=int, default=int(os.getenv("PGPORT", "5432")))
    ap.add_argument("--pg-db", default=os.getenv("PGDATABASE", "cooja_experiments"))
    ap.add_argument("--pg-user", default=os.getenv("PGUSER", "postgres"))
    ap.add_argument("--pg-pass", default=os.getenv("PGPASSWORD", "postgres"))
    
    # Control options
    ap.add_argument("--dry-run", action="store_true", help="Print what would be submitted without actually submitting")
    ap.add_argument("--resume", action="store_true", help="Resume from previous run")
    ap.add_argument("--clear-progress", action="store_true", help="Clear progress file and start fresh")
    ap.add_argument("--submission-delay", type=float, default=0.1, help="Delay in seconds between submissions (default: 0.1)")
    
    args = ap.parse_args()
    
    # Setup progress tracking
    progress_file = Path(args.repo) / "batch_progress.json"
    
    # Clear progress if requested
    if args.clear_progress:
        if progress_file.exists():
            progress_file.unlink()
    
    progress = load_progress_file(progress_file)
    
    # Generate all combinations
    combinations = generate_combinations(
        args.n_start, args.n_end, args.n_step,
        args.spacing_start, args.spacing_end, args.spacing_step,
        args.seed_start, args.seed_end, args.seed_step
    )
    
    # Update total combinations if not set
    if progress["total_combinations"] == 0:
        progress["total_combinations"] = len(combinations)
        save_progress(progress_file, progress)
    
    # Organize by spacing for progress tracking
    spacing_groups = organize_combinations_by_spacing(combinations)
    
    if args.dry_run:
        return
    
    # Create progress bars
    spacing_values = sorted(spacing_groups.keys())
    spacing_pbar = tqdm(spacing_values, desc="Spacing", position=0, leave=True)
    
    total_submitted = 0
    total_failed = 0
    
    for spacing in spacing_pbar:
        spacing_pbar.set_description(f"Spacing {spacing}")
        combinations_for_spacing = spacing_groups[spacing]
        
        # N progress bar
        n_values = sorted(set(n for n, seed in combinations_for_spacing))
        n_pbar = tqdm(n_values, desc="N", position=1, leave=False)
        
        for n in n_pbar:
            n_pbar.set_description(f"N {n}")
            
            # Get all seeds for this N and spacing
            seeds_for_n = [seed for n_val, seed in combinations_for_spacing if n_val == n]
            seeds_for_n.sort()
            
            # Seed progress bar (lowest level)
            seed_pbar = tqdm(seeds_for_n, desc="Seed", position=2, leave=False)
            
            for seed in seed_pbar:
                seed_pbar.set_description(f"Seed {seed}")
                
                # Check if already submitted
                combination_key = f"{n}_{spacing}_{seed}"
                if combination_key in progress["submitted_combinations"]:
                    continue
                
                # Check if folder already exists
                folder_exists, existing_folder = check_existing_folder(
                    args.repo, args.type, n, spacing, seed
                )
                
                if folder_exists:
                    # Mark as successful since folder exists
                    total_submitted += 1
                    progress["successful_submissions"] += 1
                    progress["submitted_combinations"].append(combination_key)
                    save_progress(progress_file, progress)
                    continue
                
                # Wait for queue space if needed
                wait_for_queue_space(args.max_jobs, seed_pbar)
                
                # Submit the job
                success = submit_single_combination(
                    args.repo,
                    args.cooja_root,
                    n, spacing, seed,
                    args.type,
                    args.duration,
                    args.tx,
                    args.ix,
                    args.sparse_max_attempts,
                    args.java_home,
                    args.with_db,
                    args.pg_host,
                    args.pg_port,
                    args.pg_db,
                    args.pg_user,
                    args.pg_pass,
                    args.partition,
                    args.individual_cpus,
                    args.individual_mem,
                    args.individual_timeout
                )
                
                if success:
                    total_submitted += 1
                    progress["successful_submissions"] += 1
                    progress["submitted_combinations"].append(combination_key)
                else:
                    total_failed += 1
                    progress["failed_submissions"] += 1
                
                # Save progress
                save_progress(progress_file, progress)
                
                # Configurable delay after each submission
                time.sleep(args.submission_delay)
    
    # Final status
    spacing_pbar.set_postfix({"Total": f"{total_submitted} submitted, {total_failed} failed"})


if __name__ == "__main__":
    main()