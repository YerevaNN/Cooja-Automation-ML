#!/usr/bin/env python3
"""
Submit a single Cooja topology run via Submitit (Slurm backend).

Pipeline per job:
1) Generate topology JSON (configurable type/n/spacing/seed/radio/duration) via `core/topology/topology_generator.py`
2) Generate CSC from topology JSON via `core/csc/csc_generator.py`
3) Run headless Cooja (gradlew -nogui <csc>)
4) Parse COOJA.TESTLOG → results.csv via `core/parse/rpl_log_to_csv_v3.py`
5) Add (x,y) from CSC → results_xy.csv via `core/parse/rpl_log_add_xyz_to_csv.py`
6) Write run_meta.json
7) (Optional) Load into PostgreSQL via `core/db/load_csv_to_postgres.py`

Defaults set for a 25-mote sparse_grid, but all parameters are CLI-configurable.


Sample Usage: (can specify max sparse attempts.)
python3 scripts/submitit_single_job.py \
  --repo <path to your repo which has the makefile and project-conf.h> \
  --cooja-root /opt/contiki-ng/tools/cooja \
  --type sparse_grid --n 25 --spacing 40 --seed 1 --duration 300 --tx 50 --ix 100 \
  --partition a100 --cpus 2 --mem 4 --timeout 30


python3 scripts/submitit_single_job.py \
  --repo <path to your repo which has the makefile and project-conf.h> \
  --cooja-root /opt/contiki-ng/tools/cooja \
  --type sparse_grid --n 25 --spacing 30 --seed 1 --duration 300 --tx 50 --ix 100 \
  --partition a100 --cpus 2 --mem 4 --timeout 30 --java-home /usr/lib/jvm/java-17-openjdk-amd64


"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import submitit


def compute_run_id(topology_id: str, topo_json_path: str) -> str:
    job = os.getenv("SLURM_JOB_ID")
    if job:
        return f"{topology_id}_job{job}"
    ts = time.strftime("%Y%m%d%H%M%S")
    h = hashlib.sha256((str(Path(topo_json_path).resolve()) + ts).encode()).hexdigest()[:8]
    return f"{topology_id}_{ts}_{h}"


def run_cmd(cmd: list[str], cwd: Path | None = None, out_file: Path | None = None, env: dict | None = None) -> int:
    if out_file is None:
        result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False, env=env)
        return result.returncode
    with out_file.open("w") as f:
        result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, stdout=f, stderr=subprocess.STDOUT, text=True, check=False, env=env)
        return result.returncode


def cooja_job(repo: str,
              cooja_root: str,
              topo_type: str,
              n: int,
              spacing: float,
              seed: int,
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
              pg_pass: str) -> dict:
    repo_path = Path(repo).resolve()
    runs_dir = repo_path / "artifacts"
    runs_dir.mkdir(parents=True, exist_ok=True)

    topology_id = f"{topo_type}_n{n}_s{int(spacing)}_seed{seed}"

    # Local run dir (per Slurm best practices, use local fast storage if available)
    run_id = compute_run_id(topology_id, str(repo_path / "topologies" / f"{topology_id}.json"))
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    top_json = run_dir / f"{topology_id}.json"
    csc_path = run_dir / "sim.csc"
    stdout_path = run_dir / "stdout.txt"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    # 1) Topology JSON
    print("[cooja_job] Step 1: Generating topology JSON...")
    gen_cmd = [
        "python3", str(repo_path / "core" / "topology" / "topology_generator.py"), topo_type,
        "-n", str(n), "-s", str(spacing), "--seed", str(seed),
        "--tx-range", str(tx_range), "--interference-range", str(interference_range),
        "--duration", str(duration_s), "--out", str(top_json),
    ]
    if topo_type == "sparse_grid" and sparse_max_attempts:
        gen_cmd += ["--sparse-max-attempts", str(sparse_max_attempts)]
    rc = run_cmd(gen_cmd)
    print(f"[cooja_job] Step 1 exit code: {rc}")
    if rc != 0:
        raise RuntimeError(f"Topology generation failed with code {rc}")

    # 2) CSC generation (point build_root to per-run dir)
    print("[cooja_job] Step 2: Generating CSC from topology...")
    rc = run_cmd([
        "python3", str(repo_path / "core" / "csc" / "csc_generator.py"),
        str(top_json), str(csc_path),
        "--build-root", str(run_dir / "build"),
    ])
    print(f"[cooja_job] Step 2 exit code: {rc}")
    if rc != 0:
        raise RuntimeError(f"CSC generation failed with code {rc}")

    # 3) Headless Cooja
    # Run Gradle project located at cooja_root, but set cwd to run_dir so COOJA.TESTLOG lands there
    gradle_cmd = [
        str(Path(cooja_root) / "gradlew"),
        "--scan",
        "-p", str(Path(cooja_root)),
        "run",
        f"--args=--no-gui {csc_path.as_posix()}",
    ]
    gradle_env = os.environ.copy()
    if java_home:
        gradle_env["JAVA_HOME"] = java_home
        gradle_env["PATH"] = f"{java_home}/bin:" + gradle_env.get("PATH", "")
        gradle_cmd.insert(1, f"-Dorg.gradle.java.home={java_home}")
    print("[cooja_job] Step 3: Running Cooja headless via Gradle...")
    print(f"[cooja_job] Step 3: Gradle command: {gradle_cmd}")
    # current time
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[cooja_job] Step 3: Time at Start of Cooja Simulation: {current_time}")
    rc = run_cmd(gradle_cmd, cwd=run_dir, out_file=stdout_path, env=gradle_env) #TODO: is this blocking or async?
    print(f"[cooja_job] Step 3 exit code: {rc}")
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[cooja_job] Step 3: Time at End of Cooja Simulation: {current_time}")
    # Do not raise immediately; continue to collect artifacts if possible

    # Normalize log filename
    testlog = run_dir / "COOJA.testlog"
    cooja_log = run_dir / "cooja.log"
    if testlog.exists():
        print(f"[cooja_job] Step 3: Moving/Renaming {testlog} to {cooja_log}")
        shutil.move(str(testlog), str(cooja_log))
    else:
        print(f"[cooja_job] Step 3: {testlog} does not exist")

    # 4) Parse logs to CSV
    results_csv = run_dir / "results.csv"
    if cooja_log.exists():
        print("[cooja_job] Step 4: Parsing cooja.log to results.csv...")
        rc_parse = run_cmd(["python3", str(repo_path / "core" / "parse" / "rpl_log_to_csv_v3.py"), str(cooja_log), str(results_csv)])
    else:
        rc_parse = 1
        # add some debugging here
        print(f"[cooja_job] Step 4: cooja.log does not exist")
        print(f"[cooja_job] Step 4: stdout_path: {stdout_path}")
        print(f"[cooja_job] Step 4: stdout_path.read_text(): {stdout_path.read_text()}")

    print(f"[cooja_job] Step 4 exit code: {rc_parse}")

    # 5) Add (x,y)
    results_xy = run_dir / "results_xy.csv"
    if results_csv.exists():
        print("[cooja_job] Step 5: Adding XY to results.csv...")
        rc_xy = run_cmd([
            "python3", str(repo_path / "core" / "parse" / "rpl_log_add_xyz_to_csv.py"),
            "--grid", str(csc_path),
            "--input", str(results_csv),
            "--output", str(results_xy),
        ])
    else:
        rc_xy = 1
    print(f"[cooja_job] Step 5 exit code: {rc_xy}")

    # 6) run_meta.json
    status = "ok" if (rc == 0 and rc_parse == 0 and rc_xy == 0 and results_xy.exists() and results_xy.stat().st_size > 0) else "fail"
    error = "" if status == "ok" else f"rc={rc} parse={rc_parse} xy={rc_xy}"
    meta = {
        "run_id": run_id,
        "topology_id": topology_id,
        "topology_path": str(top_json),
        "seed": seed,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "node_count": n,
        "error": error,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[cooja_job] Step 6: Wrote run_meta.json with status={status}")

    # roles.json for DB (server/client mapping)
    roles_path = run_dir / "roles.json"
    topo_dict = json.loads(top_json.read_text())
    roles = {int(m["id"]): m.get("role", "client") for m in topo_dict.get("motes", [])}
    roles_path.write_text(json.dumps(roles))
    print("[cooja_job] Wrote roles.json")

    # 7) Optional PostgreSQL load
    if with_db and results_xy.exists():
        env = os.environ.copy()
        env.update({
            "PGHOST": pg_host,
            "PGPORT": str(pg_port),
            "PGDATABASE": pg_db,
            "PGUSER": pg_user,
            "PGPASSWORD": pg_pass,
        })
        rc_db = subprocess.run([
            "python3", str(repo_path / "core" / "db" / "load_csv_to_postgres.py"),
            "--run-dir", str(run_dir),
            "--meta-json", str(run_dir / "run_meta.json"),
            "--roles-json", str(roles_path),
            "--topology-json", str(top_json),
        ], env=env, check=False).returncode
        if rc_db != 0:
            status = "fail"
            meta["status"] = status
            meta["error"] = (meta.get("error", "") + f" db={rc_db}").strip()
            (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


    # 8) Post-completion: Organize all job-related files into run directory
    print("[cooja_job] Pre Step 8: Organizing job files...")
    
    # Get job ID from environment (available during job execution)
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        print(f"Step 8: [cooja_job] SLURM_JOB_ID: {slurm_job_id}")
        logs_root = repo_path / "slurm_logs"  # Go up from runs/ to get to slurm_logs/
        # print repo_path
        print(f"Step 8: [cooja_job] Repo path: {repo_path}")
        print(f"Step 8: [cooja_job] Logs root directory: {logs_root}")
        # go to <job_id>_<topology_type>
        job_logs_dir = logs_root / f"{slurm_job_id}_{topology_id}"
        print(f"Step 8: [cooja_job] Job logs directory: {job_logs_dir}")
        
        # Move submitit internal files to job directory for better organization  
        submitit_files = [
            f"{slurm_job_id}_0_log.out",
            f"{slurm_job_id}_0_log.err",
            f"{slurm_job_id}_submission.sh",
            f"{slurm_job_id}_submitted.pkl", 
            f"{slurm_job_id}_0_result.pkl"
        ]
        
        for file_name in submitit_files:
            src_file = logs_root / file_name
            if src_file.exists():
                dst_file = job_logs_dir / file_name
                try:
                    shutil.move(str(src_file), str(dst_file))
                    print(f"Step 8: [cooja_job] Moved {file_name} to job directory")
                except Exception as e:
                    print(f"Step 8: [cooja_job] Could not move {file_name}: {e}")
            else:
                print(f"Step 8: [cooja_job] {file_name} does not exist")
        
        print(f"Step 8: [cooja_job] All job files organized in: {job_logs_dir}")



    return {
        "run_id": run_id,
        "status": status,
        "run_dir": str(run_dir),
        "stdout": str(stdout_path),
    }


def check_existing_folder(repo: str, topo_type: str, n: int, spacing: float, seed: int) -> tuple[bool, str]:
    """Check if a folder for this exact combination already exists.
    
    Returns:
        tuple: (folder_exists, existing_folder_name)
    """
    repo_path = Path(repo).resolve()
    runs_dir = repo_path / "runs"
    
    topology_id = f"{topo_type}_n{n}_s{int(spacing)}_seed{seed}"
    folder_pattern = f"{topology_id}_job*"
    existing_folders = list(runs_dir.glob(folder_pattern))
    
    if existing_folders:
        return True, existing_folders[0].name
    return False, ""


def main():
    ap = argparse.ArgumentParser(description="Submit one Cooja topology run via Submitit")
    ap.add_argument("--repo", default="/auto/home/aram.dovlatyan/cooja-experiments/battery_rpl_v2")
    ap.add_argument("--cooja-root", default="/opt/contiki-ng/tools/cooja")

    # Topology params (defaults for 25-mote sparse_grid)
    ap.add_argument("--type", default="sparse_grid")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--spacing", type=float, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=int, default=180)
    ap.add_argument("--tx", type=float, default=50)
    ap.add_argument("--ix", type=float, default=100)
    ap.add_argument("--sparse-max-attempts", type=int, default=10)
    ap.add_argument("--java-home", default=os.getenv("JAVA_HOME", None), help="Path to JDK 17+ for Gradle/Cooja")

    # Slurm resources
    ap.add_argument("--partition", default="a100")
    ap.add_argument("--cpus", type=int, default=2)
    ap.add_argument("--mem", type=int, default=4, help="Memory in GB")
    ap.add_argument("--timeout", type=int, default=30, help="Time limit (minutes)")
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--account", default=None, help="Optional Slurm account to avoid InvalidAccount")

    # Optional DB
    ap.add_argument("--with-db", action="store_true")
    ap.add_argument("--pg-host", default=os.getenv("PGHOST", "localhost"))
    ap.add_argument("--pg-port", type=int, default=int(os.getenv("PGPORT", "5432")))
    ap.add_argument("--pg-db", default=os.getenv("PGDATABASE", "cooja"))
    ap.add_argument("--pg-user", default=os.getenv("PGUSER", "cooja"))
    ap.add_argument("--pg-pass", default=os.getenv("PGPASSWORD", ""))

    args = ap.parse_args()
    
    # Check if folder already exists before submitting to SLURM
    folder_exists, existing_folder = check_existing_folder(
        args.repo, args.type, args.n, args.spacing, args.seed
    )
    
    if folder_exists:
        topology_id = f"{args.type}_n{args.n}_s{int(args.spacing)}_seed{args.seed}"
        print(f"Folder already exists for {topology_id}: {existing_folder}")
        print("Skipping job submission - results may already be available")
        return

    logs_root = Path(args.repo) / "slurm_logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    # For submission, use the common logs_root; configure Slurm to write stdout/err into a per-job directory
    # Where submitit keeps its internal files (submission.sh, submitted.pkl, result.pkl)
    ex = submitit.AutoExecutor(folder=str(logs_root))

    # Per-job log directory (expanded by Slurm; %j is job id, %t is task id)
    job_dir_template = logs_root / f"%j_{args.type}_n{args.n}_s{int(args.spacing)}_seed{args.seed}"
    
    # Only keep "nice" (or any other non-output params) under slurm_additional_parameters
    slurm_params = {
        "nice": args.nice
    }

    # Optional Slurm account to avoid InvalidAccount?
    if args.account:
        slurm_params["account"] = args.account
    
    # Send srun stdout/err to the per-job directory, avoiding empty #SBATCH files
    ex.update_parameters(
        slurm_partition=args.partition,
        cpus_per_task=args.cpus,
        mem_gb=args.mem,
        timeout_min=args.timeout,
        name=f"cooja-{args.type}-n{args.n}-s{int(args.spacing)}-seed{args.seed}",
        slurm_additional_parameters=slurm_params,
    )

    job = ex.submit(
        cooja_job,
        args.repo,
        args.cooja_root,
        args.type,
        args.n,
        args.spacing,
        args.seed,
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
    )
    print(f"Submitted job_id={job.job_id}")
    print("Use: squeue -u $USER to monitor")
    # Create per-job logs directory starting with job id (for later consolidation by the worker)
    job_logs_dir = logs_root / f"{job.job_id}_{args.type}_n{args.n}_s{int(args.spacing)}_seed{args.seed}"
    job_logs_dir.mkdir(parents=True, exist_ok=True)

    
    print(f"Submitit files remain in: {logs_root}")
    print(f"Slurm stdout/stderr will go to: {job_logs_dir}")


if __name__ == "__main__":
    main()


