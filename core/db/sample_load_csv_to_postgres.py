#!/usr/bin/env python3
"""
Append-only loader for PostgreSQL. Safe for parallel inserts (distinct run_id).
Requires env vars or arguments for connection.

Schema alignment:
- Inserts into tables defined in db/db_postgres.sql: runs, simulations
- No upserts; assumes unique run_id per attempt.
"""

import os
import csv
import json
import argparse
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values


def connect_pg(host: str, port: int, db: str, user: str, password: str):
    return psycopg2.connect(host=host, port=port, dbname=db, user=user, password=password)


def insert_run(cur, meta: dict, topo: dict):
    payload = {
        "run_id": meta["run_id"],
        "topology_id": meta["topology_id"],
        "topology_type": topo.get("type"),
        "topology_path": meta["topology_path"],
        "platform": topo.get("platform"),
        "duration_s": int(topo.get("timing", {}).get("duration_s", 0)),
        "seed": meta.get("seed"),
        "started_at": meta["started_at"],
        "finished_at": meta.get("finished_at"),
        "status": meta["status"],
        "node_count": meta["node_count"],
        "error": meta.get("error"),
    }
    cur.execute(
        """
        INSERT INTO runs (
          run_id, topology_id, topology_type, topology_path, platform, duration_s,
          csc_path, seed, started_at, finished_at, status, node_count, error
        ) VALUES (
          %(run_id)s, %(topology_id)s, %(topology_type)s, %(topology_path)s, %(platform)s, %(duration_s)s,
           %(seed)s, %(started_at)s, %(finished_at)s, %(status)s, %(node_count)s, %(error)s
        )
        """,
        payload,
    )


def insert_simulations(cur, run_id: str, csv_path: str, role_map: dict):
    with open(csv_path) as f:
        r = csv.DictReader(f)
        rows = []
        for row in r:
            mote = int(row["mote"]) if row.get("mote") else None
            rows.append(
                (
                    run_id,
                    mote,
                    role_map.get(mote, "client"),
                    _to_float(row.get("x")),
                    _to_float(row.get("y")),
                    _to_int(row.get("cpu")),
                    _to_int(row.get("lpm")),
                    _to_int(row.get("deep_lpm")),
                    _to_int(row.get("listen")),
                    _to_int(row.get("transmit")),
                    _to_int(row.get("off")),
                    _to_int(row.get("total")),
                    _to_int(row.get("initial_battery")),
                    _to_int(row.get("consumed")),
                    _to_int(row.get("remaining")),
                    row.get("status"),
                    _to_int(row.get("uptime")),
                    _to_int(row.get("last_msg_recv_by_root")),
                    _to_int(row.get("sent")),
                    _to_int(row.get("forwarded")),
                )
            )
    sql = (
        "INSERT INTO simulations (run_id, mote, role, x, y, cpu, lpm, deep_lpm, listen, transmit, off, total, initial_battery, consumed, remaining, status, uptime, last_msg_recv_by_root, sent, forwarded) VALUES %s"
    )
    execute_values(cur, sql, rows)


def _to_int(v):
    try:
        return int(v) if v not in (None, "") else None
    except Exception:
        return None


def _to_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except Exception:
        return None


def load(run_dir: str, meta: dict, topo: dict, pg: dict, role_map: dict):
    conn = connect_pg(**pg)
    try:
        with conn:
            with conn.cursor() as cur:
                insert_run(cur, meta, topo)
                insert_simulations(cur, meta["run_id"], str(Path(run_dir) / "results_xy.csv"), role_map)
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--meta-json", required=True)
    ap.add_argument("--roles-json", required=True, help="Map of mote_id->role extracted from topology")
    ap.add_argument("--topology-json", required=True, help="Topology JSON used to derive type/platform/duration")
    ap.add_argument("--pg-host", default=os.getenv("PGHOST", "localhost"))
    ap.add_argument("--pg-port", type=int, default=int(os.getenv("PGPORT", "5432")))
    ap.add_argument("--pg-db", default=os.getenv("PGDATABASE", "cooja"))
    ap.add_argument("--pg-user", default=os.getenv("PGUSER", "cooja"))
    ap.add_argument("--pg-pass", default=os.getenv("PGPASSWORD", ""))
    args = ap.parse_args()

    meta = json.loads(Path(args.meta_json).read_text())
    roles = json.loads(Path(args.roles_json).read_text())
    topo = json.loads(Path(args.topology_json).read_text())
    load(
        run_dir=args.run_dir,
        meta=meta,
        topo=topo,
        pg={"host": args.pg_host, "port": args.pg_port, "db": args.pg_db, "user": args.pg_user, "password": args.pg_pass},
        role_map=roles,
    )



