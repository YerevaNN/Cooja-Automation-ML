-- PostgreSQL schema for parallel inserts (append-only workload)

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  topology_id TEXT NOT NULL,
  topology_type TEXT NOT NULL,
  topology_path TEXT NOT NULL,
  platform TEXT NOT NULL,
  duration_s INTEGER NOT NULL,
  seed INTEGER,
  started_at TIMESTAMPTZ NOT NULL,
  finished_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  node_count INTEGER NOT NULL,
  error TEXT
);

CREATE TABLE IF NOT EXISTS simulations (
  run_id TEXT NOT NULL,
  mote INTEGER NOT NULL,
  role TEXT,
  x DOUBLE PRECISION,
  y DOUBLE PRECISION,
  cpu INTEGER,
  lpm INTEGER,
  deep_lpm INTEGER,
  listen INTEGER,
  transmit INTEGER,
  off INTEGER,
  total INTEGER,
  initial_battery BIGINT,
  consumed BIGINT,
  remaining BIGINT,
  status TEXT,
  uptime INTEGER,
  last_msg_recv_by_root INTEGER,
  sent INTEGER,
  forwarded INTEGER,
  PRIMARY KEY (run_id, mote),
  FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_topology ON runs(topology_id);
CREATE INDEX IF NOT EXISTS idx_motes_run ON simulations(run_id);



