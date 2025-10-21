import json
from pathlib import Path
import subprocess


def test_generate_topology_and_csc(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    topo_json = tmp_path / "grid5.json"
    # Generate a tiny topology
    rc = subprocess.run([
        "python3", str(repo / "core" / "topology" / "topology_generator.py"),
        "grid", "-n", "5", "-s", "10", "-o", str(topo_json)
    ], capture_output=True, text=True)
    assert rc.returncode == 0
    topo = json.loads(topo_json.read_text())
    assert "motes" in topo and len(topo["motes"]) == 5
    assert topo["motes"][0]["role"].lower() == "server"

    # Generate CSC from topology
    csc_path = tmp_path / "sim.csc"
    rc2 = subprocess.run([
        "python3", str(repo / "core" / "csc" / "csc_generator.py"),
        str(topo_json), str(csc_path)
    ], capture_output=True, text=True)
    assert rc2.returncode == 0
    text = csc_path.read_text()
    assert "<simulation>" in text
    assert "Z1MoteType" in text or "MoteType" in text
    # add some more assertions and  as you see fit
    # for example, check that the CSC is valid XML...


