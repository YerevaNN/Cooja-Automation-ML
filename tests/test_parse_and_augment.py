from pathlib import Path
import subprocess


SAMPLE_LOG = """
00:00.100	ID:1	[INFO: Server]	ENERGEST: CPU=0s LPM=0s DEEP_LPM=0s LISTEN=0s TRANSMIT=0s OFF=0s TOTAL=0s
00:01.000	ID:2	[INFO: Client]	Sending msg seq=1 to root
00:01.050	ID:1	[INFO: Server]	[1] Received msg seq=1 from originator=2 via neighbor=2
00:02.000	ID:2	[INFO: Client]	BATTERY: initial=50mAh consumed=1mAh remaining=49mAh status=alive uptime=2s
"""


def test_parse_and_augment(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    # Write sample log
    log_path = tmp_path / "cooja.log"
    log_path.write_text(SAMPLE_LOG.strip())

    # Parse to CSV
    csv_path = tmp_path / "results.csv"
    rc = subprocess.run([
        "python3", str(repo / "core" / "parse" / "rpl_log_to_csv_v3.py"),
        str(log_path), str(csv_path)
    ], capture_output=True, text=True)
    assert rc.returncode == 0
    assert csv_path.exists() and csv_path.read_text().strip().splitlines()[0].startswith("mote,")

    # Create tiny CSC with 2 motes
    csc_path = tmp_path / "sim.csc"
    csc_text = """
<simconf>
  <simulation>
    <title>t</title>
    <randomseed>1</randomseed>
  </simulation>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>0.0</x>
      <y>0.0</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>1</id>
    </interface_config>
    <motetype_identifier>z1_server</motetype_identifier>
  </mote>
  <mote>
    <interface_config>
      org.contikios.cooja.interfaces.Position
      <x>10.0</x>
      <y>0.0</y>
      <z>0.0</z>
    </interface_config>
    <interface_config>
      org.contikios.cooja.mspmote.interfaces.MspMoteID
      <id>2</id>
    </interface_config>
    <motetype_identifier>z1_client</motetype_identifier>
  </mote>
</simconf>
"""
    csc_path.write_text(csc_text.strip())

    # Augment
    out_csv = tmp_path / "results_xy.csv"
    rc2 = subprocess.run([
        "python3", str(repo / "core" / "parse" / "rpl_log_add_xyz_to_csv.py"),
        "--grid", str(csc_path),
        "--input", str(csv_path),
        "--output", str(out_csv)
    ], capture_output=True, text=True)
    assert rc2.returncode == 0
    content = out_csv.read_text().strip().splitlines()
    assert content[0].startswith("mote,") and ",x," in content[0] and ",y" in content[0]


