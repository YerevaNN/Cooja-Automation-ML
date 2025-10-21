#!/usr/bin/env python3
"""
RPL Log to CSV Parser - Version 2 (extended)
Adds two columns per mote:
- sent: number of application messages originated by this mote (client-side logs)
- forwarded: number of packets this mote forwarded as the last hop to the root
  (derived from server logs: 'via neighbor=<mote_id>' and originator != neighbor)
"""

import argparse
import csv
import re
from collections import defaultdict

def parse_simulation_time_to_seconds(time_str):
    """
    Parse Cooja simulation time format (MM:SS.mmm) to total seconds
    Examples: 
    - "02:46.219" -> 166.219 seconds -> 166 (integer)
    - "01:10.192" -> 70.192 seconds -> 70 (integer)
    """
    try:
        if ':' in time_str and '.' in time_str:
            parts = time_str.split(':')
            minutes = int(parts[0])
            seconds_parts = parts[1].split('.')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1])
            total_seconds = minutes * 60 + seconds + milliseconds / 1000.0
            return int(total_seconds)  # Return as integer seconds
    except:
        pass
    return None

def process_file(input_log, output_csv):
    """
    Processes the input log file and writes ONE ROW PER MOTE with final state.
    Handles ENERGEST, BATTERY, message reception logs, plus per-mote 'sent' and 'forwarded'.
    """
    
    # Regex patterns for different log types
    # Cooja format: "02:46.219	ID:2	[INFO: Module] message"
    time_node_re = re.compile(r'^(\d+:\d+\.\d+)\s+ID:(\d+)\s+\[([^\]]+)\]\s+(.*)')
    
    # ENERGEST log pattern (from clients and server)
    energest_re = re.compile(
        r'ENERGEST: CPU=(\d+)s LPM=(\d+)s DEEP_LPM=(\d+)s LISTEN=(\d+)s TRANSMIT=(\d+)s OFF=(\d+)s TOTAL=(\d+)s'
    )
    
    # BATTERY log pattern (from both clients and server)
    battery_re = re.compile(
        r'BATTERY: initial=(\d+)mAh consumed=(\d+)mAh remaining=(\d+)mAh status=(\w+) uptime=(\d+)s'
    )
    
    # Server message reception pattern
    # "[1] Received msg seq=5 from originator=24 via neighbor=24 (total_received=212)"
    server_receive_re = re.compile(
        r'\[(\d+)\] Received msg seq=(\d+) from originator=(\d+) via neighbor=(\d+)'
    )

    # Client-originated send pattern
    # "[<id>] Sending msg seq=X to root"
    client_send_re = re.compile(
        r'Sending msg seq=(\d+)\s+to root'
    )

    # Client IP statistics pattern from battery_client logs
    # "IPSTATS: sent=XX recv=YY fwd=ZZ drop=WW"
    ipstats_re = re.compile(
        r'IPSTATS:\s+sent=(\d+)\s+recv=(\d+)\s+fwd=(\d+)\s+drop=(\d+)'
    )
    
    # Data structures to track state PER MOTE
    mote_data = {}  # mote_id -> {latest state data}
    last_message_time = {}  # mote_id -> last time (in seconds) server received message

    # Helper to ensure mote_data initialized with counters
    def ensure_mote(mote_id, sim_time_str=None, sim_time_seconds=None):
        if mote_id not in mote_data:
            mote_data[mote_id] = {
                'mote': mote_id,
                'time': sim_time_str if sim_time_str else '',
                'time_seconds': sim_time_seconds if sim_time_seconds is not None else None,
                'cpu': '',
                'lpm': '',
                'deep_lpm': '',
                'listen': '',
                'transmit': '',
                'off': '',
                'total': '',
                'initial_battery': '',
                'consumed': '',
                'remaining': '',
                'status': '',
                'uptime': '',
                'last_msg_recv_by_root': '',
                'sent': 0,
                'forwarded': 0,
            }

    print("Processing log file...")
    line_count = 0
    
    with open(input_log, 'r') as infile:
        for line in infile:
            line_count += 1
            
            # Parse the Cooja log line format
            time_node_match = time_node_re.match(line)
            if not time_node_match:
                continue
                
            sim_time_str = time_node_match.group(1)
            mote_id = time_node_match.group(2)
            log_level = time_node_match.group(3)
            message = time_node_match.group(4)
            
            sim_time_seconds = parse_simulation_time_to_seconds(sim_time_str)

            # Make sure this mote has an entry
            ensure_mote(mote_id, sim_time_str, sim_time_seconds)
            mote_data[mote_id]['time'] = sim_time_str
            mote_data[mote_id]['time_seconds'] = sim_time_seconds

            # Check for client-originated send logs
            # We rely on client message: "Sending msg seq=X to root"
            client_send_match = client_send_re.search(message)
            if client_send_match:
                mote_data[mote_id]['sent'] = mote_data[mote_id].get('sent', 0) + 1
                # no 'continue' so we still allow energest/battery parsing on the same line if ever present

            # Prefer forwarded from IP statistics when available
            ipstats_match = ipstats_re.search(message)
            if ipstats_match:
                try:
                    mote_data[mote_id]['forwarded'] = int(ipstats_match.group(3))
                except Exception:
                    pass

            # Check for ENERGEST logs
            energest_match = energest_re.search(message)
            if energest_match:
                mote_data[mote_id].update({
                    'cpu': energest_match.group(1),
                    'lpm': energest_match.group(2),
                    'deep_lpm': energest_match.group(3),
                    'listen': energest_match.group(4),
                    'transmit': energest_match.group(5),
                    'off': energest_match.group(6),
                    'total': energest_match.group(7)
                })
                continue
            
            # Check for BATTERY logs
            battery_match = battery_re.search(message)
            if battery_match:
                mote_data[mote_id].update({
                    'initial_battery': battery_match.group(1),
                    'consumed': battery_match.group(2),
                    'remaining': battery_match.group(3),
                    'status': battery_match.group(4),
                    'uptime': battery_match.group(5)
                })
                
                # Add last message received time if we have it
                if mote_id in last_message_time:
                    mote_data[mote_id]['last_msg_recv_by_root'] = last_message_time[mote_id]
                
                continue
            
            # Check for server receiving messages
            server_receive_match = server_receive_re.search(message)
            if server_receive_match and 'Server' in log_level:
                # Fields: [server_id] seq, originator, neighbor(last hop)
                originator_id = server_receive_match.group(3)
                neighbor_id = server_receive_match.group(4)

                # Update last message time for the originator
                last_message_time[originator_id] = sim_time_seconds if sim_time_seconds else 0
                
                # Ensure originator and neighbor entries exist
                ensure_mote(originator_id)
                ensure_mote(neighbor_id)

                # Update CSV data for originator with last message time
                mote_data[originator_id]['last_msg_recv_by_root'] = last_message_time[originator_id]

                # Do not infer forwarding from server logs: sender address equals originator here.
                
                continue
    
    print(f"Processed {line_count} lines")
    print(f"Found data for {len(mote_data)} motes")
    
    # Write to CSV - ONE ROW PER MOTE
    with open(output_csv, 'w', newline='') as outfile:
        fieldnames = [
            'mote',  # Mote ID
            'cpu', 'lpm', 'deep_lpm', 'listen', 'transmit', 'off', 'total',  # Energest
            'initial_battery', 'consumed', 'remaining', 'status', 'uptime',  # Battery
            'last_msg_recv_by_root',  # Last time server received msg (in seconds)
            'sent', 'forwarded'  # New: per-mote counts
        ]
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort by mote ID for consistent output
        for mote_id in sorted(mote_data.keys(), key=lambda x: int(x)):
            row = mote_data[mote_id]
            # Only include the fields we want in the CSV
            csv_row = {field: row.get(field, '') for field in fieldnames}
            writer.writerow(csv_row)
    
    print(f"Wrote {len(mote_data)} rows to {output_csv}")
    
    # Summary statistics
    if mote_data:
        # Count final status
        dead_nodes = sum(1 for m in mote_data.values() if m['status'] == 'dead')
        alive_nodes = sum(1 for m in mote_data.values() if m['status'] == 'alive')
        unknown_nodes = len(mote_data) - dead_nodes - alive_nodes
        
        print(f"\nFinal status summary:")
        print(f"  Alive: {alive_nodes}")
        print(f"  Dead: {dead_nodes}")
        if unknown_nodes > 0:
            print(f"  Unknown: {unknown_nodes}")
        
        # Check which nodes never got messages through
        motes_with_msgs = set(str(k) for k in last_message_time.keys())
        all_motes = set(mote_data.keys())
        no_msgs = all_motes - motes_with_msgs - {'1'}  # Exclude root node
        if no_msgs:
            print(f"\nMotes that never got messages to root: {sorted(no_msgs, key=int)}")
        
        # Find last simulation time
        last_times = [m.get('time_seconds', 0) for m in mote_data.values() if m.get('time_seconds') is not None]
        if last_times:
            max_time = max(last_times)
            print(f"\nSimulation duration: {max_time} seconds ({max_time//60}m {max_time%60}s)")

def main():
    parser = argparse.ArgumentParser(
        description='Process RPL simulation logs into CSV with energy, battery, last message time, and per-mote sent/forwarded counts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 %(prog)s simulation.log results.csv

The script extracts:
  - ENERGEST statistics (CPU, LPM, radio usage)
  - Battery status (initial, consumed, remaining)
  - Last message received time at the server for each mote
  - Per-mote counts: 'sent' (originations) and 'forwarded' (last-hop to root)
        """
    )
    
    parser.add_argument('input_log', type=str, 
                       help='Path to the input log file from Cooja simulation')
    parser.add_argument('output_csv', type=str, 
                       help='Path to the output CSV file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output for debugging')
    parser.add_argument('--show-intermediate', action='store_true',
                       help='Show intermediate states (for debugging)')

    args = parser.parse_args()
    
    if args.verbose:
        print(f"Processing: {args.input_log} -> {args.output_csv}")
    
    try:
        process_file(args.input_log, args.output_csv)
        print(f"\nSuccessfully wrote results to {args.output_csv}")
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()