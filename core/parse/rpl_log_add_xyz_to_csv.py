#!/usr/bin/env python3
import argparse
import csv
import sys
import os
import xml.etree.ElementTree as ET


def extract_positions_from_csc(csc_path):
    """
    Return dict[int, (float, float, float|None)] mapping mote id -> (x, y, z?) from a Cooja .csc file.
    If <z> is absent, z is None.
    """
    tree = ET.parse(csc_path)
    root = tree.getroot()
    positions = {}

    for mote in root.findall(".//simulation/mote"):
        mote_id = None
        x = None
        y = None
        z = None

        for ic in mote.findall("./interface_config"):
            x_el = ic.find("x")
            y_el = ic.find("y")
            z_el = ic.find("z")
            id_el = ic.find("./{*}id") or ic.find("id")

            if x_el is not None and y_el is not None:
                try:
                    x = float(x_el.text.strip())
                    y = float(y_el.text.strip())
                except Exception:
                    pass
            if z_el is not None and z_el.text is not None:
                try:
                    z = float(z_el.text.strip())
                except Exception:
                    z = None
            if id_el is not None and id_el.text is not None:
                try:
                    mote_id = int(id_el.text.strip())
                except Exception:
                    pass

        if mote_id is not None and x is not None and y is not None:
            if mote_id in positions:
                print(f"Warning: duplicate position for mote {mote_id}; keeping first", file=sys.stderr)
                continue
            positions[mote_id] = (x, y, z)
        else:
            print(f"Warning: incomplete data for a mote (id={mote_id}, x={x}, y={y}); skipping", file=sys.stderr)

    return positions


def determine_output_path(input_path, output_arg):
    if output_arg:
        return output_arg
    base, ext = os.path.splitext(input_path)
    return f"{base}_with_xy{ext or '.csv'}"


def main():
    ap = argparse.ArgumentParser(
        description="Add x,y (and optional z) coordinates from a Cooja .csc topology into a results CSV."
    )
    ap.add_argument("--grid", required=True, help="Path to Cooja .csc file")
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", default=None, help="Path to output CSV; defaults to <input>_with_xy.csv")
    ap.add_argument("--id-column", default="mote", help="CSV column name containing mote IDs (default: mote)")
    ap.add_argument("--x-col-name", default="x", help="Name for the x column (default: x)")
    ap.add_argument("--y-col-name", default="y", help="Name for the y column (default: y)")
    ap.add_argument("--enable-z", action="store_true", help="If set, also attach z coordinate when present in CSC")
    ap.add_argument("--z-col-name", default="z", help="Name for the z column when --enable-z is used (default: z)")
    args = ap.parse_args()

    # Load positions from .csc
    id_to_xyz = extract_positions_from_csc(args.grid)
    if not id_to_xyz:
        print("Error: no positions found in CSC; check the file.", file=sys.stderr)
        sys.exit(1)

    # Read input CSV, sniff dialect for robustness
    with open(args.input, "r", newline="") as fin:
        sample = fin.read(4096)
        fin.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(fin, dialect=dialect)
        if reader.fieldnames is None:
            print("Error: input CSV has no header.", file=sys.stderr)
            sys.exit(1)

        if args.id_column not in reader.fieldnames:
            print(f"Error: id column '{args.id_column}' not found in CSV headers: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        # Build output headers; insert x,y (and z) right after id column if possible
        out_fields = []
        inserted = False
        for f in reader.fieldnames:
            out_fields.append(f)
            if f == args.id_column:
                if args.x_col_name not in reader.fieldnames:
                    out_fields.append(args.x_col_name)
                if args.y_col_name not in reader.fieldnames:
                    out_fields.append(args.y_col_name)
                if args.enable_z and args.z_col_name not in reader.fieldnames:
                    out_fields.append(args.z_col_name)
                inserted = True
        if not inserted:
            if args.x_col_name not in out_fields:
                out_fields.append(args.x_col_name)
            if args.y_col_name not in out_fields:
                out_fields.append(args.y_col_name)
            if args.enable_z and args.z_col_name not in out_fields:
                out_fields.append(args.z_col_name)

        out_path = determine_output_path(args.input, args.output)
        with open(out_path, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields, dialect=dialect)
            writer.writeheader()

            missing = set()
            for row in reader:
                raw_id = row.get(args.id_column, "").strip()
                try:
                    mote_id = int(raw_id)
                except Exception:
                    print(f"Warning: skipping row with non-integer {args.id_column}='{raw_id}'", file=sys.stderr)
                    mote_id = None

                if mote_id is not None and mote_id in id_to_xyz:
                    x, y, z = id_to_xyz[mote_id]
                    row[args.x_col_name] = x
                    row[args.y_col_name] = y
                    if args.enable_z:
                        row[args.z_col_name] = (z if z is not None else "")
                else:
                    row[args.x_col_name] = ""
                    row[args.y_col_name] = ""
                    if args.enable_z:
                        row[args.z_col_name] = ""
                    if mote_id is not None:
                        missing.add(mote_id)

                writer.writerow(row)

        if missing:
            print(f"Note: {len(missing)} mote(s) in CSV missing from CSC: {sorted(missing)}", file=sys.stderr)

        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


