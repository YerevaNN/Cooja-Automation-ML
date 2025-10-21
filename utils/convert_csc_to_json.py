import xml.etree.ElementTree as ET
import json
import argparse
import sys


# Usage
# to print JSON to console
# python convert_csc_to_json.py battery_sim.csc

# to save JSON to a file
# python convert_csc_to_json.py battery_sim.csc --output battery_sim.json

# this is just a utility that converts a .csc file to a JSON file
# purposes can be readability, compression, etc.

def parse_element_to_dict(element):
    """Recursively parse an XML element and its children into a dictionary."""
    # Handle simple text-only elements
    if not list(element) and element.text and element.text.strip():
        # Try to convert to float/int if possible
        text = element.text.strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    # Handle elements with attributes and/or children
    d = {child.tag: parse_element_to_dict(child) for child in element}
    if element.attrib:
        d.update(('@' + k, v) for k, v in element.attrib.items())
    
    # If element has text and children, store text under a special key
    if element.text and element.text.strip():
        d['#text'] = element.text.strip()
        
    return d

def simplify_class_path(path):
    """Extracts the class name from a Java-style path."""
    if path and '.' in path:
        return path.split('.')[-1]
    return path

def compress_cooja_config(root):
    """
    Converts the parsed XML tree into a compressed and structured JSON format.
    """
    output = {
        "simulation": {},
        "mote_types": [],
        "motes": [],
        "plugins": {}
    }

    # --- 1. Parse Simulation Details ---
    sim_node = root.find('simulation')
    if sim_node is not None:
        for child in sim_node:
            if child.tag in ['title', 'randomseed', 'motedelay_us']:
                output['simulation'][child.tag] = child.text.strip()
            elif child.tag == 'radiomedium':
                rm_type = simplify_class_path(child.text.strip())
                output['simulation']['radiomedium'] = {'type': rm_type}
                for prop in child:
                    output['simulation']['radiomedium'][prop.tag] = float(prop.text)
            elif child.tag == 'events':
                 output['simulation']['events'] = {
                     sub.tag: int(sub.text) for sub in child
                 }

    # --- 2. Parse Mote Types ---
    mote_types_map = {}
    for mt_node in sim_node.findall('motetype'):
        mote_type = {}
        identifier = mt_node.find('identifier')
        mote_type['id'] = identifier.text.strip() if identifier is not None else None
        
        for child in mt_node:
            if child.tag in ['description', 'source', 'commands']:
                mote_type[child.tag] = child.text.strip()
            elif child.tag == 'moteinterface':
                if 'interfaces' not in mote_type:
                    mote_type['interfaces'] = []
                mote_type['interfaces'].append(simplify_class_path(child.text.strip()))
        
        output['mote_types'].append(mote_type)
        if mote_type['id']:
            mote_types_map[mote_type['id']] = mote_type

    # --- 3. Parse Motes ---
    # Motes can be defined at the top level or inside a motetype
    all_mote_nodes = sim_node.findall('mote')
    for mt_node in sim_node.findall('motetype'):
        all_mote_nodes.extend(mt_node.findall('mote'))

    for mote_node in all_mote_nodes:
        mote = {}
        for config in mote_node.findall('interface_config'):
            class_path = config.text.strip()
            if 'Position' in class_path:
                pos_node = config.find('pos') or config # Handle both <pos> and direct x/y
                mote['position'] = {
                    'x': float(pos_node.get('x', pos_node.find('x').text)),
                    'y': float(pos_node.get('y', pos_node.find('y').text))
                }
            elif 'ContikiMoteID' in class_path:
                mote['id'] = int(config.find('id').text)
            elif 'ContikiRadio' in class_path:
                mote['radio_bitrate'] = float(config.find('bitrate').text)
        
        type_id_node = mote_node.find('motetype_identifier')
        if type_id_node is not None:
            mote['type'] = type_id_node.text.strip()
            
        output['motes'].append(mote)

    # --- 4. Parse Plugins ---
    for plugin_node in root.findall('plugin'):
        plugin_name = simplify_class_path(plugin_node.text.strip())
        plugin_config = {}
        config_node = plugin_node.find('plugin_config')
        if config_node is not None:
            for child in config_node:
                # Handle simple flags
                if child.text is None or not child.text.strip():
                    plugin_config[child.tag] = True
                # Handle key-value pairs
                else:
                    plugin_config[child.tag] = child.text.strip()
            
            # Special handling for Visualizer skins
            skins = config_node.findall('skin')
            if skins:
                plugin_config['skins'] = [simplify_class_path(s.text.strip()) for s in skins]
                if 'skin' in plugin_config: del plugin_config['skin']

        bounds_node = plugin_node.find('bounds')
        if bounds_node is not None:
            plugin_config['bounds'] = {k: int(v) for k, v in bounds_node.attrib.items()}

        output['plugins'][plugin_name] = plugin_config

    return output

def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(
        description="Compress a Cooja .csc (XML) file into a structured JSON format for LLM context.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to the input .csc file."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        help="Path to the output .json file. If not provided, prints to standard output."
    )
    parser.add_argument(
        "-i", "--indent",
        type=int,
        default=2,
        help="Indentation level for the output JSON. Default is 2."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    try:
        tree = ET.parse(args.input_file)
        root = tree.getroot()
        compressed_data = compress_cooja_config(root)
        
        json_output = json.dumps(compressed_data, indent=args.indent)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(json_output)
            print(f"Successfully converted '{args.input_file}' to '{args.output_file}'")
        else:
            print(json_output)

    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error: Failed to parse XML file '{args.input_file}'.\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()