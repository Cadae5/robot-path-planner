import re

def load_gcode(filepath: str) -> list[dict]:
    """
    Loads a G-code file and parses G0/G1 commands to extract movement coordinates.
    Coordinates (X, Y, Z) can appear in any order.

    Args:
        filepath: The path to the G-code file.

    Returns:
        A list of dictionaries, where each dictionary represents a parsed G0 or G1
        command with its associated X, Y, Z coordinates.
        Example: [{'command': 'G1', 'x': 10.0, 'y': 5.0, 'z': 1.0}]
    """
    parsed_commands = []
    # Regex to capture the G0 or G1 command part
    command_regex = re.compile(r"^(G[01])", re.IGNORECASE)
    # Regex to find X, Y, Z parameters with their values, case insensitive
    param_regex_x = re.compile(r"X([+-]?\d*\.?\d+)", re.IGNORECASE)
    param_regex_y = re.compile(r"Y([+-]?\d*\.?\d+)", re.IGNORECASE)
    param_regex_z = re.compile(r"Z([+-]?\d*\.?\d+)", re.IGNORECASE)

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Remove comments (material from ';' onwards) and strip whitespace
                line_content = line.split(';', 1)[0].strip()

                if not line_content:  # Skip empty lines resulting from comments or blank lines
                    continue

                command_match = command_regex.match(line_content)
                if command_match:
                    command_str = command_match.group(1).upper()
                    command_data = {'command': command_str}

                    # The rest_of_line is everything after the G0/G1 command string
                    rest_of_line = line_content[len(command_str):].strip()

                    x_match = param_regex_x.search(rest_of_line)
                    if x_match:
                        command_data['x'] = float(x_match.group(1))

                    y_match = param_regex_y.search(rest_of_line)
                    if y_match:
                        command_data['y'] = float(y_match.group(1))

                    z_match = param_regex_z.search(rest_of_line)
                    if z_match:
                        command_data['z'] = float(z_match.group(1))

                    parsed_commands.append(command_data)
                # Optional: else print(f"Ignoring non-G0/G1 line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return [] # Or raise an exception
    except Exception as e:
        print(f"An error occurred while parsing G-code: {e}")
        return [] # Or raise an exception

    return parsed_commands
