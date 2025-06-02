import sys

def convert_tabs(filename, spaces_per_tab=2, fix=False):
    with open(filename, 'r') as f:
        lines = f.readlines()

    found_tabs = False
    new_lines = []
    for idx, line in enumerate(lines, 1):
        if '\t' in line:
            found_tabs = True
            print(f"Tab found on line {idx}: {line.rstrip()}")
            if fix:
                line = line.replace('\t', ' ' * spaces_per_tab)
        new_lines.append(line)

    if fix and found_tabs:
        with open(filename, 'w') as f:
            f.writelines(new_lines)
        print(f"\nAll tabs replaced with {spaces_per_tab} spaces in {filename}.")
    elif not found_tabs:
        print("No tabs found in the file.")
    else:
        print("\nRun with fix=True to replace tabs with spaces.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect and optionally fix tabs in a YAML file.")
    parser.add_argument("filename", help="YAML file to check")
    parser.add_argument("--fix", action="store_true", help="Replace tabs with spaces")
    parser.add_argument("--spaces", type=int, default=2, help="Number of spaces per tab (default: 2)")
    args = parser.parse_args()

    convert_tabs(args.filename, spaces_per_tab=args.spaces, fix=args.fix)