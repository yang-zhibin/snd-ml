import json
from pathlib import Path
import shutil


def pretty_json(path):
    data = json.loads(Path(path).read_text())
    return json.dumps(data, indent=4, sort_keys=True).splitlines()


def side_by_side(lines1, lines2, label1="LAB", label2="TESTBEAM"):
    width = shutil.get_terminal_size((120, 20)).columns
    col_width = width // 2 - 2

    print(f"{label1:<{col_width}} | {label2:<{col_width}}")
    print("-" * width)

    max_lines = max(len(lines1), len(lines2))

    for i in range(max_lines):
        left = lines1[i] if i < len(lines1) else ""
        right = lines2[i] if i < len(lines2) else ""

        print(f"{left:<{col_width}.{col_width}} | {right:<{col_width}.{col_width}}")


def main():
    lab_file = "/eos/user/z/zhibin/TestBeam/run_000041/board_mapping.json"
    testbeam_file = "/eos/experiment/sndlhc/raw_data/testbeam_24/run_100890/board_mapping.json"
    lab_lines = pretty_json(lab_file)
    tb_lines = pretty_json(testbeam_file)

    side_by_side(lab_lines, tb_lines)


if __name__ == "__main__":
    main()




