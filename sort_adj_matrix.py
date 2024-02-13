import sys

def sort_file_lines_by_length(input_file, output_file):
    """
    Read lines from an input file, sort them by length in descending order,
    and write the sorted lines to an output file.

    Args:
    input_file (str): The path to the file containing the original lines.
    output_file (str): The path to the file where sorted lines will be written.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()

    sorted_lines = sorted(lines, key=len, reverse=True)

    with open(output_file, 'w') as file:
        file.writelines(sorted_lines)

sort_file_lines_by_length(sys.argv[1], sys.argv[2])