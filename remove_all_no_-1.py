import csv
import sys

input_file = sys.argv[1]  # Path to the input CSV file
output_file = sys.argv[2]  # Path to the output CSV file

with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write the header row to the output file
    headers = next(reader)
    writer.writerow(headers)
    
    # Write rows that contain '-1' in the distances column
    for row in reader:
        if '-1' in row[2]:
            writer.writerow(row)
