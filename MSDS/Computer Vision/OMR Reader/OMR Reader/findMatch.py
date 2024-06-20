import sys


if __name__ == '__main__':

    if(len(sys.argv) < 2):
        raise Exception("error: Please give an output file name which has student answers, correct answers recognized , like this: \n"
                     "python3 ./findMatch.py output.txt groundtruth.txt")
 
# Define the paths to the two files you want to compare
file_path1 = sys.argv[1]
file_path2 = sys.argv[2]

# Initialize a counter for the number of lines that do not match
non_matching_lines_count = 0

# Open both files and compare them line by line
with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
    for line1, line2 in zip(file1, file2):
        # Increment the counter if the lines do not match
        if line1 != line2:
            non_matching_lines_count += 1

# If the files have different numbers of lines, count the remaining lines in the longer file
with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()

# Add the difference in line counts to the non-matching lines count
non_matching_lines_count += abs(len(lines1) - len(lines2))

# Print the result
print(f"Diff in no of lines for ground truth and detected data : {non_matching_lines_count}")
print(f"Accuracy of the program: {(1-(non_matching_lines_count/len(lines1)))*100}")
