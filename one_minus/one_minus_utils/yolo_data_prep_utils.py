import pandas as pd
import os

# Set the directory where your annotation files are stored
directory = 'path/to/your/annotation/files'

# Class IDs to remove (replace with the correct IDs for 'fish' and 'boat')
class_ids_to_remove = {0, 1}  # Assuming 0 for fish and 1 for boat

# Process each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Keep lines that don't start with the specified class IDs
        new_lines = [line for line in lines if int(line.split()[0]) not in class_ids_to_remove]

        # Write the filtered lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(new_lines)

print("Annotation files have been updated.")
