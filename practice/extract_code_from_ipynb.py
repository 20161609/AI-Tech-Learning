import json

# Function to extract Python code from a Jupyter Notebook (.ipynb) file and save it as a .py file
def extract_code_from_ipynb(file_path, output_file):
    # Open the .ipynb file in read mode with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)  # Load the JSON content of the notebook
    
    all_code = []  # Initialize a list to store code from all code cells

    # Loop through all the cells in the notebook
    for cell in notebook.get('cells', []):
        # Check if the cell type is 'code'
        if cell.get('cell_type') == 'code':
            # Extract the source code from the cell
            code_lines = cell.get('source', [])
            # Join the code lines and append to the list
            all_code.append(''.join(code_lines))
    
    # Open the output file in write mode with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write all code to the output file, separating cells with two newlines
        f.write('\n\n'.join(all_code))
    
    # Print a confirmation message with the output file name
    print(f"Code extracted to {output_file}")

# Define the input .ipynb file path and the output .py file path
FILE_PATH = 'practice/KNN_Wine_Example_Practice.ipynb'  # Path to the Jupyter Notebook
OUTPUT_PATH = 'generated.py'  # Path for the generated Python file

# Call the function to extract code
extract_code_from_ipynb(FILE_PATH, OUTPUT_PATH)
