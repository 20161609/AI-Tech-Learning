import json

# Function to extract Python code and text (comments) from a Jupyter Notebook (.ipynb) file and save them as a .py file
def extract_content_from_ipynb(file_path, output_file):
    # Open the .ipynb file in read mode with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)  # Load the JSON content of the notebook

    all_content = []  # Initialize a list to store all content (text and code)

    # Loop through all the cells in the notebook
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type')  # Get the type of the cell
        if cell_type == 'markdown':  # For markdown cells
            text_lines = cell.get('source', [])
            # Add the text as comments to the content list
            all_content.append('# ' + '\n# '.join(''.join(text_lines).splitlines()))
        elif cell_type == 'raw':  # For raw cells
            raw_lines = cell.get('source', [])
            # Add the raw text as comments to the content list
            all_content.append('# ' + '\n# '.join(''.join(raw_lines).splitlines()))
        elif cell_type == 'code':  # For code cells
            code_lines = cell.get('source', [])
            # Add the code directly to the content list
            all_content.append(''.join(code_lines))

    # Open the output file in write mode with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write all content to the output file, separating cells with two newlines
        f.write('\n\n'.join(all_content))

    # Print a confirmation message with the output file name
    print(f"Content extracted to {output_file}")

# Define the input .ipynb file path and the output .py file path
FILE_PATH = '20_cnn_bin_cats_vs_dogs_augmentation.ipynb'  # Path to the Jupyter Notebook
OUTPUT_PATH = 'generated_with_comments.py'  # Path for the generated Python file

# Call the function to extract content
extract_content_from_ipynb(FILE_PATH, OUTPUT_PATH)
