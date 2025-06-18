import os
import re
from pathlib import Path

def clean_filename(filename):
    """Clean filename by removing special characters and spaces."""
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r'[^\w\s-]', '', filename)
    cleaned = re.sub(r'[-\s]+', '_', cleaned)
    return cleaned.lower()

def extract_document_info(content):
    """Extract document title and create filename from content."""
    lines = content.strip().split('\n')
    
    # Find the title line (starts with # Document)
    title_line = ""
    for line in lines:
        if line.startswith('# Document'):
            title_line = line
            break
    
    if title_line:
        # Extract just the title part after "Document X: "
        match = re.search(r'# Document \d+:\s*(.+)', title_line)
        if match:
            title = match.group(1).strip()
            return title, clean_filename(title)
    
    # Fallback: use first non-empty line or generic name
    for line in lines:
        if line.strip() and not line.startswith('#'):
            return line.strip(), clean_filename(line.strip())
    
    return "untitled", "untitled"

def split_documents(input_file='data.md', output_dir='data'):
    """
    Split a markdown file into separate documents based on ----- separator.
    
    Args:
        input_file (str): Path to the input markdown file
        output_dir (str): Directory to save the split documents
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Split by ----- separator
    documents = content.split('-----')
    
    # Remove empty documents and process each one
    document_count = 0
    
    for i, doc_content in enumerate(documents):
        # Skip empty or whitespace-only documents
        if not doc_content.strip():
            continue
        
        document_count += 1
        
        # Extract document info
        title, filename_base = extract_document_info(doc_content)
        
        # Create filename with document number for uniqueness
        filename = f"{document_count:02d}_{filename_base}.md"
        filepath = os.path.join(output_dir, filename)
        
        # Clean up the content (remove extra whitespace at start/end)
        cleaned_content = doc_content.strip()
        
        # Write the document to file
        try:
            with open(filepath, 'w', encoding='utf-8') as output_file:
                output_file.write(cleaned_content)
            
            print(f"Created: {filepath}")
            print(f"  Title: {title}")
            
        except Exception as e:
            print(f"Error writing file {filepath}: {e}")
    
    print(f"\nSplit complete! Created {document_count} documents in '{output_dir}' directory.")

def main():
    """Main function to run the document splitter."""
    print("Document Splitter")
    print("=================")
    
    # You can modify these paths as needed
    input_file = 'data.md'
    output_dir = 'data'
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()
    
    split_documents(input_file, output_dir)

if __name__ == "__main__":
    main()