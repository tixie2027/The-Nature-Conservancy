import os
from pathlib import Path
from docling.document_converter import DocumentConverter

# Source document path
def run(source, output_path):

    # Convert the document
    converter = DocumentConverter()
    result = converter.convert(source)
    md_content = result.document.export_to_markdown()

    # Get the directory where this Python file is located
    parent_dir = Path(__file__).parent.resolve()
    output_dir =  parent_dir / "finished_data"
    output_path = os.path.join(output_dir, "output.md")

    # Save the Markdown content to the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Markdown content saved to {output_path}")