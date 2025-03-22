**This repo is created by Harvey Mudd College students in collaboration with the Nature Conservancy 🍀**
- This repo is develop to turn TNC's PDF articles in JSON format, a machine-readable format
  - It is written in Python
  - **docling_test.py** performs **Optical Character Recognition" on the PDFs and turn them from PDFs to Markdown format during IBM's Docling package
  - **mdtojson.py** converts the Markdown format into better JSON format
  - **json_editor.py** fixes the Unicode issues that arise with the JSON encoding
  - **main.py** automate the whole process, and takes less than 1 minute to process one article
  - **json_heading_fix** converts the documents into a Introduction, Method, Results, Discussions, Tables, format. It standardizes the format of each article
  - **example_data.zip** contains the examples of PDFs to JSON conversion
  
**General Workflow**
![image](https://github.com/user-attachments/assets/99370082-66ec-4c12-ade6-42b52101e4f0)

