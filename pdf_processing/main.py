import docling_test
from pathlib import Path
import mdtojson
import json_editor
import os
import json

def main():
    parent_dir = Path(__file__).parent.resolve()
    training_dir = parent_dir / "training_data"
    output_dir =  parent_dir / "finished_data"
    print(parent_dir)
    print(training_dir)
    print(output_dir)

    for filename in os.listdir(training_dir):

        file_path = os.path.join(training_dir, filename)
        print(file_path)
        docling_test.run(file_path, output_dir)


        parent_dir = Path(__file__).parent.resolve()
        current_directory =  parent_dir / "finished_data"
        md_file = os.path.join(current_directory, "output.md")
        output_filename = f"{os.path.splitext(filename)[0]}.json"
        json_file = os.path.join(current_directory, output_filename)
        mdtojson.convert_md_to_json(md_file, json_file)


        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        cleaned_data = json_editor.clean_json(data)
        cleaned_filename = f"cleaned_{os.path.splitext(filename)[0]}.json"

        cleaned_file_path = os.path.join(output_dir, cleaned_filename)

        with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
            json.dump(cleaned_data, cleaned_file, indent=4, ensure_ascii=False)
        print("Cleaning complete! Cleaned file saved as", cleaned_filename) 

        

if __name__ == "__main__":
    main()
