from main.components.menu_maneger.config.menu_maneger_config import pdf_source
import re, os
from main.functions.common_function import save_json_file
import pdfplumber
# obj = pdf_source()
# source = obj.create_source_dir()
# print(source['pdf_url'])

class pdf_extractor:
    def __init__(self):
        obj = pdf_source()
        self.source = obj.create_source_dir()
        self.pdf_dir = self.source['pdf_dir']
        self.pdf_url = self.source['pdf_url']



    def extract_pdf_text(self)-> dict:
        """
          Function to extract text from a pdf file
          Args:
                str (pdf): pdf file path
          Returns: josn file with extracted text
         """
        menu_items = []
        # categories = []
        with pdfplumber.open(self.pdf_url) as pdf_filess:
            for pages in pdf_filess.pages:
                text = pages.extract_text()
                text_new_lines = text.split('\n')
                for line in text_new_lines:
                    match = re.search(r'^(.+?)\s+(\d{3,4})$', line.strip())
                    if match:
                        item_name = match.group(1).strip()
                        price = match.group(2).strip()
                        description_line = text_new_lines[text_new_lines.index(line) + 1]
    
                        menu_items.append({
                            'item_name': item_name,
                            'price': price,
                            'description': description_line.strip()
                        })
        # json_file_menu = save_json_file(path= self.pdf_dir, data=menu_items)
        output_file_path = os.path.join(self.pdf_dir, "menu_data.json")
        json_file_menu = save_json_file(path=output_file_path, data=menu_items)

        return json_file_menu


    def review_and_save(self, data: list, filename:)
  
# Create object and call extraction
pdf_extractor_obj = pdf_extractor()
print(pdf_extractor_obj.extract_pdf_text())