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
        raw_menu_items = []
        # categories = []
        with pdfplumber.open(self.pdf_url) as pdf_filess:
            # for pages in pdf_filess.pages:
                pages = pdf_filess.pages[3]
                text = pages.extract_text()
                text_new_lines = text.split('\n')
                for line in text_new_lines:
                    match = re.search(r'^(.+?)\s+(\d{3,4})$', line.strip())
                    if match:
                        item_name = match.group(1).strip()
                        price = match.group(2).strip()
                        description_line = text_new_lines[text_new_lines.index(line) + 1]
    
                        raw_menu_items.append({
                            'item_name': item_name,
                            'price': price,
                            'description': description_line.strip()
                        })
   
        output_file_path = os.path.join(self.pdf_dir, "raw_menu_data.json")
        save_json_file(path=output_file_path, data=raw_menu_items)

        return raw_menu_items
    
    
    
    def review_json(self, raw_menu_items: list, file_name:str = 'revwied_menu_data.json'):
        print('Review the JSON data before finalizing')

        reviewed_data = []
    
        for idx, items in enumerate(raw_menu_items):
            print('---------------------------------')
            print(f"[{idx+1}] Item Name      : {items['item_name']}")
            print(f"    Price          : {items['price']}")
            print(f"    Description    : {items['description']}")
            print('---------------------------------')
        
            action = input("Is this correct? (y/n): or press 'd' to delete or 'all' to save raw json file   ➤ ").strip().lower()
        
            if action == 'y':
                print("✅ Item confirmed.")
                reviewed_data.append(items)
        
            elif action == 'n':
                print("✏️  Editing the item.")
                items['item_name'] = input(f"    Edit Item Name [{items['item_name']}]: ") or items['item_name']
                items['price'] = input(f"    Edit Price [{items['price']}]: ") or items['price']
                items['description'] = input(f"    Edit Description [{items['description']}]: ") or items['description']
                reviewed_data.append(items)
        
            elif action == 'd':
                print("🗑️  Item deleted.")
                # Do not add to reviewed_data

            elif action == 'all':
                break

            else:
                print("⚠️  Invalid input. Please enter 'y', 'n', or 'd'")
                # Optionally re-prompt here if you want strict control
        
        output_file_path = os.path.join(self.pdf_dir, "reviewed_menu_data.json")
        if not reviewed_data:
            print("❌ No items were confirmed. Exiting...")
            save_json_file(path=output_file_path, data= raw_menu_items)

            return raw_menu_items
        
        save_json_file(path=output_file_path, data= reviewed_data)
        return reviewed_data


  
# # Create object and call extraction
# pdf_extractor_obj = pdf_extractor()
# print(pdf_extractor_obj.extract_pdf_text())