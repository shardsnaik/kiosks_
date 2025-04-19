from main.components.menu_maneger.config.menu_maneger_config import pdf_source
import re, os, json
from main.functions.common_function import save_json_file
import pdfplumber
from typing import List, Dict, Any
from dotenv import load_dotenv 
import groq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document    
import pytesseract
from transformers import LayoutLMv3Processor
from PIL import Image

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
groq_client = groq.Client(api_key=groq_api_key)


class pdf_extractor:
    def __init__(self):
        obj = pdf_source()
        self.source = obj.create_source_dir()
        self.pdf_dir = self.source['pdf_dir']
        self.pdf_url = self.source['pdf_url']
        self.extracted_menu_dir = self.source['extracted_menu']



    def manual_extract_pdf_text(self)-> dict:
        """
          Function to extract text from a pdf file
          Args:
                str (pdf): pdf file path
          Returns: josn file with extracted text
         """
        raw_menu_items = []
        # categories = []
        with pdfplumber.open(self.pdf_url) as pdf_filess:
            for pages in pdf_filess.pages:
                # pages = pdf_filess.pages[3]
                text = pages.extract_text()
                text_new_lines = text.split('\n')
                for line in text_new_lines:
                    match = re.search(r'^(.+?)\s+(\d{2,5}(?:\.\d{2})?)$', line.strip())
                    if match:
                        item_name = match.group(1).strip()
                        price = match.group(2).strip()
                        description_line = text_new_lines[text_new_lines.index(line) + 1]
                        if not description_line:
                            description_line = ""
                            if i + 1 < len(line):
                                next_line = line[i + 1].strip()
                                next_match = re.match(r'^(.+?)\s+(\d{2,5}(?:\.\d{2})?)$', next_line)
                                if not next_match:
                                    description_line = next_line
                                    i += 1  # skip the description line in next iteration
        
            
                        raw_menu_items.append({
                            'item_name': item_name,
                            'price': price,
                            'description': description_line.strip()
                        })
   
        output_file_path = os.path.join(self.extracted_menu_dir, "raw_menu_data.json")
        save_json_file(path=output_file_path, data=raw_menu_items)

        return output_file_path
    

    def human_in_loop_manual_extractor(self, raw_menu_items: json, file_name:str = 'api_menu_extractir.json'):
        print('Review the JSON data before finalizing')

        reviewed_data = []
        with open(raw_menu_items , 'r') as f:
            raw_menu_items = json.load(f)
    
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
        
        output_file_path = os.path.join(self.extracted_menu_dir, "reviewed_menu_data.json")
        if not reviewed_data:
            print("❌ No items were confirmed. Exiting...")
            save_json_file(path=output_file_path, data= raw_menu_items)

            return raw_menu_items
        
        save_json_file(path=output_file_path, data= reviewed_data)
        return reviewed_data
    
    
# code for extracting the menu using the LLM (API)
# Need to call seperate Human-In_loop function for this abpve LLM function because of it handle double for loop (ie categories)



    def load_and_process_docs(self, file_) -> List[Document]:
        '''
        Load document from the specified directory

        split document into chuck
        '''
        all_docs = []
        if file_.endswith('.pdf'):
            loader = PyPDFLoader(file_)
            documnets = loader.load()
            all_docs.extend(documnets)
            document = all_docs
        
        elif file_.endswith('.txt'):
            loader = TextLoader(file_)
            documnets = loader.load()
            all_docs.extend(documnets)
            document = all_docs           
        
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 50
        )
        chunk = text_spliter.split_documents(document)
        print(f'Split into {len(chunk)}')
        self.vectorstore = FAISS.from_documents(chunk, self.embeddings )
        print('Vector store created successfully')


    def retrival(self, query: str, k: int= 5)->List[Document]:
        '''
        Retrieve relavent document for a given query
        '''
        if not self.vectorstore:
            raise ValueError('Vector store not initialized..')
        docs = self.vectorstore.similarity_search(query, k)
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def generate(self, query: str, context: str) -> str:
          
          """Generate response using Groq API"""
          prompt = f"""
        ## Role: AI Menu Extraction Specialist
        
        You are an expert AI system specialized in menu data extraction. Your core expertise is analyzing menu documents and converting them into structured data formats while maintaining accuracy and consistency.
        
        ## Task Description
        Extract menu information from the provided document into a structured JSON format. Follow these precise guidelines:
        
        1. Extract all menu items, organizing them by their respective categories
        2. For each item, capture:
           - Exact item name
           - Price (numeric only, without currency symbols)
           - Complete item description if available
        
        ## Output Format
        Return a well-formatted JSON structure following this exact schema:
        ```json
        [
          {{
            "category": "Category Name",
            "items": [
              {{
                "item_name": "Full Item Name",
                "price": "Price as String",
                "description": "Complete Item Description"
              }}
            ]
          }}
        ]
        ```
        ## Special Instructions
        - Create proper categories even if they're implicit in the document
        - If a menu item has no description, include the key with an empty string
        - If price format varies (e.g., "₹500", "$10", "10.99"), extract only the numeric portion
        - Maintain the order of categories and items as they appear in the document
        - If an item has multiple price points (e.g., for size variations), create separate entries for each
        - If information is unclear or ambiguous, make the best determination based on context
        
        ## Context Document:
        {context}
        
        ## Query:
        {query}
        
        ## Response:
        """
          
          chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a specialized AI system for structured data extraction from unstructured text, with particular expertise in menu analysis."},
                    {"role": "user", "content": prompt}
                ],
                model='llama-3.3-70b-versatile',
                temperature=0.1,
                max_tokens=4096,
            )
          
          response = chat_completion.choices[0].message.content
          json_data = re.search(r'```json\n(.*?)\n```', response, re.DOTALL).group(1)


          output_path = os.path.join(self.extracted_menu_dir, 'api_menu_extractir.json')
          
          with open(output_path, "w", encoding="utf-8") as f:
              json.dump(json_data, f, indent=2, ensure_ascii=False)
          
          with open("api_menu_extractir.json", "w") as f:
              f.write(json_data)
          
          print(f"Menu saved successfully to {output_path}")
          
          return output_path
    
    def human_loop_for_llm(self, raw_menu_items: json, file_name:str = 'api_menu_extractir.json'):
        print('Review the JSON data before finalizing')

        with open(raw_menu_items , 'r') as f:
            raw_menu_items = json.load(f)

        reviewed_data = []
    
        for indx, category_block in enumerate(raw_menu_items):
            print(f'here the all items of first cat {indx+1} {category_block['category']}')
            print(f"\n📦 Category {indx+1}: {category_block['category']}")
            for items in category_block['items']:
                print(f"🟢 Item Name     : {items['item_name']}")
                print(f"💰 Price         : {items['price']}")
                print(f"📝 Description   : {items['description'] or 'N/A'}")
                print('---')
        
            action = input("Is this correct? (y/n): or press 'd' to delete or 'all' to save raw json file  ➤ ").strip().lower()
        
            if action == 'y':
                print("✅ Item confirmed.")
                reviewed_data.append(items)
        
            elif action == 'n':
                print("✏️  Editing the category.")
                category_block['category'] = input(f"    Edit Category Name [{category_block['category']}]: ") or category_block['category']
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


    def _organize_text_by_layout(ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Organize OCR results into layout-aware sections"""
            layout_text = []
            
            # Group by block
            current_block = None
            current_lines = []
            current_line = []
            
            for i in range(len(ocr_data["text"])):
                if ocr_data["text"][i].strip():
                    word_info = {
                        "text": ocr_data["text"][i],
                        "conf": ocr_data["conf"][i],
                        "bbox": [
                            ocr_data["left"][i],
                            ocr_data["top"][i],
                            ocr_data["left"][i] + ocr_data["width"][i],
                            ocr_data["top"][i] + ocr_data["height"][i]
                        ],
                        "block_num": ocr_data["block_num"][i],
                        "line_num": ocr_data["line_num"][i],
                        "par_num": ocr_data["par_num"][i],
                    }
                    
                    if current_block is None or current_block != ocr_data["block_num"][i]:
                        if current_block is not None:
                            if current_line:
                                current_lines.append(current_line)
                            layout_text.append({
                                "block_num": current_block,
                                "lines": current_lines
                            })
                        current_block = ocr_data["block_num"][i]
                        current_lines = []
                        current_line = [word_info]
                    elif current_line and current_line[0]["line_num"] != ocr_data["line_num"][i]:
                        current_lines.append(current_line)
                        current_line = [word_info]
                    else:
                        current_line.append(word_info)
            
            # Add the last block
            if current_line:
                current_lines.append(current_line)
            if current_block is not None:
                layout_text.append({
                    "block_num": current_block,
                    "lines": current_lines
                })
            
            return layout_text
    
    
    def extract_text_from_image(image_path: str) -> Dict[str, Any]:
        """Extract text and layout information from an image"""
        image = Image.open(image_path).convert("RGB")
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    
        # Get OCR results with layout information
        ocr_results = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
    
        # Process with LayoutLMv3 to get layout-aware features
        encoding = processor(image, return_tensors="pt", truncation=True)
        
        # Organize text by layout
        layout_text = _organize_text_by_layout(ocr_results)
        
        return {
            "text": pytesseract.image_to_string(image),
            "layout_text": layout_text,
            "image_size": image.size,
            "ocr_data": ocr_results
        }
    
    
    def _create_layout_aware_text(extracted_data: Dict[str, Any]) -> str:
            """Convert extracted OCR data into layout-aware text format"""
            layout_text = ""
            
            for block in extracted_data["layout_text"]:
                block_text = ""
                for line in block["lines"]:
                    line_text = " ".join([word["text"] for word in line])
                    block_text += line_text + "\n"
                
                layout_text += f"[BLOCK]\n{block_text}[/BLOCK]\n\n"
            
            return layout_text
    





