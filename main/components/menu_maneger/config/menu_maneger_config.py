from pathlib import Path
from main.functions.common_function import read_yaml_file, create_directories, save_json_file
from langchain.prompts import PromptTemplate

class pdf_source:
    '''
    class to manage the source of the pdf files

    takes the source directory and the source URL as input
    and creates the source directory if it does not exist
    '''
    def __init__(self):
        self.config = read_yaml_file(Path('config/config.yaml'))
        

    def create_source_dir(self):
        config = self.config.pdf_maneger
        create_directories([config.pdf_dir])
        create_directories([config.extracted_menu_dir])

        pdf_urls = {
            'pdf_dir': Path(config['pdf_dir']),
            'extracted_menu': Path(config['extracted_menu_dir']),
            'pdf_url' : config['pdf_path'],
            'final_json_name' : config['json_file_name'],
            'manual_json_name' : config['final_json_file_name'],
            'extracter_model_name' : config['extracter_model_name']
        }

        # 💬💬 Prompt Maneger for Model(Prompt-Engineering) 🤖 
        # Using the PromptTemplate class to create a prompt template for the model.

        prompt = PromptTemplate(
            template="""
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
            """,
            input_variables=['context', 'query'],
            validate_template=True
        )
        # save_json_file(self.config['pdf_url'],)
        prompt.save('prompt.json')
    
        


        return pdf_urls, prompt

