from main.components.menu_maneger.menu_extractor import pdf_extractor
from main.components.menu_maneger.config.menu_maneger_config import pdf_source
import re, os

class MenuManagerPipeline:
    def __init__(self):
        pdf_path_obj = pdf_source()
        self.pdf_dir, self.prompt = pdf_path_obj.create_source_dir()

    def run_pipeline(self):
        obj = pdf_extractor()
        ext = self.pdf_dir['pdf_url'].split('.')[-1]
        print(' ‼️ For text and image extraction only AI extractor is available. ‼️')
        if ext == 'pdf':
            method = input("⚙️ Press A for AI-PDF extractor, B for manual PDF extractor: ").strip().lower()
            if method == 'a':
                print("📄 Running AI PDF extractor...")
                obj.load_and_process_docs(self.pdf_dir['pdf_url'])
                retrival_query = obj.retrival(query='give me complete extracted menu in json')
                raw_json_from_llm = obj.generate(query='give me complete extracted menu in json', context=retrival_query,
                filename='pdf')
                obj.human_loop_for_llm(raw_json_from_llm)
            else:
                raw_json_manual = obj.manual_extract_pdf_text()
                obj.human_in_loop_manual_extractor(raw_json_manual)
                print("<<<<< Pipeline Compelted 💢 >>>>>>>>.")
        
        elif ext == 'txt':
            print(" 📄 Running Text extractor...")
            obj.load_and_process_docs(self.pdf_dir['pdf_url'])
            retrival_query = obj.retrival(query='give me complete extracted menu in json')
            raw_json_from_llm = obj.generate(query='give me complete extracted menu in json', context=retrival_query,
            filename='text')
            obj.human_loop_for_llm(raw_json_from_llm)
            print("<<<<< Pipeline Completed 💢 >>>>>>>>.")

        elif ext in ['jpg', 'jpeg', 'png']:
            extracted_text = obj.extract_text_from_image(self.pdf_dir['pdf_url'])
            formated_text = obj._create_layout_aware_text(extracted_text)
            response = obj.generate(query='give me complete extracted menu in json', context=formated_text,
            filename='image')
            print(response)
            obj.human_loop_for_llm(response)
            print("<<<<< Pipeline Completed 💢 >>>>>>>>.")

        else:
            print("❌ Unsupported file format. Please provide a PDF, TXT, or image file.")

                
        

if __name__ == "__main__":
    obj = MenuManagerPipeline()
    obj.run_pipeline()
    