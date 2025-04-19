from main.components.menu_maneger.menu_extractor import pdf_extractor


class MenuManagerPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        obj = pdf_extractor()
        files = input('Upload the file .pdf, .jpg, .png, .txt files areonly accepted ')
        if files.endswith('.pdf'):
            methods = input('press A. for AI-Pdf extractor : Press B for Manual pdf extractor  ').lower()
            if methods == 'a':
                obj.load_and_process_docs(files)
                retrival_query = obj.retrival(query='give me complete extracted menu in json')
                response = obj.generate(query='give me complete extracted menu in json', context=retrival_query)
                print(response)


if __name__ == "__main__":
    obj = MenuManagerPipeline()
    obj.run_pipeline()
    