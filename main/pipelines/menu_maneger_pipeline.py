from main.components.menu_maneger.menu_extractor import pdf_extractor


class MenuManagerPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        obj = pdf_extractor()
        raw_json = obj.extract_pdf_text()
        # print(raw_json)
        final_json = obj.review_json(raw_json)
        # print('     ')
        # print('---------------------------' )
        # print('Final JSON data:', final_json)
        return final_json


if __name__ == "__main__":
    obj = MenuManagerPipeline()
    obj.run_pipeline()
    