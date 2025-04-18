from main.components.cashier_.configs.config import cashier_configs
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

with open('main\\components\\menu_maneger\\pdf_dir\\reviewed_menu_data.json', 'r') as sample_json:
    data = json.load(sample_json)

sample_input = 'get me couple of Classic Club Sandwich along with two Kheema Ghotala'


class cashier_root:

    def __init__(self, config, params):
        self.config = config
        self.params = params

    def download_model(self):
        '''
        Downloads and prepares the Gemma 3 1B model for local use.
    
        Args:
        token (str): Hugging Face API token for accessing gated models     
        '''
        
        token = os.getenv('HUGGING_FACE_TOKEN')
        if token is None:
            print("Error: Hugging Face token is required to download Gemma models")

        os.environ['HUGGING_FACE_TOKEN'] = token

        print("Starting to download the model. This may take some time...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config['base_model_id'],
           token = token)
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config['base_model_id'],
                device_map ='auto',
                torch_dtype="auto",
                trust_remote_code=True,
                token=token
            )
            # output_dir = Path(self.config['base_model_path']) / self.config['base_model_id']
            print(f'Saving the {self.config['base_model_id']} to local folder {self.config['dowloaded_base_model_path']}')
            self.config['dowloaded_base_model_path'].mkdir(parents=True, exist_ok=True)


            model.save_pretrained(self.config['dowloaded_base_model_path'])
            tokenizer.save_pretrained(self.config['dowloaded_base_model_path'])

            print("Model downloaded and saved successfully!")
        
            # Create a pipeline for easy usage
            print("Creating inference pipeline...")
            llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            # Return the pipeline for use
            return llm

        except Exception as e:
            raise e
    
    def load_and_process(self):

        '''
        Funtion to load a previously downloaded model from local storage.

        Args: 
            llm from AutoModelCasuallm throgh pipeline

        Returns: Generated output from model
        str() Processed prompt
        '''
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config['base_model_path'])
            model = AutoModelForCausalLM.from_pretrained(self.config['base_model_path'],
            device_map = 'auto',
            torch_dtype = 'auto'
            )

            llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
            print("Model loaded successfully!")
            return llm
        
        
        except Exception as e:
            raise e
    
    def order_maneger(self, llm, menu_data):
        '''
        Funtion for undersatnding the Natural language to capture the order from user

        Agrus:
           str() : 
           llm: The loaded language model pipeline
           menu_data: Menu items data
           customer_input: Customer's spoken order

        Return: valide Json of user order contains order_item, price and quanti
        '''
        prompt = [
        {
            "role": "system",
            "content": "You're a restaurant order extractor. Match spoken order to menu items with price and return a just JSON list of item_name, price and quantity. Nothing other than that."
        },
        {
            "role": "user",
            "content": f"Menu items: {menu_data}\nCustomer said: {sample_input}"
        }
        ]
        tokenizer = llm.tokenizer

        # Convert to the format expected by the model
        formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        
        print("Generating response...")
        response = llm(formatted_prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        
        # Extract just the model's response (not the whole prompt)
        model_response = response[len(formatted_prompt):]
        
        return model_response
        

if __name__ == '__main__':
    obj = cashier_configs()
    config_params = obj.cashier_all_configs()
    comp_obj = cashier_root(config_params, config_params)
    # if not os.path.exists(out):
    comp_obj.download_model()
    llm = comp_obj.load_and_process()
    orders_json = comp_obj.order_maneger(llm,data)
    print(orders_json)
    # print(config_params)