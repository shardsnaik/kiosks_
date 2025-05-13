# from main.components.llm_engine_.configs.config import cashier_configs
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# # from unsloth import FastLanguageModel
# from transformers import AutoTokenizer
# import os, json
# from pathlib import Path
# from dotenv import load_dotenv
# load_dotenv()

# with open('main\\components\\menu_maneger\\pdf_dir\\reviewed_menu_data.json', 'r') as sample_json:
#     data = json.load(sample_json)

# sample_input = 'get me couple of Classic Club Sandwich along with two Kheema Ghotala'


# class cashier_root:

#     def __init__(self, config, params):
#         self.config = config
#         self.params = params

#     def download_model(self):
#         '''
#         Downloads and prepares the Gemma 3 1B model for local use.
    
#         Args:
#         token (str): Hugging Face API token for accessing gated models     
#         '''
        
#         token = os.getenv('HUGGING_FACE_TOKEN')
#         if token is None:
#             print("Error: Hugging Face token is required to download Gemma models")

#         os.environ['HUGGING_FACE_TOKEN'] = token

#         print("Starting to download the model. This may take some time...")

#         try:
#             tokenizer = AutoTokenizer.from_pretrained(self.config['base_model_id'],
#            token = token)
            
#             model = AutoModelForCausalLM.from_pretrained(
#                 self.config['base_model_id'],
#                 device_map ='auto',
#                 torch_dtype="auto",
#                 trust_remote_code=True,
#                 token=token
#             )
#             # output_dir = Path(self.config['base_model_path']) / self.config['base_model_id']
#             print(f'Saving the {self.config['base_model_id']} to local folder {self.config['dowloaded_base_model_path']}')
#             self.config['dowloaded_base_model_path'].mkdir(parents=True, exist_ok=True)


#             model.save_pretrained(self.config['dowloaded_base_model_path'])
#             tokenizer.save_pretrained(self.config['dowloaded_base_model_path'])

#             print("Model downloaded and saved successfully!")
        
#             # Create a pipeline for easy usage
#             print("Creating inference pipeline...")
#             llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
#             # Return the pipeline for use
#             return llm

#         except Exception as e:
#             raise e
    
#     def load_and_process(self):

#         '''
#         Funtion to load a previously downloaded model from local storage.

#         Args: 
#             llm from AutoModelCasuallm throgh pipeline

#         Returns: Generated output from model
#         str() Processed prompt
#         '''
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(self.config['base_model_path'])
#             model = AutoModelForCausalLM.from_pretrained(self.config['base_model_path'],
#             device_map = 'auto',
#             torch_dtype = 'auto'
#             )

#             llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
#             print("Model loaded successfully!")
#             return llm
        
        
#         except Exception as e:
#             raise e
    
#     def order_maneger(self, llm, menu_data):
#         '''
#         Funtion for undersatnding the Natural language to capture the order from user

#         Agrus:
#            str() : 
#            llm: The loaded language model pipeline
#            menu_data: Menu items data
#            customer_input: Customer's spoken order

#         Return: valide Json of user order contains order_item, price and quanti
#         '''
#         prompt = [
#         {
#             "role": "system",
#             "content": "You're a restaurant order extractor. Match spoken order to menu items with price and return a just JSON list of item_name, price and quantity. Nothing other than that."
#         },
#         {
#             "role": "user",
#             "content": f"Menu items: {menu_data}\nCustomer said: {sample_input}"
#         }
#         ]
#         tokenizer = llm.tokenizer

#         # Convert to the format expected by the model
#         formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        
#         print("Generating response...")
#         response = llm(formatted_prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
        
#         # Extract just the model's response (not the whole prompt)
#         model_response = response[len(formatted_prompt):]
        
#         return model_response
        

# if __name__ == '__main__':
#     obj = cashier_configs()
#     config_params = obj.cashier_all_configs()
#     comp_obj = cashier_root(config_params, config_params)
#     # if not os.path.exists(out):
#     comp_obj.download_model()
#     llm = comp_obj.load_and_process()
#     orders_json = comp_obj.order_maneger(llm,data)
#     print(orders_json)
#     # print(config_params)






import torch, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class llm_engine:
    
    def __init__(self, config, params):
        self.config  = config 
        self.params  = params 
        
    
    def load_model_and_doc(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            print("🔄 Loading model and tokenizer...")
            model = AutoModelForCausalLM.from_pretrained(self.config['fine_tuned_model']).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.config['fine_tuned_model'])
            print("🔄 Loading RAG embedder and data...")
            embedder = SentenceTransformer('all-MiniLM-L6-v2')                

        except Exception as e:
            print('Failed to Load fine tunned Model ❌. ✔️ Check-out the path and model-id for confirmation')
            raise(e)
        
        with open(self.config['rag_file_in_json'], "r", encoding="utf-8") as f:
                rag_data = json.load(f)
                text_chunks = [json.dumps(item) if isinstance(item, dict) else str(item) for item in rag_data]
                text_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)

        return model, tokenizer,embedder, text_embeddings, text_chunks
    
    
    
    # Qwen-style formatting
    def build_qwen_prompt(self, history):
        prompt = ""
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    
    # Retrieve top-k relevant chunks
    def retrieve_relevant_docs(self, embedder, query,text_embeddings,text_chunks, k=3):
        query_embedding = embedder.encode([query], convert_to_tensor=True)
        similarities = cosine_similarity(query_embedding.cpu().numpy(), text_embeddings.cpu().numpy())[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
    
        formatted_chunks = []
        for chunk in [text_chunks[i] for i in top_k_indices]:
            try:
                data = json.loads(chunk)
                if isinstance(data, dict) and 'items' in data:
                    lines = [f"{item['item_name']} - {item['price']}" for item in data['items']]
                    formatted_chunks.append("\n".join(lines))
                else:
                    formatted_chunks.append(chunk)
            except:
                formatted_chunks.append(chunk)
    
        return formatted_chunks
    
    def llm_core(self, model, tokenizer):
          
        # Chat memory
        chat_history = [{
            "role": "system",
            "content": "You're a helpful restaurant cashier. Use the context below to answer customer questions about the menu. You should remember past orders and assist the customer politely."
        }]
        
        print("✅ Ready. Type 'exit' to quit.\n")
        while True:
            user_text = input("User Question: ")
            if user_text.lower() in ["exit", "quit"]:
                break
        
            # RAG retrieval
            rag_context = "\n".join(self.retrieve_relevant_docs(user_text))
        
            # Append user with RAG context
            chat_history.append({
                "role": "user",
                "content": f"Context:\n{rag_context}\n\nQuestion: {user_text}"
            })
        
            # Build prompt and generate
            prompt = self.build_qwen_prompt(chat_history)
            inputs_ = tokenizer(prompt, return_tensors='pt').to(self.device)
        
            with torch.no_grad():
                outputs = model.generate(**inputs_, max_new_tokens=100, temperature=0.7)
        
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_reply = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        
            # chat_history.append({
            #     "role": "assistant",
            #     "content": assistant_reply
            # })
            chat_history.append({
            "role": "user",
            "content": f"Question: {user_text}",
            "raw_input": user_text,  # Add raw input for clean display
            "context": rag_context    # Optional if you need it later
        })
            
            
            assistant_reply = re.search(r'assistant\n(.*)', assistant_reply, re.DOTALL).group(1).strip()
        
            print(f"\n👤 User: {user_text}")
            print(f"🤖 Assistant: {assistant_reply}\n")
            
