import os
import sys
import argparse
from typing import List, Dict, Any, Optional
import tempfile

import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv
import groq
from groq.types.chat import ChatCompletion
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# OCR libraries
import pytesseract
from pdf2image import convert_from_path
import layoutparser as lp

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

groq_client = groq.Client(api_key=groq_api_key)

class MenuImageProcessor:
    def __init__(self):
        # Initialize the layout model - using Detectron2 for layout detection
        # You may need to install additional dependencies:
        # pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"
        try:
            self.model = lp.Detectron2LayoutModel(
                config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65]
            )
        except Exception as e:
            print(f"Warning: Could not initialize Detectron2 model. Using basic OCR instead: {e}")
            self.model = None
            
    def process_image(self, image_path: str) -> str:
        """Process an image of a menu and extract text with layout awareness"""
        try:
            # Read the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.model:
                # Use layout parser to detect regions
                layout = self.model.detect(image)
                
                # Sort text blocks by their position (top to bottom, left to right)
                blocks = sorted(layout, key=lambda b: (b.block.y_1, b.block.x_1))
                
                # Extract text from each region while preserving layout
                text_segments = []
                
                for block in blocks:
                    # Crop the region
                    region_image = block.crop_image(image)
                    
                    # If it's a text or title block, perform OCR
                    if block.type in ["Text", "Title", "List"]:
                        # Convert numpy array to PIL Image for pytesseract
                        pil_image = Image.fromarray(region_image)
                        
                        # Perform OCR on the region
                        region_text = pytesseract.image_to_string(pil_image)
                        
                        # Add block type as context
                        text_segments.append(f"[{block.type}] {region_text.strip()}")
                
                return "\n\n".join(text_segments)
            else:
                # Fallback to basic OCR if layout parser is not available
                pil_image = Image.fromarray(image)
                return pytesseract.image_to_string(pil_image)
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return ""
    
    def process_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF with layout awareness"""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            # Process each page
            text_results = []
            for i, image in enumerate(images):
                # Save the image temporarily
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    image.save(tmp.name, 'JPEG')
                    tmp_path = tmp.name
                
                # Process the image
                page_text = self.process_image(tmp_path)
                text_results.append(f"--- Page {i+1} ---\n{page_text}")
                
                # Remove the temporary file
                os.unlink(tmp_path)
            
            return "\n\n".join(text_results)
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""

class RAGApplication:
    def __init__(self, docs_dir: str, model_name: str = "llama-3.3-70b-versatile"):
        self.docs_dir = docs_dir
        self.model_name = model_name
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.menu_processor = MenuImageProcessor()
        
    def load_documents(self) -> List[Document]:
        """Load documents from the specified directory"""
        # Load text documents
        text_loader = DirectoryLoader(
            self.docs_dir, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            show_progress=True
        )
        text_documents = text_loader.load()
        print(f"Loaded {len(text_documents)} text documents")
        
        # Process image and PDF files in the directory
        image_documents = []
        pdf_documents = []
        
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    print(f"Processing image: {file_path}")
                    text = self.menu_processor.process_image(file_path)
                    if text:
                        image_documents.append(Document(page_content=text, metadata={"source": file_path}))
                
                elif file_ext == '.pdf':
                    print(f"Processing PDF: {file_path}")
                    text = self.menu_processor.process_pdf(file_path)
                    if text:
                        pdf_documents.append(Document(page_content=text, metadata={"source": file_path}))
        
        print(f"Processed {len(image_documents)} image documents")
        print(f"Processed {len(pdf_documents)} PDF documents")
        
        # Combine all documents
        all_documents = text_documents + image_documents + pdf_documents
        return all_documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]):
        """Create a vector store from document chunks"""
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        print("Vector store created successfully")
    
    def setup(self):
        """Set up the RAG application by loading documents and creating the vector store"""
        documents = self.load_documents()
        chunks = self.process_documents(documents)
        self.create_vectorstore(chunks)
        
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a given query"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call setup() first.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
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
- Handle OCR errors intelligently, making reasonable corrections based on context
- Identify section headers vs. item names using formatting and layout clues
- If information is unclear or ambiguous, make the best determination based on context

## Context Document:
{context}

## Query:
{query}

## Response:
"""
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a specialized AI system for structured data extraction from unstructured text, with particular expertise in menu analysis and OCR post-processing."},
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            temperature=0.1,
            max_tokens=4096,
        )

        return chat_completion.choices[0].message.content
    
    def query(self, query: str, k: int = 5) -> str:
        """Process a query through the RAG pipeline"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query, k=k)
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate response
        response = self.generate(query, context)
        return response

    def process_single_image(self, image_path: str, query: str) -> str:
        """Process a single image and generate a response directly"""
        print(f"Processing image: {image_path}")
        extracted_text = self.menu_processor.process_image(image_path)
        if not extracted_text:
            return "Could not extract text from the image."
        
        return self.generate(query, extracted_text)
    
    def process_single_pdf(self, pdf_path: str, query: str) -> str:
        """Process a single PDF and generate a response directly"""
        print(f"Processing PDF: {pdf_path}")
        extracted_text = self.menu_processor.process_pdf(pdf_path)
        if not extracted_text:
            return "Could not extract text from the PDF."
        
        return self.generate(query, extracted_text)
    
    def save_vectorstore(self, path: str):
        """Save the vector store to disk"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call setup() first.")
        self.vectorstore.save_local(path)
        print(f"Vector store saved to {path}")
    
    def load_vectorstore(self, path: str):
        """Load the vector store from disk"""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        print(f"Vector store loaded from {path}")

def main():
    parser = argparse.ArgumentParser(description="RAG Application with OCR using Groq API")
    parser.add_argument("--docs_dir", type=str, help="Directory containing documents")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--save_vs", type=str, help="Path to save vector store")
    parser.add_argument("--load_vs", type=str, help="Path to load vector store from")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="Groq model to use")
    parser.add_argument("--image", type=str, help="Process a single image file")
    parser.add_argument("--pdf", type=str, help="Process a single PDF file")
    
    args = parser.parse_args()
    
    # Initialize RAG application
    rag_app = RAGApplication(docs_dir=args.docs_dir or ".", model_name=args.model)
    
    # Process a single image or PDF if specified
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file {args.image} not found")
            return
        
        query = args.query or "Extract all menu items with their prices and descriptions"
        response = rag_app.process_single_image(args.image, query)
        print("\nQuery:", query)
        print("\nResponse:", response)
        return
    
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"Error: PDF file {args.pdf} not found")
            return
        
        query = args.query or "Extract all menu items with their prices and descriptions"
        response = rag_app.process_single_pdf(args.pdf, query)
        print("\nQuery:", query)
        print("\nResponse:", response)
        return
    
    # Set up or load vector store
    if args.load_vs:
        rag_app.load_vectorstore(args.load_vs)
    else:
        if not args.docs_dir:
            print("Error: --docs_dir is required when not loading a vector store")
            return
        rag_app.setup()
        if args.save_vs:
            rag_app.save_vectorstore(args.save_vs)
    
    # Process query if provided
    if args.query:
        response = rag_app.query(args.query, k=args.k)
        print("\nQuery:", args.query)
        print("\nResponse:", response)
    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            query = input("\nEnter query: ")
            if query.lower() == 'exit':
                break
            
            response = rag_app.query(query, k=args.k)
            print("\nResponse:", response)

if __name__ == "__main__":
    main() 