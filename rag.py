import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image


# Stream processing function
def stream_response(response):
    """Yields words in a response one-by-one for streaming."""
    words = response[0].split()
    for word in words:
        yield word + " "

# Image Resizing Helper
def resize_image(image_path, size=(700, 2100)):
    """Resizes an image to a fixed size for consistency."""
    with Image.open(image_path) as img:
        img_resized = img.resize(size, Image.LANCZOS)
        img_resized.save(image_path)

class RAG:
    """Retrieval-Augmented Generation (RAG) system."""
    
    def __init__(self, retriever, llm_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        self.llm_name = llm_name
        self._setup_llm()
        self.retriever = retriever

    def _setup_llm(self):
        """Loads Qwen 2.5 VL-3B model and processor."""
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.llm_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir="/content/drive/MyDrive/multimodal-rag/Qwen/hf_cache"
        )
        self.processor = AutoProcessor.from_pretrained(
            self.llm_name,
            cache_dir="/content/drive/MyDrive/multimodal-rag/Qwen/hf_cache"
        )

    def generate_context(self, query):
        """Retrieve relevant images based on query from Qdrant."""
        results = self.retriever.search(query)
        
        # Fix: Handle case where no results are returned
        if not results or len(results) == 0:
            return None  

        # Fix: Access first result directly (no `points` attribute needed)
        return f"./images/page{results[0].id}.jpg"

    def query(self, query):
        """Generate a response using retrieved image and text input."""
        image_context = self.generate_context(query=query)

        if not image_context:
            print("\nNo relevant images found. Proceeding with text-only response.\n")

            # Text-only query (No images available)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
            ]
        else:
            print(f"\nRetrieved image for query: {query}")
            resize_image(image_context)  # Ensure the image is properly formatted

            # Construct multimodal prompt
            qa_prompt_tmpl_str = f"""The user has asked the following question:

            ---------------------
            Query: {query}
            ---------------------

            Some images are available to you for this question.
            You have to understand these images thoroughly and 
            extract all relevant information that will 
            help you answer the query.
            ---------------------
            """
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_context},
                        {"type": "text", "text": qa_prompt_tmpl_str},
                    ],
                }
            ]

        # Prepare model inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate response from the model
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return stream_response(output_text)
