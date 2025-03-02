import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from tqdm import tqdm
from PIL import Image

# Batch processing helper
def batch_iterate(lst, batch_size):
    """Yield batches of size batch_size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    """Handles text and image embeddings using ColPali."""
    
    def __init__(self, embed_model_name="vidore/colpali-v1.2", batch_size=1):
        self.embed_model_name = embed_model_name
        self.batch_size = batch_size
        self.embed_model, self.processor = self._load_embed_model()
        self.embeddings = []

    def _load_embed_model(self):
        """Load the ColPali embedding model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_model = ColPali.from_pretrained(
            self.embed_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True, 
            cache_dir="/content/drive/MyDrive/multimodal-rag/colpali/hf_cache"  # Updated cache directory
        )
        processor = ColPaliProcessor.from_pretrained(self.embed_model_name)
        return embed_model, processor
    
    def get_query_embedding(self, query):
        """Generate an embedding for a text query."""
        with torch.no_grad():
            query_input = self.processor.process_queries([query]).to(self.embed_model.device)
            query_embedding = self.embed_model(**query_input)
        return query_embedding[0].cpu().float().numpy().tolist()

    def generate_embedding(self, images):
        """Generate embeddings for a batch of images."""
        with torch.no_grad():
            batch_images = self.processor.process_images(images).to(self.embed_model.device)
            image_embeddings = self.embed_model(**batch_images).cpu().float().numpy().tolist()
        return image_embeddings
        
    def embed(self, images):
      """Ensure images are PIL objects before processing"""
      
      # Convert paths to PIL Image objects if necessary
      self.images = [Image.open(img) if isinstance(img, str) else img for img in images]

      self.all_embeddings = []
      for batch_images in tqdm(batch_iterate(self.images, self.batch_size), desc="Generating embeddings"):
          batch_embeddings = self.generate_embedding(batch_images)
          self.embeddings.extend(batch_embeddings)

