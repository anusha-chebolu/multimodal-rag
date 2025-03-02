from qdrant_client import QdrantClient, models
import base64
from io import BytesIO
from tqdm import tqdm
from embedding import batch_iterate  # Import batch processing from embedding.py

# Image processing helper
def image_to_base64(image):
    """Convert image to base64 format."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class QdrantVDB_QB:
    """Manages Qdrant vector storage for image embeddings."""
    
    def __init__(self, collection_name="multimodal_rag", vector_dim=128, batch_size=2):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.client = self._define_client()

    def _define_client(self):
        """Initialize the Qdrant client."""
        return QdrantClient(":memory:")

    def create_collection(self):
        """Create a new Qdrant collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )

    def ingest_data(self, embeddata):
        """Ingests embeddings into Qdrant."""
        for i, batch_embeddings in tqdm(enumerate(batch_iterate(embeddata.embeddings, self.batch_size)), desc="Ingesting data"):
            points = []
            for j, embedding in enumerate(batch_embeddings):
                image_bs64 = image_to_base64(embeddata.images[i * self.batch_size + j])
                current_point = models.PointStruct(
                    id=i * self.batch_size + j,
                    vector=embedding,
                    payload={"image": image_bs64}
                )
                points.append(current_point)
    
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

class Retriever:
    """Handles search queries to retrieve similar images from Qdrant."""
    
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        """Retrieve top matches for the given query."""
        query_embedding = self.embeddata.get_query_embedding(query)
        query_result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            limit=4,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0
                )
            )
        )
        return query_result
