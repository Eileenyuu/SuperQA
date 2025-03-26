from threading import Lock
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingLoader:
    """
    A singleton class for loading and managing a pre-trained sentence embedding model.

    This class ensures that only one instance of the embedding model is loaded into memory,
    preventing redundant model initializations. It supports GPU acceleration if available.

    Attributes:
        _instance (EmbeddingLoader): The singleton instance of the class.
        _lock (Lock): A threading lock to prevent race conditions during initialization.
        _is_initialized (bool): Flag to ensure the model is only initialized once.
        MODEL (str): The Hugging Face model name used for embeddings.
        tokenizer (AutoTokenizer): Tokenizer instance for text preprocessing.
        model (AutoModel): Pre-trained transformer model for text embeddings.

    Methods:
        process_text(text: str) -> np.ndarray:
            Converts input text into a numerical embedding using mean pooling.
    """

    _instance = None
    _lock = Lock()
    _is_initialized = False

    def __new__(cls):
        """
        Ensures that only one instance of the EmbeddingLoader class is created (Singleton Pattern).
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingLoader, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        """
        Initializes the embedding model and tokenizer. Ensures that the model is loaded only once.
        """
        if not self.__class__._is_initialized:
            self.MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            self._initialize_models()
            self.__class__._is_initialized = True

    def _initialize_models(self):
        """
        Loads the tokenizer and model from the Hugging Face Transformers library.
        Moves the model to GPU if available.
        """
        with self._lock:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
            self.model = AutoModel.from_pretrained(self.MODEL)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()  # Set the model to evaluation mode

    def process_text(self, text):
        """
        Converts a single input text to an embedding using mean pooling.

        Args:
            text (str): The input text to encode.

        Returns:
            np.ndarray: The embedding as a NumPy array with shape (hidden_size,).
        """
        # Tokenize the input text; set padding and truncation as needed
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Get token embeddings from the last hidden state
        token_embeddings = model_output.last_hidden_state  # shape: (1, seq_length, hidden_size)
        attention_mask = encoded_input['attention_mask']

        # Mean Pooling: Compute the average of the token embeddings, considering the attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
        embedding = sum_embeddings / sum_mask  # Compute mean embedding

        # Move embedding to CPU and convert to numpy array
        if torch.cuda.is_available():
            embedding = embedding.cpu()

        return embedding.squeeze().numpy()