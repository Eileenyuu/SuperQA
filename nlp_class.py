from threading import Lock
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class EmbeddingLoader:
    _instance = None
    _lock = Lock()
    _is_initialized = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingLoader, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        if not self.__class__._is_initialized:
            self.MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            self._initialize_models()
            self.__class__._is_initialized = True

    def _initialize_models(self):
        with self._lock:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
            self.model = AutoModel.from_pretrained(self.MODEL)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model.eval()

    def process_text(self, text):
        """
        Converts a single input text to an embedding using mean pooling.

        Args:
            text (str): The input text to encode.

        Returns:
            numpy.ndarray: The embedding as a NumPy array with shape (hidden_size,).
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

        # Mean Pooling: Compute the average of the token embeddings, taking the attention mask into account
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask

        # Move embedding to CPU and convert to numpy array
        if torch.cuda.is_available():
            embedding = embedding.cpu()

        return embedding.squeeze().numpy()
