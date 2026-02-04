from abc import ABC, abstractmethod
import os
from glob import glob
import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image

class BaseSearcher(ABC):
    """
    Abstract Base Class for retrieval. 
    """
    @abstractmethod
    def index_context(self, source_data, frame_meta_dict=None):
        """
        Ingests data and creates embeddings.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 3):
        """
        Performs vector search and returns top-k results.
        """
        pass

class ImageSearcher(BaseSearcher):
    def __init__(self, model_name='sentence-transformers/clip-ViT-L-14', device='cuda', seed=420, alpha= 0.7):
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.current_image_paths = []
        self.current_embeddings = None
        self.current_meta_embeddings = None
        self.current_meta_mask = []
        self.seed = seed
        self.alpha = alpha

    def index_context(self, path_or_paths, frame_meta_dict = None):
        """
        Loads images, embeds them, and stores tensors in memory.
        Accepts: Directory String, Single File String, OR List of Strings.
        """
        # Handle List of Paths (The "Zoom-in" logic after the first iterance)
        if isinstance(path_or_paths, list):
            self.current_image_paths = path_or_paths
        
        # Handle Directory
        elif os.path.isdir(path_or_paths): 
            self.current_image_paths = glob(os.path.join(path_or_paths, "*.jpg")) + \
                                       glob(os.path.join(path_or_paths, "*.png"))
        
        # Handle Single File
        else:
            self.current_image_paths = [path_or_paths]

        if not self.current_image_paths:
            print(f"Warning: No images found.")
            self.current_embeddings = None
            self.current_meta_embeddings = None
            return
        
        self.current_meta_mask = []

        # Load and Embed
        try:
            images = [Image.open(p).convert('RGB') for p in self.current_image_paths]
            self.current_embeddings = self.model.encode(images, convert_to_tensor=True, show_progress_bar=False)
        except Exception as e:
            print(f"Error indexing images: {e}")
            self.current_embeddings = None
            self.current_meta_mask = []
            return
        
        if frame_meta_dict:
            meta_texts = []
            for p in self.current_image_paths:
                filename = os.path.splitext(os.path.basename(p))[0]
                frame_id = filename.split('_seq')[0] 
                
                metas = frame_meta_dict.get(frame_id, [])
                has_meta = bool(metas)
                meta_text = ", ".join([str(m) for m in metas if m]) if has_meta else ""
                meta_texts.append(meta_text)
                self.current_meta_mask.append(has_meta)
            
            try:
                self.current_meta_embeddings = self.model.encode(meta_texts, convert_to_tensor=True, show_progress_bar=False)
            except Exception as e:
                print(f"Error indexing metadata: {e}")
                self.current_meta_embeddings = None
                self.current_meta_mask = []
                return
            
        else:
            self.current_meta_embeddings = None
            self.current_meta_mask = []

        if self.current_embeddings is not None:
            assert self.current_embeddings.shape[0] == len(self.current_image_paths)
            
            if self.current_meta_embeddings is not None:
                assert self.current_embeddings.shape[0] == self.current_meta_embeddings.shape[0]
                
                self.current_meta_mask = torch.tensor(
                                            self.current_meta_mask,
                                            dtype=torch.bool,
                                            device=self.current_embeddings.device
                                        )

    def search(self, query: str, k: int = 3):
        if self.current_embeddings is None:
            return []

        query_embedding = self.model.encode([query], convert_to_tensor=True).to(self.current_embeddings.device)
        
        # Ensure k isn't larger than the number of available images
        real_k = min(k, len(self.current_image_paths))

        visual_scores = util.cos_sim(query_embedding, self.current_embeddings)[0]

        final_scores = visual_scores.clone()
        if self.current_meta_embeddings is not None:
            text_scores = util.cos_sim(query_embedding, self.current_meta_embeddings)[0]
            final_scores[self.current_meta_mask] = self.alpha * visual_scores[self.current_meta_mask] + (1 - self.alpha) * text_scores[self.current_meta_mask]

        top_results = torch.topk(final_scores, k=real_k)
        top_indices = top_results.indices.cpu().numpy()

        results = []
        for idx in top_indices:
            results.append(self.current_image_paths[idx])
            
        return results

class SummarySearcher(BaseSearcher):
    """
    Retriever for Text Summaries using a text-optimized model.
    """
    def __init__(self, model_name, device='cuda', seed=420):
        print(f"Loading Text Searcher with model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})
        self.model._first_module().auto_model.config.use_cache = False
        self.device = device
        self.seed = seed
        self.query_instruction = "Instruction: "
        
        self.current_ids = []
        self.current_texts = []
        self.current_embeddings = None

    def index_context(self, source_data, frame_meta_dict=None):
        # Reset state
        self.current_ids = []
        self.current_texts = []
        self.current_embeddings = None

        if not source_data:
            return

        self.current_ids = list(source_data.keys())
        self.current_texts = list(source_data.values())

        try:
            self.current_embeddings = self.model.encode(
                self.current_texts, 
                convert_to_tensor=True, 
                show_progress_bar=False
            )
        except Exception as e:
            print(f"Error indexing summaries: {e}")
            self.current_embeddings = None

    def search(self, query: str, k: int = 3):
        if self.current_embeddings is None:
            return []

        query_embedding = self.model.encode([self.query_instruction + query], convert_to_tensor=True).to(self.current_embeddings.device)
        real_k = min(k, len(self.current_texts))
        
        # Pure Cosine Similarity for text
        scores = util.cos_sim(query_embedding, self.current_embeddings)[0]
        
        top_results = torch.topk(scores, k=real_k)
        top_indices = top_results.indices.cpu().numpy()

        results = {}
        for idx in top_indices:
            results[self.current_ids[idx]] = self.current_texts[idx]
            
        return results