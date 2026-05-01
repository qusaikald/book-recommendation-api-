import pandas as pd
import numpy as np
import faiss
import logging
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self):
        self.df = None
        self.model = None
        self.index = None
        # Mock database for user read history (user_id -> list of book titles)
        self.users = defaultdict(list)
        
    def load_data(self):
        """Loads the book dataset from the CSV."""
        self.df = pd.read_csv(settings.CSV_PATH)
        
    def initialize_model(self):
        """Loads the sentence transformer model."""
        self.model = SentenceTransformer(settings.MODEL_NAME, device='cpu')
        
    def _build_embedding_text(self, row: pd.Series) -> str:
        """Combines metadata and description for embedding."""
        return (
            f"[{row['category']}] {row['title']} by {row['author']} ({row['year']}) | "
            f"{row['description']}"
        )
        
    def build_index(self):
        """Embeds all books and creates a new FAISS index, saving it to disk."""
        logger.info("Building new FAISS index...")
        texts = self.df.apply(self._build_embedding_text, axis=1).tolist()
        
        embeddings = self.model.encode(
            texts, 
            batch_size=settings.BATCH_SIZE,
            show_progress_bar=True, 
            normalize_embeddings=True
        ).astype(np.float32)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
        logger.info(f"Index built and saved to {settings.FAISS_INDEX_PATH}.")
        
    def load_index(self, force_rebuild: bool = False):
        """Loads the FAISS index from disk if it exists, otherwise builds it."""
        if force_rebuild:
            logger.info("Force rebuild requested...")
            self.build_index()
            return
            
        try:
            self.index = faiss.read_index(settings.FAISS_INDEX_PATH)
            logger.info(f"Loaded existing index from {settings.FAISS_INDEX_PATH}.")
        except RuntimeError:
            self.build_index()

    def search_similar_books(self, title: str, top_k: int = settings.TOP_K):
        """Searches for books similar to the given title in our database."""
        # Find the book in our database
        mask = self.df['title'].str.lower().str.contains(title.lower())
        matches = self.df[mask]
        
        if matches.empty:
            raise ValueError(f'No book found matching: "{title}"')
            
        query_row = matches.iloc[0]
        query_idx = query_row.name
        
        # Embed the query row dynamically so we don't rely on global state
        query_text = self._build_embedding_text(query_row)
        query_vec = self.model.encode([query_text], normalize_embeddings=True).astype(np.float32)
        
        # Perform search
        scores, indices = self.index.search(query_vec, top_k + 1)
        scores, indices = scores[0], indices[0]
        
        results = []
        for idx, score in zip(indices, scores):
            if idx != query_idx:
                book_data = self.df.iloc[idx].to_dict()
                book_data['similarity'] = float(score)
                results.append(book_data)
                
                
        return results[:top_k]

    def add_to_read_list(self, user_id: str, title: str):
        """Adds a book to a user's read list, validating it exists first."""
        mask = self.df['title'].str.lower().str.contains(title.lower())
        matches = self.df[mask]
        if matches.empty:
            raise ValueError(f'Book matching "{title}" not found in database.')
        
        # Grab the exact title from the database
        actual_title = matches.iloc[0]['title']
        
        if actual_title not in self.users[user_id]:
            self.users[user_id].append(actual_title)
            logger.info(f'Added "{actual_title}" to user {user_id}\'s read list.')
            
        return self.users[user_id]

    def get_user_recommendations(self, user_id: str, top_k: int = settings.TOP_K):
        """
        Generates blended recommendations based on the user's latest read books.
        If they read 'The Pragmatic Programmer' and then 'The Art of War',
        this interleaves recommendations from both to give a 50/50 split.
        """
        read_history = self.users.get(user_id, [])
        if not read_history:
            return [] # No history, maybe return popular books in future
            
        # Focus on the 2 most recently added books
        focus_books = read_history[-2:]
        
        # We fetch slightly more candidates to filter out ones already read
        candidates_per_book = top_k
        all_candidates = []
        
        for book_title in focus_books:
            # fetch similar books for this specific book
            similars = self.search_similar_books(book_title, top_k=candidates_per_book + len(read_history))
            all_candidates.append(similars)
            
        final_results = []
        seen_titles = set(read_history) # Never recommend books they've already read
        
        # Round-robin blending
        pointers = [0] * len(all_candidates)
        
        while len(final_results) < top_k:
            added_in_round = False
            for i in range(len(all_candidates)):
                if len(final_results) >= top_k:
                    break
                    
                # Take the next valid recommendation from this book's candidate list
                while pointers[i] < len(all_candidates[i]):
                    candidate = all_candidates[i][pointers[i]]
                    pointers[i] += 1
                    
                    if candidate['title'] not in seen_titles:
                        final_results.append(candidate)
                        seen_titles.add(candidate['title'])
                        added_in_round = True
                        break # Move to the next focus book for the 50/50 split
                        
            # If we went through all candidate lists and found nothing new, exit early
            if not added_in_round:
                break
                
        return final_results
