import json
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
import time
from dataclasses import dataclass

@dataclass
class SentenceMatch:
    """Simplified data class for sentence matches"""
    sentence1_id: str
    sentence2_id: str
    sentence1_text: str
    sentence2_text: str
    document1: str
    document2: str
    similarity_score: float

class SentenceMatcher:
    def __init__(self, embeddings_file: str = "embeddings/sentence_embeddings.json"):
        """
        Initialize the simplified sentence matcher
        """
        self.embeddings_file = Path(embeddings_file)
        self.embeddings_data = None
        self.embeddings_matrix = None
        self.sentence_lookup = {}
        
        # Only one threshold - minimum similarity to consider
        self.min_similarity_threshold = 0.85
        
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings from JSON file"""
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        
        print("Loading embeddings...")
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            self.embeddings_data = json.load(f)
        
        # Extract embeddings and create lookup
        embeddings_list = []
        for i, item in enumerate(self.embeddings_data["embeddings"]):
            embeddings_list.append(item["embedding"])
            self.sentence_lookup[i] = {
                "sentence_id": item["sentence_id"],
                "sentence_text": item["sentence_text"],
                "document": item["document"],
                "sentence_index": item["sentence_index"]
            }
        
        # Convert to numpy array for efficient similarity calculations
        self.embeddings_matrix = np.array(embeddings_list)
        print(f"Loaded {len(embeddings_list)} sentence embeddings")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def calculate_similarity_matrix(self) -> np.ndarray:
        """Calculate cosine similarity matrix for all sentence pairs"""
        print("Calculating similarity matrix...")
        n_sentences = len(self.embeddings_matrix)
        
        # Normalize all embeddings
        normalized_embeddings = self.embeddings_matrix / np.linalg.norm(
            self.embeddings_matrix, axis=1, keepdims=True
        )
        
        # Calculate similarity matrix using matrix multiplication
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix
    
    def find_similar_sentences(self) -> List[SentenceMatch]:
        """
        Find all sentence pairs with similarity ≥ 0.85
        """
        print(f"Finding similar sentences (≥ {self.min_similarity_threshold} similarity only)...")
        
        similarity_matrix = self.calculate_similarity_matrix()
        matches = []
        
        n_sentences = len(self.sentence_lookup)
        processed_pairs = 0
        skipped_pairs = 0
        
        # Iterate through upper triangle of similarity matrix
        for i in range(n_sentences):
            for j in range(i + 1, n_sentences):
                similarity = similarity_matrix[i, j]
                
                # Hard threshold - only process if >= min_similarity
                if similarity >= self.min_similarity_threshold:
                    processed_pairs += 1
                    
                    # Create simplified match object
                    match = SentenceMatch(
                        sentence1_id=self.sentence_lookup[i]["sentence_id"],
                        sentence2_id=self.sentence_lookup[j]["sentence_id"],
                        sentence1_text=self.sentence_lookup[i]["sentence_text"],
                        sentence2_text=self.sentence_lookup[j]["sentence_text"],
                        document1=self.sentence_lookup[i]["document"],
                        document2=self.sentence_lookup[j]["document"],
                        similarity_score=similarity
                    )
                    
                    matches.append(match)
                else:
                    skipped_pairs += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{n_sentences} sentences - Found {processed_pairs} matches")
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        print(f"✅ Found {len(matches)} similar sentence pairs (all ≥ {self.min_similarity_threshold})")
        print(f"📊 Processed {processed_pairs} qualifying pairs, skipped {skipped_pairs} below threshold")
        return matches
    
    def save_matches_to_file(self, matches: List[SentenceMatch], output_file: str = "analysis/sentence_matches.json"):
        """
        Save sentence matches to JSON file - simplified format
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert matches to simplified serializable format
        matches_data = {
            "total_matches": len(matches),
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "minimum_similarity_threshold": self.min_similarity_threshold,
            "matches": []
        }
        
        for match in matches:
            match_dict = {
                "sentence1_id": match.sentence1_id,
                "sentence2_id": match.sentence2_id,
                "sentence1_text": match.sentence1_text,
                "sentence2_text": match.sentence2_text,
                "document1": match.document1,
                "document2": match.document2,
                "similarity_score": round(match.similarity_score, 4)
            }
            matches_data["matches"].append(match_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matches_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(matches)} matches to {output_path}")

def main():
    """
    Main function to run simplified sentence matching
    """
    print("🔍 Starting Simplified Sentence Similarity Analysis")
    
    # Initialize matcher
    matcher = SentenceMatcher()
    
    # Find all similar sentences
    print("\n1. Finding similar sentences (≥ 0.85 similarity only)...")
    similar_sentences = matcher.find_similar_sentences()
    
    # Save matches to file
    print("\n2. Saving matches to file...")
    matcher.save_matches_to_file(similar_sentences)
    
    # Print simple summary
    print("\n📊 ANALYSIS SUMMARY")
    print("=" * 30)
    print(f"Total sentence matches found: {len(similar_sentences)}")
    
    # Show top 3 most similar sentence pairs
    if similar_sentences:
        print("\n🔥 TOP 3 MOST SIMILAR SENTENCE PAIRS:")
        print("=" * 50)
        for i, match in enumerate(similar_sentences[:3], 1):
            print(f"\n{i}. Similarity: {match.similarity_score:.4f}")
            print(f"   📄 {match.document1}")
            print(f"   📝 \"{match.sentence1_text[:80]}...\"")
            print(f"   📄 {match.document2}")
            print(f"   📝 \"{match.sentence2_text[:80]}...\"")
    
    print("\n✅ Simplified analysis complete!")

if __name__ == "__main__":
    main()