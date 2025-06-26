import json
import numpy as np
from typing import List, Dict, Tuple, Any
from pathlib import Path
import time
from dataclasses import dataclass

@dataclass
class SentenceMatch:
    """Data class to represent a sentence match"""
    sentence1_id: str
    sentence2_id: str
    sentence1_text: str
    sentence2_text: str
    document1: str
    document2: str
    similarity_score: float
    match_type: str  # 'exact_duplicate', 'high_similarity', 'moderate_similarity'

class SentenceMatcher:
    def __init__(self, embeddings_file: str = "embeddings/sentence_embeddings.json"):
        """
        Initialize the sentence matcher with pre-computed embeddings
        """
        self.embeddings_file = Path(embeddings_file)
        self.embeddings_data = None
        self.embeddings_matrix = None
        self.sentence_lookup = {}
        
        # Similarity thresholds - much more strict
        self.exact_duplicate_threshold = 1.0  # Only perfect matches
        self.high_similarity_threshold = 0.95  # Very high similarity
        self.moderate_similarity_threshold = 0.85  # Minimum threshold to consider
        
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings from JSON file and prepare for similarity calculations"""
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
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def calculate_similarity_matrix(self) -> np.ndarray:
        """
        Calculate cosine similarity matrix for all sentence pairs
        This can be memory intensive for large datasets
        """
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
    
    def find_similar_sentences(self, min_similarity: float = 0.85) -> List[SentenceMatch]:
        """
        Find all sentence pairs with similarity above the threshold
        Only considers matches ≥ 0.85 similarity - nothing below this threshold will be processed or saved
        """
        
        print(f"Finding similar sentences with minimum similarity: {min_similarity}")
        print("⚠️  Note: Only matches ≥ 0.85 will be processed and saved")
        
        similarity_matrix = self.calculate_similarity_matrix()
        matches = []
        
        n_sentences = len(self.sentence_lookup)
        processed_pairs = 0
        skipped_pairs = 0
        
        # Iterate through upper triangle of similarity matrix (avoid duplicates)
        for i in range(n_sentences):
            for j in range(i + 1, n_sentences):
                similarity = similarity_matrix[i, j]
                
                # Hard threshold - only process if >= min_similarity
                if similarity >= min_similarity:
                    processed_pairs += 1
                    
                    # Determine match type with strict criteria
                    if similarity >= self.exact_duplicate_threshold:
                        match_type = "exact_duplicate"
                    elif similarity >= self.high_similarity_threshold:
                        match_type = "high_similarity"  # Likely same content, minor differences
                    else:
                        match_type = "potential_conflict"  # Similar enough to review, might be conflicting info
                    
                    # Create match object
                    match = SentenceMatch(
                        sentence1_id=self.sentence_lookup[i]["sentence_id"],
                        sentence2_id=self.sentence_lookup[j]["sentence_id"],
                        sentence1_text=self.sentence_lookup[i]["sentence_text"],
                        sentence2_text=self.sentence_lookup[j]["sentence_text"],
                        document1=self.sentence_lookup[i]["document"],
                        document2=self.sentence_lookup[j]["document"],
                        similarity_score=similarity,
                        match_type=match_type
                    )
                    
                    matches.append(match)
                else:
                    skipped_pairs += 1
            
            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{n_sentences} sentences - Found {processed_pairs} matches, Skipped {skipped_pairs} below threshold")
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        print(f"✅ Found {len(matches)} similar sentence pairs (all ≥ {min_similarity})")
        print(f"📊 Processed {processed_pairs} qualifying pairs, skipped {skipped_pairs} below threshold")
        return matches
    
    def find_matches_for_sentence(self, target_sentence: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find the most similar sentences to a given input sentence
        This is useful for search functionality
        """
        # This would require generating embedding for the target sentence
        # For now, we'll implement finding matches for an existing sentence by ID
        pass
    
    def find_matches_by_sentence_id(self, sentence_id: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find the most similar sentences to a sentence identified by its ID
        """
        # Find the sentence index
        target_index = None
        for idx, info in self.sentence_lookup.items():
            if info["sentence_id"] == sentence_id:
                target_index = idx
                break
        
        if target_index is None:
            print(f"Sentence ID {sentence_id} not found")
            return []
        
        # Calculate similarities
        target_embedding = self.embeddings_matrix[target_index]
        similarities = []
        
        for idx, embedding in enumerate(self.embeddings_matrix):
            if idx != target_index:  # Don't include the sentence itself
                similarity = self.cosine_similarity(target_embedding, embedding)
                similarities.append((idx, similarity))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:top_k]
        
        # Format results
        results = []
        for idx, similarity in top_matches:
            sentence_info = self.sentence_lookup[idx].copy()
            results.append((sentence_info, similarity))
        
        return results
    
    def group_matches_by_document_pairs(self, matches: List[SentenceMatch]) -> Dict[str, List[SentenceMatch]]:
        """
        Group similar sentences by document pairs to identify document-level conflicts
        """
        document_pairs = {}
        
        for match in matches:
            # Create a consistent key for document pairs
            doc1, doc2 = match.document1, match.document2
            pair_key = f"{min(doc1, doc2)} <-> {max(doc1, doc2)}"
            
            if pair_key not in document_pairs:
                document_pairs[pair_key] = []
            
            document_pairs[pair_key].append(match)
        
        # Sort each group by similarity score
        for pair_key in document_pairs:
            document_pairs[pair_key].sort(key=lambda x: x.similarity_score, reverse=True)
        
        return document_pairs
    
    def save_matches_to_file(self, matches: List[SentenceMatch], output_file: str = "analysis/sentence_matches.json", min_similarity: float = 0.85):
        """
        Save sentence matches to a JSON file for later analysis
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)  # Creates 'analysis' directory if it doesn't exist
        
        # Convert matches to serializable format
        matches_data = {
            "total_matches": len(matches),
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "minimum_similarity_threshold": min_similarity,
            "note": f"Only matches with similarity ≥ {min_similarity} are included",
            "thresholds": {
                "exact_duplicate": self.exact_duplicate_threshold,  # 1.0
                "high_similarity": self.high_similarity_threshold,  # 0.95
                "potential_conflict": self.moderate_similarity_threshold  # 0.85
            },
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
                "similarity_score": round(match.similarity_score, 4),
                "match_type": match.match_type
            }
            matches_data["matches"].append(match_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matches_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(matches)} matches to {output_path}")
    
    def generate_conflict_report(self, matches: List[SentenceMatch]) -> Dict[str, Any]:
        """
        Generate a summary report of potential conflicts
        """
        # Group by match type
        by_type = {"exact_duplicate": [], "high_similarity": [], "potential_conflict": []}
        for match in matches:
            by_type[match.match_type].append(match)
        
        # Group by document pairs
        document_pairs = self.group_matches_by_document_pairs(matches)
        
        # Identify policy conflicts (same policy, different versions)
        policy_conflicts = []
        for pair_key, pair_matches in document_pairs.items():
            doc1, doc2 = pair_key.split(" <-> ")
            
            # Check if these are different versions of the same policy
            # Look for version patterns in filenames
            base1 = doc1.replace("_v10", "").replace("_v20", "").replace("_v30", "").replace("_2023", "").replace("_2024", "")
            base2 = doc2.replace("_v10", "").replace("_v20", "").replace("_v30", "").replace("_2023", "").replace("_2024", "")
            
            if base1 == base2 and doc1 != doc2:
                policy_conflicts.extend(pair_matches)
        
        report = {
            "summary": {
                "total_matches": len(matches),
                "exact_duplicates": len(by_type["exact_duplicate"]),
                "high_similarity": len(by_type["high_similarity"]),
                "potential_conflicts": len(by_type["potential_conflict"]),
                "document_pairs_with_matches": len(document_pairs),
                "potential_policy_conflicts": len(policy_conflicts)
            },
            "top_conflicts": matches[:10],  # Top 10 highest similarity matches
            "document_pairs": document_pairs,
            "policy_conflicts": policy_conflicts
        }
        
        return report

def main():
    """
    Main function to demonstrate the sentence matching functionality
    """
    print("🔍 Starting Sentence Similarity Analysis")
    
    # Initialize matcher
    matcher = SentenceMatcher()
    
    # Find all similar sentences with strict 0.85 minimum
    print("\n1. Finding all similar sentences (≥ 0.85 similarity only)...")
    similar_sentences = matcher.find_similar_sentences(min_similarity=0.85)
    
    # Save matches to file
    print("\n2. Saving matches to file...")
    matcher.save_matches_to_file(similar_sentences, min_similarity=0.85)
    
    # Generate conflict report
    print("\n3. Generating conflict report...")
    report = matcher.generate_conflict_report(similar_sentences)
    
    # Print summary
    print("\n📊 SIMILARITY ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total sentence matches found: {report['summary']['total_matches']}")
    print(f"  • Exact duplicates (1.0): {report['summary']['exact_duplicates']}")
    print(f"  • High similarity (≥95%): {report['summary']['high_similarity']}")
    print(f"  • Potential conflicts (≥85%): {report['summary']['potential_conflicts']}")
    print(f"Document pairs with matches: {report['summary']['document_pairs_with_matches']}")
    print(f"Potential policy conflicts: {report['summary']['potential_policy_conflicts']}")
    
    # Show top 5 most similar sentence pairs
    if similar_sentences:
        print("\n🔥 TOP 5 MOST SIMILAR SENTENCE PAIRS:")
        print("=" * 50)
        for i, match in enumerate(similar_sentences[:5], 1):
            print(f"\n{i}. Similarity: {match.similarity_score:.3f} ({match.match_type})")
            print(f"   📄 {match.document1}")
            print(f"   📝 \"{match.sentence1_text[:100]}...\"")
            print(f"   📄 {match.document2}")
            print(f"   📝 \"{match.sentence2_text[:100]}...\"")
    
    # Save full report
    report_file = "analysis/conflict_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        # Make report JSON serializable
        serializable_report = {
            "summary": report["summary"],
            "top_conflicts": [
                {
                    "sentence1_id": m.sentence1_id,
                    "sentence2_id": m.sentence2_id,
                    "document1": m.document1,
                    "document2": m.document2,
                    "similarity_score": round(m.similarity_score, 4),
                    "match_type": m.match_type,
                    "sentence1_text": m.sentence1_text,
                    "sentence2_text": m.sentence2_text
                } for m in report["top_conflicts"]
            ]
        }
        json.dump(serializable_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full report saved to: {report_file}")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()