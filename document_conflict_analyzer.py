import json
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentConflict:
    """Simplified data class for document conflicts"""
    doc1: str
    doc2: str
    conflict_count: int
    avg_similarity: float
    conflict_sentences: List[Dict[str, Any]]

class DocumentConflictAnalyzer:
    def __init__(self, matches_file: str = "analysis/sentence_matches.json"):
        """
        Initialize the simplified conflict analyzer
        """
        self.matches_file = Path(matches_file)
        self.matches_data = None
        
        self.load_matches()
    
    def load_matches(self):
        """Load sentence matches from JSON file"""
        if not self.matches_file.exists():
            raise FileNotFoundError(f"Matches file not found: {self.matches_file}")
        
        print("Loading sentence matches...")
        with open(self.matches_file, 'r', encoding='utf-8') as f:
            self.matches_data = json.load(f)
        
        print(f"Loaded {self.matches_data['total_matches']} sentence matches")
    
    def analyze_document_conflicts(self) -> List[DocumentConflict]:
        """
        Analyze conflicts between document pairs - simplified version
        """
        print("Analyzing document-level conflicts...")
        
        # Group matches by document pairs
        doc_pair_matches = {}
        
        for match in self.matches_data["matches"]:
            doc1, doc2 = match["document1"], match["document2"]
            
            # Skip matches within the same document
            if doc1 == doc2:
                continue
            
            # Create consistent pair key
            pair_key = tuple(sorted([doc1, doc2]))
            
            if pair_key not in doc_pair_matches:
                doc_pair_matches[pair_key] = []
            
            doc_pair_matches[pair_key].append(match)
        
        # Analyze each document pair
        conflicts = []
        for (doc1, doc2), matches in doc_pair_matches.items():
            conflict_count = len(matches)
            
            # Skip document pairs with fewer than 3 conflicts
            if conflict_count < 3:
                continue
                
            similarities = [match["similarity_score"] for match in matches]
            avg_similarity = sum(similarities) / len(similarities)
            
            conflict = DocumentConflict(
                doc1=doc1,
                doc2=doc2,
                conflict_count=conflict_count,
                avg_similarity=avg_similarity,
                conflict_sentences=matches  # Keep ALL conflicts, not just top 3
            )
            
            conflicts.append(conflict)
        
        # Sort by conflict count (most conflicts first)
        conflicts.sort(key=lambda x: (x.conflict_count, x.avg_similarity), reverse=True)
        
        print(f"Found {len(conflicts)} document pairs with conflicts (≥3 conflicts each)")
        return conflicts
    
    def save_conflict_analysis(self, conflicts: List[DocumentConflict]):
        """
        Save simplified conflict analysis to file
        """
        output_file = "analysis/document_conflicts.json"
        
        # Prepare simplified data for JSON serialization
        conflicts_data = []
        for conflict in conflicts:
            conflicts_data.append({
                "doc1": conflict.doc1,
                "doc2": conflict.doc2,
                "conflict_count": conflict.conflict_count,
                "avg_similarity": round(conflict.avg_similarity, 4),
                "top_conflicts": conflict.conflict_sentences  # All conflicts for expansion
            })
        
        output_data = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_document_pairs_with_conflicts": len(conflicts)
            },
            "document_conflicts": conflicts_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Simplified conflict analysis saved to: {output_file}")

def main():
    """
    Main function to run simplified document conflict analysis
    """
    print("📊 Starting Simplified Document Conflict Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DocumentConflictAnalyzer()
    
    # Analyze document conflicts
    conflicts = analyzer.analyze_document_conflicts()
    
    # Print summary
    print(f"\n📋 CONFLICT SUMMARY")
    print("=" * 30)
    print(f"Document pairs with conflicts: {len(conflicts)}")
    
    # Show top conflicting document pairs
    if conflicts:
        print(f"\n🔥 TOP 5 MOST CONFLICTED DOCUMENT PAIRS:")
        print("=" * 50)
        for i, conflict in enumerate(conflicts[:5], 1):
            print(f"\n{i}. {conflict.doc1} ↔ {conflict.doc2}")
            print(f"   Conflicts: {conflict.conflict_count} sentences")
            print(f"   Avg Similarity: {conflict.avg_similarity:.3f}")
    
    # Save analysis
    analyzer.save_conflict_analysis(conflicts)
    
    print(f"\n✅ Simplified analysis complete!")
    print(f"📄 Results saved to: analysis/document_conflicts.json")

if __name__ == "__main__":
    main()