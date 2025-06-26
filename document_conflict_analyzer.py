import json
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentConflict:
    """Data class to represent conflicts between two documents"""
    doc1: str
    doc2: str
    conflict_count: int
    avg_similarity: float
    max_similarity: float
    conflict_sentences: List[Dict[str, Any]]
    relationship_type: str  # 'version_conflict', 'duplicate_content', 'related_policy'

@dataclass
class DocumentRecommendation:
    """Data class for document removal/consolidation recommendations"""
    document: str
    recommendation_type: str  # 'remove_outdated', 'consolidate', 'review_conflicts'
    reason: str
    conflicting_documents: List[str]
    confidence_score: float
    evidence: List[str]

class DocumentConflictAnalyzer:
    def __init__(self, matches_file: str = "analysis/sentence_matches.json"):
        """
        Initialize the document conflict analyzer
        """
        self.matches_file = Path(matches_file)
        self.matches_data = None
        self.document_conflicts = {}
        self.document_metadata = {}
        
        self.load_matches()
        self.extract_document_metadata()
    
    def load_matches(self):
        """Load sentence matches from JSON file"""
        if not self.matches_file.exists():
            raise FileNotFoundError(f"Matches file not found: {self.matches_file}")
        
        print("Loading sentence matches...")
        with open(self.matches_file, 'r', encoding='utf-8') as f:
            self.matches_data = json.load(f)
        
        print(f"Loaded {self.matches_data['total_matches']} sentence matches")
    
    def extract_document_metadata(self):
        """Extract metadata from document names (version, date, department)"""
        documents = set()
        
        # Collect all unique documents
        for match in self.matches_data["matches"]:
            documents.add(match["document1"])
            documents.add(match["document2"])
        
        for doc in documents:
            metadata = self.parse_document_name(doc)
            self.document_metadata[doc] = metadata
        
        print(f"Extracted metadata for {len(documents)} documents")
    
    def parse_document_name(self, filename: str) -> Dict[str, Any]:
        """
        Parse document filename to extract version, date, and policy type
        """
        # Remove file extension
        name = filename.replace('.md', '')
        
        # Extract version information
        version_match = re.search(r'_v(\d)(\d)', name)
        version = None
        if version_match:
            major, minor = version_match.groups()
            version = f"v{major}.{minor}"
        
        # Extract year information
        year_match = re.search(r'_(\d{4})', name)
        year = int(year_match.group(1)) if year_match else None
        
        # Determine base policy name (remove version and year info)
        base_name = name
        base_name = re.sub(r'_v\d\d', '', base_name)  # Remove version
        base_name = re.sub(r'_\d{4}', '', base_name)  # Remove year
        base_name = re.sub(r'_updated', '', base_name)  # Remove "updated" prefix
        base_name = re.sub(r'^\d+_', '', base_name)   # Remove number prefix
        
        # Determine department based on keywords
        department = self.classify_department(name)
        
        return {
            "filename": filename,
            "base_name": base_name,
            "version": version,
            "year": year,
            "department": department,
            "is_updated": "updated" in name.lower(),
            "has_version": version is not None,
            "has_year": year is not None
        }
    
    def classify_department(self, filename: str) -> str:
        """Classify document by department based on filename"""
        filename_lower = filename.lower()
        
        if any(keyword in filename_lower for keyword in ['hr', 'employee', 'benefits', 'onboarding', 'orientation', 'performance', 'review']):
            return "HR"
        elif any(keyword in filename_lower for keyword in ['it', 'security', 'cyber', 'tech']):
            return "IT"
        elif any(keyword in filename_lower for keyword in ['engineering', 'dev', 'code', 'software']):
            return "Engineering"
        elif any(keyword in filename_lower for keyword in ['sales', 'marketing', 'client', 'customer']):
            return "Sales/Marketing"
        elif any(keyword in filename_lower for keyword in ['finance', 'expense', 'vendor', 'supplier']):
            return "Finance"
        elif any(keyword in filename_lower for keyword in ['legal', 'privacy', 'data']):
            return "Legal"
        elif any(keyword in filename_lower for keyword in ['safety', 'covid']):
            return "Safety"
        else:
            return "General"
    
    def analyze_document_conflicts(self) -> List[DocumentConflict]:
        """
        Analyze conflicts between document pairs
        """
        print("Analyzing document-level conflicts...")
        
        # Group matches by document pairs
        doc_pair_matches = {}
        
        for match in self.matches_data["matches"]:
            doc1, doc2 = match["document1"], match["document2"]
            
            # Skip matches within the same document - these are not conflicts
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
            max_similarity = max(similarities)
            
            # Determine relationship type
            relationship_type = self.classify_document_relationship(doc1, doc2, matches)
            
            conflict = DocumentConflict(
                doc1=doc1,
                doc2=doc2,
                conflict_count=conflict_count,
                avg_similarity=avg_similarity,
                max_similarity=max_similarity,
                conflict_sentences=matches,
                relationship_type=relationship_type
            )
            
            conflicts.append(conflict)
        
        # Sort by conflict count (most conflicts first)
        conflicts.sort(key=lambda x: (x.conflict_count, x.avg_similarity), reverse=True)
        
        print(f"Found {len(conflicts)} document pairs with conflicts (≥3 conflicts each, between different documents)")
        return conflicts
    
    def classify_document_relationship(self, doc1: str, doc2: str, matches: List[Dict]) -> str:
        """
        Classify the relationship between two conflicting documents
        """
        meta1 = self.document_metadata[doc1]
        meta2 = self.document_metadata[doc2]
        
        # Check if same base policy with different versions
        if meta1["base_name"] == meta2["base_name"]:
            if meta1["has_version"] and meta2["has_version"]:
                return "version_conflict"
            elif meta1["has_year"] and meta2["has_year"] and meta1["year"] != meta2["year"]:
                return "version_conflict"
            else:
                return "duplicate_content"
        
        # Check if high similarity suggests duplicate content
        avg_similarity = sum(match["similarity_score"] for match in matches) / len(matches)
        if avg_similarity > 0.92:
            return "duplicate_content"
        
        # Same department, related policies
        if meta1["department"] == meta2["department"]:
            return "related_policy"
        
        return "related_policy"
    
    def generate_removal_recommendations(self, conflicts: List[DocumentConflict]) -> List[DocumentRecommendation]:
        """
        Generate recommendations for which documents to remove or consolidate
        """
        print("Generating removal recommendations...")
        
        recommendations = []
        processed_docs = set()
        
        # Group conflicts by base policy name for version analysis
        policy_groups = {}
        for conflict in conflicts:
            if conflict.relationship_type == "version_conflict":
                base_name1 = self.document_metadata[conflict.doc1]["base_name"]
                base_name2 = self.document_metadata[conflict.doc2]["base_name"]
                
                if base_name1 == base_name2:
                    if base_name1 not in policy_groups:
                        policy_groups[base_name1] = set()
                    policy_groups[base_name1].add(conflict.doc1)
                    policy_groups[base_name1].add(conflict.doc2)
        
        # Analyze version conflicts
        for base_name, docs in policy_groups.items():
            docs_list = list(docs)
            if len(docs_list) > 1:
                recommendation = self.analyze_version_group(docs_list, conflicts)
                if recommendation:
                    recommendations.append(recommendation)
                    processed_docs.update(docs_list)
        
        # Analyze high-conflict document pairs
        for conflict in conflicts:
            if conflict.doc1 in processed_docs or conflict.doc2 in processed_docs:
                continue
            
            # Only consider pairs with significant conflicts (≥3) and high similarity
            if conflict.conflict_count >= 3 and conflict.avg_similarity > 0.90:
                # High duplicate content
                older_doc = self.determine_older_document(conflict.doc1, conflict.doc2)
                if older_doc:
                    evidence = [
                        f"{conflict.conflict_count} conflicting sentences",
                        f"Average similarity: {conflict.avg_similarity:.3f}",
                        f"Relationship: {conflict.relationship_type}"
                    ]
                    
                    recommendation = DocumentRecommendation(
                        document=older_doc,
                        recommendation_type="remove_outdated",
                        reason=f"High content overlap with newer version",
                        conflicting_documents=[conflict.doc1 if older_doc == conflict.doc2 else conflict.doc2],
                        confidence_score=min(0.95, conflict.avg_similarity),
                        evidence=evidence
                    )
                    
                    recommendations.append(recommendation)
                    processed_docs.add(older_doc)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        print(f"Generated {len(recommendations)} removal recommendations")
        return recommendations
    
    def analyze_version_group(self, docs: List[str], conflicts: List[DocumentConflict]) -> DocumentRecommendation:
        """
        Analyze a group of document versions and recommend which to keep/remove
        """
        # Sort documents by version/year
        doc_metadata = [(doc, self.document_metadata[doc]) for doc in docs]
        
        # Sort by version first, then by year
        def sort_key(item):
            doc, meta = item
            version_order = 0
            if meta["version"]:
                version_num = float(meta["version"].replace('v', ''))
                version_order = version_num
            
            year_order = meta["year"] if meta["year"] else 0
            return (version_order, year_order)
        
        doc_metadata.sort(key=sort_key)
        
        if len(doc_metadata) < 2:
            return None
        
        # Keep the newest version, recommend removing older ones
        newest_doc = doc_metadata[-1][0]
        older_docs = [item[0] for item in doc_metadata[:-1]]
        
        # Calculate evidence
        total_conflicts = 0
        for conflict in conflicts:
            if conflict.doc1 in docs and conflict.doc2 in docs:
                total_conflicts += conflict.conflict_count
        
        evidence = [
            f"Multiple versions of same policy found",
            f"Total conflicts between versions: {total_conflicts}",
            f"Keeping newest: {newest_doc}",
            f"Older versions: {', '.join(older_docs)}"
        ]
        
        return DocumentRecommendation(
            document=", ".join(older_docs),
            recommendation_type="remove_outdated",
            reason=f"Superseded by newer version: {newest_doc}",
            conflicting_documents=[newest_doc],
            confidence_score=0.90,
            evidence=evidence
        )
    
    def determine_older_document(self, doc1: str, doc2: str) -> str:
        """
        Determine which document is older based on version/year
        """
        meta1 = self.document_metadata[doc1]
        meta2 = self.document_metadata[doc2]
        
        # Compare by year first
        if meta1["year"] and meta2["year"]:
            return doc1 if meta1["year"] < meta2["year"] else doc2
        
        # Compare by version
        if meta1["version"] and meta2["version"]:
            v1 = float(meta1["version"].replace('v', ''))
            v2 = float(meta2["version"].replace('v', ''))
            return doc1 if v1 < v2 else doc2
        
        # Prefer documents without "updated" in name
        if meta1["is_updated"] != meta2["is_updated"]:
            return doc2 if meta1["is_updated"] else doc1
        
        # Default to alphabetical (earlier filename)
        return doc1 if doc1 < doc2 else doc2
    
    def save_conflict_analysis(self, conflicts: List[DocumentConflict], recommendations: List[DocumentRecommendation]):
        """
        Save document conflict analysis to file
        """
        output_file = "analysis/document_conflicts.json"
        
        # Prepare data for JSON serialization
        conflicts_data = []
        for conflict in conflicts:
            conflicts_data.append({
                "doc1": conflict.doc1,
                "doc2": conflict.doc2,
                "conflict_count": conflict.conflict_count,
                "avg_similarity": round(conflict.avg_similarity, 4),
                "max_similarity": round(conflict.max_similarity, 4),
                "relationship_type": conflict.relationship_type,
                "top_conflicts": conflict.conflict_sentences[:3]  # Show top 3 conflicts
            })
        
        recommendations_data = []
        for rec in recommendations:
            recommendations_data.append({
                "document": rec.document,
                "recommendation_type": rec.recommendation_type,
                "reason": rec.reason,
                "conflicting_documents": rec.conflicting_documents,
                "confidence_score": round(rec.confidence_score, 3),
                "evidence": rec.evidence
            })
        
        output_data = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filtering_criteria": {
                "minimum_conflicts_per_pair": 3,
                "excludes_same_document_matches": True,
                "note": "Only shows conflicts between different documents with at least 3 conflicting sentences"
            },
            "summary": {
                "total_document_pairs_with_conflicts": len(conflicts),
                "total_removal_recommendations": len(recommendations),
                "high_confidence_recommendations": len([r for r in recommendations if r.confidence_score > 0.8])
            },
            "document_conflicts": conflicts_data,
            "removal_recommendations": recommendations_data,
            "document_metadata": self.document_metadata
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Document conflict analysis saved to: {output_file}")

def main():
    """
    Main function to run document conflict analysis
    """
    print("📊 Starting Document Conflict Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DocumentConflictAnalyzer()
    
    # Analyze document conflicts
    conflicts = analyzer.analyze_document_conflicts()
    
    # Generate removal recommendations
    recommendations = analyzer.generate_removal_recommendations(conflicts)
    
    # Print summary
    print(f"\n📋 DOCUMENT CONFLICT SUMMARY")
    print("=" * 50)
    print(f"Document pairs with conflicts: {len(conflicts)}")
    print(f"Removal recommendations: {len(recommendations)}")
    print(f"High confidence recommendations: {len([r for r in recommendations if r.confidence_score > 0.8])}")
    
    # Show top conflicting document pairs
    print(f"\n🔥 TOP 5 MOST CONFLICTED DOCUMENT PAIRS:")
    print("=" * 50)
    for i, conflict in enumerate(conflicts[:5], 1):
        print(f"\n{i}. {conflict.doc1} ↔ {conflict.doc2}")
        print(f"   Conflicts: {conflict.conflict_count} sentences")
        print(f"   Avg Similarity: {conflict.avg_similarity:.3f}")
        print(f"   Relationship: {conflict.relationship_type}")
    
    # Show removal recommendations
    if recommendations:
        print(f"\n🗑️  REMOVAL RECOMMENDATIONS:")
        print("=" * 50)
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. RECOMMEND: {rec.recommendation_type.upper()}")
            print(f"   Document(s): {rec.document}")
            print(f"   Reason: {rec.reason}")
            print(f"   Confidence: {rec.confidence_score:.1%}")
            print(f"   Conflicts with: {', '.join(rec.conflicting_documents)}")
    
    # Save analysis
    analyzer.save_conflict_analysis(conflicts, recommendations)
    
    print(f"\n✅ Document conflict analysis complete!")
    print(f"📄 Detailed results saved to: analysis/document_conflicts.json")

if __name__ == "__main__":
    main()