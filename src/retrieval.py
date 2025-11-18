"""
Retrieval system untuk RAG (Retrieval Augmented Generation)
"""
import json
import os
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher


class KnowledgeBase:
    """Manage local knowledge base untuk retrieval"""

    def __init__(self, kb_path: str = "./data/knowledge_base/local_db.json"):
        self.kb_path = kb_path
        self.kb: Dict[str, str] = {}
        self.load_kb()

    def load_kb(self) -> None:
        """Load knowledge base dari JSON file"""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                self.kb = json.load(f)
            print(f"✓ Loaded {len(self.kb)} entries from knowledge base")
        else:
            print(f"⚠ Knowledge base tidak ditemukan, membuat baru: {self.kb_path}")
            self.kb = {}
            self.save_kb()

    def save_kb(self) -> None:
        """Save knowledge base ke JSON file"""
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump(self.kb, f, ensure_ascii=False, indent=2)

    def add_entry(self, query: str, answer: str) -> None:
        """Add entry baru ke knowledge base"""
        query_key = query.lower().strip()
        self.kb[query_key] = answer
        self.save_kb()
        print(f"✓ Added new entry: {query_key}")

    def get_exact_match(self, query: str) -> Optional[str]:
        """Get exact match dari knowledge base"""
        query_key = query.lower().strip()
        return self.kb.get(query_key)

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity antara 2 string (0.0 - 1.0)"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def get_similar_matches(self, query: str, top_k: int = 3, threshold: float = 0.6) -> List[Tuple[str, str, float]]:
        """
        Get similar matches dari knowledge base berdasarkan similarity score

        Returns:
            List of (query, answer, similarity_score) tuples
        """
        query_lower = query.lower().strip()
        matches = []

        for kb_query, kb_answer in self.kb.items():
            similarity = self.calculate_similarity(query_lower, kb_query)
            if similarity >= threshold:
                matches.append((kb_query, kb_answer, similarity))

        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[2], reverse=True)

        return matches[:top_k]

    def retrieve(self, query: str, use_similarity: bool = True, top_k: int = 3) -> Dict:
        """
        Main retrieval method

        Returns:
            {
                'found': bool,
                'exact_match': bool,
                'answer': str or None,
                'similar_matches': list of tuples,
                'confidence': float
            }
        """
        result = {
            'found': False,
            'exact_match': False,
            'answer': None,
            'similar_matches': [],
            'confidence': 0.0
        }

        # Try exact match first
        exact_answer = self.get_exact_match(query)
        if exact_answer:
            result['found'] = True
            result['exact_match'] = True
            result['answer'] = exact_answer
            result['confidence'] = 1.0
            return result

        # Try similarity search
        if use_similarity:
            similar_matches = self.get_similar_matches(query, top_k=top_k)
            if similar_matches:
                result['found'] = True
                result['exact_match'] = False
                result['similar_matches'] = similar_matches
                # Use the best match
                best_match = similar_matches[0]
                result['answer'] = best_match[1]
                result['confidence'] = best_match[2]
                return result

        return result

    def show_all(self) -> None:
        """Display semua entries di knowledge base"""
        if not self.kb:
            print("Knowledge base kosong")
            return

        print("\n=== Knowledge Base ===")
        for i, (query, answer) in enumerate(self.kb.items(), 1):
            print(f"\n{i}. Query: {query}")
            print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        print(f"\nTotal: {len(self.kb)} entries")


class RAGRetriever:
    """
    Complete RAG retrieval system
    Combines local knowledge base with optional web search fallback
    """

    def __init__(self, kb_path: str = "./data/knowledge_base/local_db.json", use_web_fallback: bool = False):
        self.kb = KnowledgeBase(kb_path)
        self.use_web_fallback = use_web_fallback

    def retrieve_context(self, query: str, top_k: int = 3) -> Dict:
        """
        Retrieve context untuk query

        Returns:
            {
                'context': str,
                'sources': list,
                'confidence': float,
                'method': str  # 'exact', 'similar', 'web', or 'none'
            }
        """
        # Try local knowledge base
        kb_result = self.kb.retrieve(query, use_similarity=True, top_k=top_k)

        if kb_result['found']:
            context = kb_result['answer']

            # Add similar matches to context if available
            if kb_result['similar_matches'] and len(kb_result['similar_matches']) > 1:
                context += "\n\nRelated information:"
                for _, answer, score in kb_result['similar_matches'][1:]:
                    if score > 0.5:  # Only include highly similar matches
                        context += f"\n- {answer[:200]}..."

            return {
                'context': context,
                'sources': ['local_knowledge_base'],
                'confidence': kb_result['confidence'],
                'method': 'exact' if kb_result['exact_match'] else 'similar'
            }

        # Web fallback (if enabled)
        if self.use_web_fallback:
            # TODO: Implement web search fallback
            # For now, return no context
            pass

        # No context found
        return {
            'context': f"Maaf, saya tidak menemukan informasi tentang: {query}",
            'sources': [],
            'confidence': 0.0,
            'method': 'none'
        }


# Usage example
if __name__ == "__main__":
    # Test KnowledgeBase
    kb = KnowledgeBase()
    kb.show_all()

    print("\n=== Testing Retrieval ===")
    test_queries = [
        "cara install laravel",
        "bagaimana install laravel",  # similar
        "apa itu eloquent",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = kb.retrieve(query)
        print(f"Found: {result['found']}")
        print(f"Confidence: {result['confidence']:.2f}")
        if result['answer']:
            print(f"Answer: {result['answer'][:100]}...")
