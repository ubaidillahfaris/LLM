"""
Generate QA pairs dari scraped content dan handle user approval
"""
import json
import os
from typing import List, Dict
from datetime import datetime


class QAGenerator:
    """Generate question-answer pairs dari raw content"""

    def __init__(self, model_manager=None):
        self.model_manager = model_manager
        self.generated_qa = []
        self.approved_qa = []
        self.rejected_qa = []

    def generate_qa_from_content(self, content: Dict) -> List[Dict]:
        """
        Generate QA pairs dari scraped content

        Args:
            content: Dict dengan keys 'title' dan 'content'

        Returns:
            List of QA pairs
        """
        qa_pairs = []

        # Strategy 1: Title sebagai question
        if 'title' in content and 'content' in content:
            # Convert title ke question format
            title = content['title']
            question = self._title_to_question(title)

            qa_pairs.append({
                'id': f"gen_{len(self.generated_qa)}_{datetime.now().timestamp()}",
                'question': question,
                'answer': content['content'][:500],  # Limit answer length
                'source': content.get('source', 'unknown'),
                'topic': content.get('topic', 'general'),
                'generated_at': datetime.now().isoformat(),
                'approved': None,  # None = pending, True = approved, False = rejected
                'confidence': 0.5
            })

        # Strategy 2: Extract dari content patterns
        # Look for "How to", "What is", etc. patterns
        content_text = content.get('content', '')

        # Extract code examples if present
        if 'Code:' in content_text:
            code_sections = [s.strip() for s in content_text.split('Code:') if s.strip()]
            for i, code in enumerate(code_sections[:3]):  # Limit to 3 examples
                if len(code) > 50:
                    qa_pairs.append({
                        'id': f"code_{len(self.generated_qa)}_{i}_{datetime.now().timestamp()}",
                        'question': f"Contoh code untuk {content.get('title', 'topic ini')}?",
                        'answer': code[:500],
                        'source': content.get('source', 'unknown'),
                        'topic': content.get('topic', 'code_example'),
                        'generated_at': datetime.now().isoformat(),
                        'approved': None,
                        'confidence': 0.6
                    })

        self.generated_qa.extend(qa_pairs)
        return qa_pairs

    def _title_to_question(self, title: str) -> str:
        """Convert title ke question format"""
        title = title.strip()

        # Common patterns
        if title.lower().startswith(('how', 'what', 'why', 'when', 'where')):
            # Already a question
            return title if title.endswith('?') else title + '?'

        # Convert to question
        question_starters = {
            'installation': 'Bagaimana cara install',
            'configuration': 'Bagaimana cara configure',
            'usage': 'Bagaimana cara menggunakan',
            'creating': 'Bagaimana cara membuat',
            'introduction': 'Apa itu',
            'overview': 'Apa itu',
        }

        for key, starter in question_starters.items():
            if key in title.lower():
                return f"{starter} {title}?"

        # Default
        return f"Bagaimana cara kerja {title}?"

    def batch_generate_from_scraped_data(self, scraped_data: List[Dict]) -> List[Dict]:
        """Generate QA pairs dari batch of scraped content"""
        all_qa = []

        for content in scraped_data:
            qa_pairs = self.generate_qa_from_content(content)
            all_qa.extend(qa_pairs)

        print(f"✅ Generated {len(all_qa)} QA pairs from {len(scraped_data)} content items")
        return all_qa

    def mark_as_approved(self, qa_id: str):
        """Mark QA pair as approved"""
        for qa in self.generated_qa:
            if qa['id'] == qa_id:
                qa['approved'] = True
                qa['approved_at'] = datetime.now().isoformat()
                self.approved_qa.append(qa)
                print(f"✅ Approved: {qa_id}")
                return True

        return False

    def mark_as_rejected(self, qa_id: str, reason: str = None):
        """Mark QA pair as rejected"""
        for qa in self.generated_qa:
            if qa['id'] == qa_id:
                qa['approved'] = False
                qa['rejected_at'] = datetime.now().isoformat()
                qa['rejection_reason'] = reason
                self.rejected_qa.append(qa)
                print(f"❌ Rejected: {qa_id}")
                return True

        return False

    def edit_qa(self, qa_id: str, new_question: str = None, new_answer: str = None):
        """Edit QA pair before approval"""
        for qa in self.generated_qa:
            if qa['id'] == qa_id:
                if new_question:
                    qa['question'] = new_question
                    qa['edited'] = True
                if new_answer:
                    qa['answer'] = new_answer
                    qa['edited'] = True
                qa['edited_at'] = datetime.now().isoformat()
                print(f"✏️  Edited: {qa_id}")
                return True

        return False

    def get_pending_qa(self) -> List[Dict]:
        """Get QA pairs yang belum di-review"""
        return [qa for qa in self.generated_qa if qa['approved'] is None]

    def get_approved_qa(self) -> List[Dict]:
        """Get approved QA pairs"""
        return [qa for qa in self.generated_qa if qa['approved'] is True]

    def get_rejected_qa(self) -> List[Dict]:
        """Get rejected QA pairs"""
        return [qa for qa in self.generated_qa if qa['approved'] is False]

    def save_to_training_dataset(self, filepath: str, approved_only: bool = True):
        """
        Save QA pairs ke training dataset format

        Args:
            filepath: Output file path
            approved_only: Only save approved QA pairs
        """
        qa_to_save = self.get_approved_qa() if approved_only else self.generated_qa

        # Convert to training format
        training_data = []
        for qa in qa_to_save:
            training_data.append({
                'id': len(training_data) + 1,
                'question': qa['question'],
                'answer': qa['answer'],
                'category': qa.get('topic', 'general'),
                'difficulty': 'intermediate',  # Can be adjusted
                'source': qa.get('source', 'generated'),
                'metadata': {
                    'generated_at': qa.get('generated_at'),
                    'approved_at': qa.get('approved_at'),
                    'edited': qa.get('edited', False)
                }
            })

        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(training_data)} QA pairs to {filepath}")
        return training_data

    def export_review_data(self, filepath: str):
        """Export all generated QA for review"""
        review_data = {
            'total': len(self.generated_qa),
            'pending': len(self.get_pending_qa()),
            'approved': len(self.get_approved_qa()),
            'rejected': len(self.get_rejected_qa()),
            'qa_pairs': self.generated_qa
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)

        print(f"✅ Exported review data to {filepath}")
        return review_data

    def load_review_data(self, filepath: str):
        """Load previously exported review data"""
        with open(filepath, 'r', encoding='utf-8') as f:
            review_data = json.load(f)

        self.generated_qa = review_data['qa_pairs']

        # Rebuild approved/rejected lists
        self.approved_qa = self.get_approved_qa()
        self.rejected_qa = self.get_rejected_qa()

        print(f"✅ Loaded {len(self.generated_qa)} QA pairs")
        print(f"   Pending: {len(self.get_pending_qa())}")
        print(f"   Approved: {len(self.approved_qa)}")
        print(f"   Rejected: {len(self.rejected_qa)}")

        return review_data

    def get_stats(self) -> Dict:
        """Get statistics about generated QA"""
        return {
            'total_generated': len(self.generated_qa),
            'pending_review': len(self.get_pending_qa()),
            'approved': len(self.get_approved_qa()),
            'rejected': len(self.get_rejected_qa()),
            'approval_rate': len(self.get_approved_qa()) / len(self.generated_qa) * 100 if self.generated_qa else 0
        }


# Example usage
if __name__ == "__main__":
    generator = QAGenerator()

    # Example content
    sample_content = {
        'title': 'Eloquent ORM Introduction',
        'content': 'Eloquent ORM adalah Active Record pattern di Laravel...',
        'source': 'laravel_docs',
        'topic': 'eloquent'
    }

    # Generate QA
    qa_pairs = generator.generate_qa_from_content(sample_content)

    print(f"Generated {len(qa_pairs)} QA pairs:")
    for qa in qa_pairs:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer'][:100]}...")

    # Show stats
    print(f"\nStats: {generator.get_stats()}")
