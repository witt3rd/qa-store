import os
from typing import Any, Dict, List, Optional

from qa_store.qa_kb import QuestionAnswerKB
from qa_store.qa_tree import QuestionAnswerTree


class QuestionAnswerSystem:
    def __init__(self, db_dir: str, kb_collection_name: str):
        db_path = os.path.join(db_dir, f"{kb_collection_name}.db")
        print(db_path)
        self.tree = QuestionAnswerTree(db_path)
        self.kb = QuestionAnswerKB(
            db_dir=db_dir,
            collection_name=kb_collection_name,
        )

    def add_question(self, question: str, parent_id: Optional[int] = None) -> int:
        tree_id = self.tree.add_question(question, parent_id)
        self.kb.add_tree_question(question, tree_id)
        return tree_id

    def answer_question(self, question_id: int, answer: str):
        self.tree.update_answer(question_id, answer)
        self.kb.update_tree_question(question_id, answer)

    def get_unanswered_questions(self) -> List[Dict[str, Any]]:
        return [
            {"id": q.id, "question": q.question, "parent_id": q.parent_id}
            for q in self.tree.get_unanswered_questions()
        ]

    def get_answered_questions(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": q.id,
                "question": q.question,
                "answer": q.answer,
                "parent_id": q.parent_id,
            }
            for q in self.tree.get_answered_questions()
        ]

    def sync_kb_to_tree(self):
        kb_questions = self.kb.get_tree_questions()
        for q in kb_questions:
            tree_id = q["metadata"]["tree_id"]
            if not self.tree.is_answered(tree_id) and q["answer"] is not None:
                self.tree.update_answer(tree_id, q["answer"])

    def sync_tree_to_kb(self):
        tree_questions = self.tree.get_answered_questions()
        for q in tree_questions:
            self.kb.update_tree_question(q.id, q.answer)

    def suggest_next_question(self) -> Optional[Dict[str, Any]]:
        # Ensure the tree is built and priorities are calculated
        if not self.tree.root:
            self.tree.build_tree()
        self.tree.calculate_priorities()

        # Get high priority questions
        high_priority_questions = self.tree.get_high_priority_questions()

        # Filter for unanswered questions
        unanswered_high_priority = [q for q in high_priority_questions if not q.answer]

        if not unanswered_high_priority:
            return None

        # Return the highest priority unanswered question
        suggested_question = unanswered_high_priority[0]
        return {
            "id": suggested_question.id,
            "question": suggested_question.question,
            "parent_id": suggested_question.parent_id,
            "priority": suggested_question.priority,
        }

    def query(
        self,
        question: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        num_rewordings: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for answers to a given question.

        Parameters
        ----------
        question : str
            The question to query.
        n_results : int, optional
            Number of results to return (default is 5).
        metadata_filter : dict, optional
            Optional metadata filter (default is None).
        num_rewordings : int, optional
            Number of question rewordings to generate and query (default is 0).

        Returns
        -------
        list of dict
            A list of dictionaries containing the question, answer, metadata,
            and similarity score for each result.
        """
        return self.kb.query(
            question=question,
            n_results=n_results,
            metadata_filter=metadata_filter,
            num_rewordings=num_rewordings,
        )
