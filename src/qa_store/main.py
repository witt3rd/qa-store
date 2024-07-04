from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions


class QuestionAnswerKB:
    def __init__(self, collection_name: str = "qa_kb"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def add_qa(self, question: str, answer: Any, metadata: Dict[str, Any] = None):
        """
        Add a question-answer pair to the KB.

        :param question: The question string
        :param answer: The answer (converted to string)
        :param metadata: Optional metadata about the QA pair
        """
        metadata = metadata or {}
        metadata["answer"] = str(answer)
        self.collection.add(
            documents=[question],
            metadatas=[metadata],
            ids=[f"qa_{self.collection.count()}"],
        )

    def query(self, question: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the KB for answers to a given question.

        :param question: The query question
        :param n_results: Number of results to return
        :return: List of matching QA pairs
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            matches.append(
                {
                    "question": doc,
                    "answer": metadata["answer"],
                    "metadata": {k: v for k, v in metadata.items() if k != "answer"},
                    "similarity": 1 - distance,  # Convert distance to similarity
                }
            )

        return matches

    def update_answer(self, question: str, new_answer: Any):
        """
        Update the answer for a specific question.

        :param question: The exact question to update
        :param new_answer: The new answer
        """
        results = self.collection.query(
            query_texts=[question], n_results=1, include=["metadatas"]
        )

        if results["ids"][0]:
            id = results["ids"][0][0]
            metadata = results["metadatas"][0][0]
            metadata["answer"] = str(new_answer)
            self.collection.update(ids=[id], documents=[question], metadatas=[metadata])
        else:
            raise ValueError("Question not found in KB")

    def get_all_questions(self) -> List[str]:
        """
        Get all questions in the KB.

        :return: List of all questions
        """
        return self.collection.get(include=["documents"])["documents"]

    def clear(self):
        """
        Clear all QA pairs from the KB.
        """
        self.collection.delete(ids=self.collection.get(include=["ids"])["ids"])
