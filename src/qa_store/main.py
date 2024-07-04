from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions


class QuestionAnswerKB:
    def __init__(self, collection_name: str = "qa_kb") -> None:
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )

    def add_qa(
        self, question: str, answer: Any, metadata: Dict[str, Any] = None
    ) -> None:
        metadata = metadata or {}
        metadata["answer"] = str(answer)
        self.collection.add(
            documents=[question],
            metadatas=[metadata],
            ids=[f"qa_{self.collection.count()}"],
        )

    def query(
        self,
        question: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_params = {
            "query_texts": [question],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if metadata_filter:
            query_params["where"] = metadata_filter

        results = self.collection.query(**query_params)

        matches = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            matches.append(
                {
                    "question": doc,
                    "answer": metadata["answer"],
                    "metadata": {k: v for k, v in metadata.items() if k != "answer"},
                    "similarity": 1 - distance,
                }
            )

        return matches

    def update_answer(self, question: str, new_answer: Any) -> None:
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
        return self.collection.get(include=["documents"])["documents"]

    def clear(self) -> None:
        self.collection.delete(ids=self.collection.get(include=["ids"])["ids"])

    def reset_database(self) -> None:
        """
        Drop the entire collection and recreate it, effectively resetting the database.
        """
        try:
            # Delete the existing collection
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' has been deleted.")
        except ValueError:
            # If the collection doesn't exist, a ValueError is raised
            print(f"Collection '{self.collection_name}' did not exist.")

        # Recreate the collection
        self.collection = self.client.create_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )
        print(f"Collection '{self.collection_name}' has been recreated.")
