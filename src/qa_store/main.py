import os
from textwrap import dedent
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from litellm import completion
from loguru import logger

#

load_dotenv()

DB_DIR = os.getenv("DB_DIR", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
REWORDING_MODEL_NAME = os.getenv("REWORDING_MODEL_NAME", "gpt-4-turbo")

#


class QuestionAnswerKB:
    def __init__(
        self,
        collection_name: str = "qa_kb",
        openai_api_key: str | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

        # Set OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "OpenAI API key must be provided or set as an environment variable."
            )
        logger.info(f"QuestionAnswerKB initialized for collection '{collection_name}'.")

    def generate_rewordings(
        self,
        question: str,
        num_rewordings: int,
    ) -> List[str]:
        """
        Generate rewordings of a given question using GPT-4.

        Args:
            question (str): The question string.
            num_rewordings (int): Number of rewordings to generate.

        Returns:
            List[str]: A list of reworded questions, including the original question.
        """
        if num_rewordings == 0:
            return [question]
        prompt = dedent(f"""
            Please reword the following question in {num_rewordings}
            different ways, maintaining its original meaning. Provide
            only the reworded questions, one per line (no numbers or
            bullets).

            Original question: {question}

            Rewordings:
            """).strip()

        response = completion(
            model=REWORDING_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        rewordings = response.choices[0].message.content.strip().split("\n")
        questions = [question] + rewordings
        return questions

    def add_qa(
        self,
        question: str,
        answer: Any,
        metadata: Dict[str, Any] | None = None,
        num_rewordings: int = 0,
    ) -> Set[str]:
        """
        Add a question-answer pair to the KB, including rewordings if specified.

        Args:
            question (str): The question string.
            answer (Any): The answer (converted to string).
            metadata (Dict[str, Any] | None, optional): Optional metadata about
            the QA pair. Defaults to None.
            num_rewordings (int, optional): Number of question rewordings to
            generate and index. Defaults to 0.

        Returns:
            Set[str]: A set of all questions (original and rewordings) that were
            indexed.

        """
        metadata = metadata or {}
        metadata["answer"] = str(answer)

        questions = self.generate_rewordings(question, num_rewordings)

        documents = []
        metadatas = []
        ids = []

        for i, q in enumerate(questions):
            logger.trace(f"Adding question: {q}")
            documents.append(q)
            metadatas.append(metadata.copy())
            ids.append(f"qa_{self.collection.count()}_{i}")

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return set(questions)

    def query(
        self,
        question: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        num_rewordings: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query the KB for answers to a given question, with optional metadata
        filtering and question rewordings.

        Args:
            question (str): The question string.
            n_results (int, optional): Number of results to return. Defaults to 5.
            metadata_filter (Optional[Dict[str, Any]], optional): Optional metadata
            filter. Defaults to None.
            num_rewordings (int, optional): Number of question rewordings to
            generate and query. Defaults to 0.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the question,
            answer, metadata, and similarity score for each result.
        """
        questions = self.generate_rewordings(question, num_rewordings)

        all_results = []
        for q in questions:
            logger.trace(f"Querying question: {q}")
            query_params = {
                "query_texts": [q],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }

            if metadata_filter:
                query_params["where"] = metadata_filter

            results = self.collection.query(**query_params)

            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                all_results.append(
                    {
                        "question": doc,
                        "answer": metadata["answer"],
                        "metadata": {
                            k: v for k, v in metadata.items() if k != "answer"
                        },
                        "similarity": 1 - distance,
                    }
                )

        # Deduplicate results based on the answer
        seen_answers = set()
        unique_results = []
        for result in all_results:
            if result["answer"] not in seen_answers:
                seen_answers.add(result["answer"])
                unique_results.append(result)

        # Sort by similarity and return top n_results
        return sorted(
            unique_results,
            key=lambda x: x["similarity"],
            reverse=True,
        )[:n_results]

    def update_answer(self, question: str, new_answer: Any) -> None:
        """
        Update the answer to a question in the KB.

        Args:
            question (str): The question string.
            new_answer (Any): The new answer (converted to string).

        Raises:
            ValueError: If the question is not found in the KB.
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
        logger.info(f"Answer to question '{question}' has been updated.")

    def get_all_questions(self) -> List[str]:
        """
        Get all questions in the KB.

        Returns:
            List[str]: A list of all questions in the KB.
        """
        return self.collection.get(include=["documents"])["documents"]

    def clear(self) -> None:
        """
        Clear the entire collection, effectively deleting all question-answer pairs.
        """
        self.collection.delete(ids=self.collection.get(include=["ids"])["ids"])
        logger.trace(f"Collection '{self.collection_name}' has been cleared.")

    def reset_database(self) -> None:
        """
        Drop the entire collection and recreate it, effectively resetting the database.
        """
        try:
            # Delete the existing collection
            self.client.delete_collection(self.collection_name)
            logger.trace(f"Collection '{self.collection_name}' has been deleted.")
        except ValueError:
            # If the collection doesn't exist, a ValueError is raised
            logger.trace(f"Collection '{self.collection_name}' did not exist.")

        # Recreate the collection
        self.collection = self.client.create_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )
        logger.trace(f"Collection '{self.collection_name}' has been recreated.")
