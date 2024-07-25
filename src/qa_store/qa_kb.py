import os
from textwrap import dedent
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from litellm import completion
from loguru import logger

from qa_store.helpers import get_json_list

#

load_dotenv()

DB_DIR = os.getenv("DB_DIR", "db")
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "qa_kb")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
REWORDING_MODEL_NAME = os.getenv("REWORDING_MODEL_NAME", "gpt-4o-mini")
QA_PAIRS_MODEL_NAME = os.getenv("QA_PAIRS_MODEL_NAME", "gpt-4o-mini")

#

REWORDING_PROMPT = dedent("""
Rephrase the [ORIGINAL QUESTION] provided in {num_rewordings} distinct ways,
ensuring the meaning and context remain intact.
Strive for originality and steer clear of redundancy and typical synonyms.
Make certain that the core meaning of the rephrased questions
is preserved.
Provide the reworded questions in a list format, with each question
on a new line, without numbers or bullets.

## Example:
[ORIGINAL QUESTION]: What is the capital of Italy?
[REWRITTEN QUESTIONS]:
Which city serves as the capital of Italy?
Can you name the capital city of Italy?
What is the name of Italy's capital?

## Input:
[ORIGINAL QUESTION]: {question}
[REWRITTEN QUESTIONS]:
""").strip()


QA_PAIRS_PROMPT = dedent("""
Derive various questions (along with their answers) that this text implies:
Format your result as a JSON list of objects, where each object has a 'q' key
for the question and an 'a' key for the answer.
""").strip()

QA_PAIRS_JSON_PROMPT = dedent("""
You will be given a text containing question and answer pairs in various
formats. Your task is to extract these pairs and convert them into a JSON array
of objects. Each object should have two keys: 'q' for the question and 'a' for
the answer.
The output should be structured as follows:
[
{
"q": "Question 1",
"a": "Answer 1"
},
{
"q": "Question 2",
"a": "Answer 2"
},
...
]
Ensure that:

The entire output is a valid JSON array.
Each question-answer pair is represented as a separate object within the array.
The 'q' key contains the full question text.
The 'a' key contains the full answer text.
Any formatting or structure from the original text is removed, presenting clean
question and answer strings.
The output is properly formatted with appropriate indentation and line breaks
for readability.

Process the entire input text and include all question-answer pairs found in the
output JSON array.
""").strip()

#


class QuestionAnswerKB:
    def __init__(
        self,
        db_dir: str = DB_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.client = chromadb.PersistentClient(path=db_dir)
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

        logger.info(f"QuestionAnswerKB initialized for collection '{collection_name}'.")

    def _parse_qa_pairs_as_json(
        self,
        text: str,
        model: str = QA_PAIRS_MODEL_NAME,
    ) -> List[Dict[str, str]]:
        """
        Parse a string of question-answer pairs as JSON.

        Args:
            text (str): The string of question-answer pairs.

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a
            question and its corresponding answer.
        """
        text = text.strip()
        if not text:
            return []

        try:
            qa_pairs = get_json_list(text)
            if isinstance(qa_pairs, list):
                return qa_pairs
        except Exception:
            pass

        retries = 0
        previous_error = None

        while retries < 3:
            user_content = (
                dedent(f"""
                Previous error:\n\n{previous_error}\n\n
                """).strip()
                if previous_error
                else ""
            )
            user_content += "Input text:\n\n" + text

            try:
                response = completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": QA_PAIRS_JSON_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                )
                content: str = response.choices[0].message.content  # type: ignore
                qa_pairs = get_json_list(content)
                if not isinstance(qa_pairs, list):
                    raise ValueError(
                        "Expected a list of QA pairs, but got a different format."
                    )
            except Exception as e:
                previous_error = e
                retries += 1
                print(f"Error: {str(e)}")

    def generate_qa_pairs(
        self,
        input_text: str,
        model: str = QA_PAIRS_MODEL_NAME,
    ) -> List[Dict[str, str]]:
        """
        Generate a set of Question-Answer pairs from the given input text.

        Args:
            input_text (str): The text to generate questions and answers from.
            model (str): The model to use for completion. Defaults to "gpt-3.5-turbo".

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing a
            question and its corresponding answer.
        """
        prompt = f"[INPUT TEXT]\n{input_text}\n\n[JSON OUTPUT]\n"

        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": QA_PAIRS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            content: str = response.choices[0].message.content  # type: ignore
            qa_pairs = self._parse_qa_pairs_as_json(content, model=model)

            for pair in qa_pairs:
                if not isinstance(pair, dict) or "q" not in pair or "a" not in pair:
                    raise ValueError("Invalid QA pair format in the response.")
            return qa_pairs

        except Exception as e:
            print(str(e))
            return []

    def generate_rewordings(
        self,
        question: str,
        num_rewordings: int,
        model: str = REWORDING_MODEL_NAME,
    ) -> List[str]:
        """
        Generate rewordings of a given question using GPT-4.

        Parameters
        ----------
        question : str
            The question string.
        num_rewordings : int
            Number of rewordings to generate.

        Returns
        -------
        List[str]
            A list of reworded questions, including the original question.
        """
        if num_rewordings == 0:
            return [question]

        prompt = REWORDING_PROMPT.format(
            num_rewordings=num_rewordings,
            question=question,
        )

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        rewordings = response.choices[0].message.content.strip().split("\n")
        questions = [question] + rewordings
        questions = [q.strip() for q in questions]
        for i, q in enumerate(questions):
            logger.trace(f"Reworded question {i+1}: {q}")
        return questions

    def add_qa(
        self,
        question: str | list[str],
        answer: Any = None,
        metadata: Dict[str, Any] | None = None,
        num_rewordings: int = 0,
    ) -> Set[str]:
        """
        Add a question-answer pair to the KB, including rewordings if specified.

        Parameters
        ----------
        question : str or list of str
            The question string or list of question strings.
        answer : Any
            The answer (converted to string).
        metadata : dict of {str: Any} or None, optional
            Optional metadata about the QA pair. Default is None.
        num_rewordings : int, optional
            Number of question rewordings to generate and index. Default is 0.

        Returns
        -------
        set of str
            A set of all questions (original and rewordings) that were indexed.

        """
        metadata = metadata or {}
        metadata["answer"] = answer if answer else ""

        if isinstance(question, str) and num_rewordings > 0:
            questions = self.generate_rewordings(question, num_rewordings)
        elif isinstance(question, list) and num_rewordings > 0:
            questions = question
        else:
            questions = [question]

        documents = []
        metadatas = []
        ids = []

        for i, q in enumerate(questions):
            logger.trace(f"Adding question: {q}")
            documents.append(q)
            metadatas.append(metadata.copy())
            ids.append(f"qa_{self.collection.count()}_{i}")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        return set(questions)

    def query(
        self,
        question: str | list[str],
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        num_rewordings: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query the KB for answers to a given question, with optional metadata
        filtering and question rewordings.

        Parameters
        ----------
        question : str or list of str
            The question string or list of question strings.
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
        if isinstance(question, str) and num_rewordings > 0:
            questions = self.generate_rewordings(question, num_rewordings)
        elif isinstance(question, list) and num_rewordings > 0:
            questions = question
        else:
            questions = [question]

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
        final_results = sorted(
            unique_results,
            key=lambda x: x["similarity"],
            reverse=True,
        )[:n_results]
        for i, result in enumerate(final_results):
            logger.trace(f"Query result {i+1}: {result}")
        return final_results

    def update_answer(
        self,
        question: str,
        new_answer: Any,
    ) -> None:
        """
        Update the answer to a question in the KB.

        Parameters
        ----------
        question : str
            The question string.
        new_answer : Any
            The new answer (converted to string).

        Raises
        ------
        ValueError
            If the question is not found in the KB.
        """

        results = self.collection.query(
            query_texts=[question],
            n_results=1,
            include=["metadatas"],
        )

        if results["ids"][0]:
            id = results["ids"][0][0]
            metadata = results["metadatas"][0][0]
            metadata["answer"] = str(new_answer)
            self.collection.update(
                ids=[id],
                documents=[question],
                metadatas=[metadata],
            )
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
        Clear the entire collection, effectively deleting all question-answer
        pairs.
        """
        self.collection.delete(ids=self.collection.get(include=["ids"])["ids"])
        logger.trace(f"Collection '{self.collection_name}' has been cleared.")

    def reset_database(self) -> None:
        """
        Drop the entire collection and recreate it, effectively resetting the
        database.
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

    def add_tree_question(
        self, question: str, tree_id: int, answer: Optional[str] = None
    ):
        metadata = {"tree_id": tree_id, "from_tree": True}
        self.add_qa(question, answer, metadata=metadata)

    def get_tree_questions(self) -> List[Dict[str, Any]]:
        return self.query(
            "", metadata_filter={"from_tree": True}, n_results=self.collection.count()
        )

    def update_tree_question(self, tree_id: int, answer: str):
        results = self.query("", metadata_filter={"tree_id": tree_id}, n_results=1)
        if results:
            question = results[0]["question"]
            self.update_answer(question, answer)


if __name__ == "__main__":
    kb = QuestionAnswerKB()
    # QA pairs
    results = kb.generate_qa_pairs(
        dedent("""
        Machine learning is a subset of artificial intelligence that focuses on
        the development of algorithms and statistical models that enable
        computer systems to improve their performance on a specific task through
        experience. It involves training models on large datasets to recognize
        patterns and make predictions or decisions without being explicitly
        programmed. Common types of machine learning include supervised
        learning, unsupervised learning, and reinforcement learning.
        Applications of machine learning are widespread, including in areas such
        as image and speech recognition, natural language processing,
        recommendation systems, and autonomous vehicles.
        """).strip()
    )
    for i, pair in enumerate(results, 1):
        print(f"Q{i}: {pair['q']}")
        print(f"A{i}: {pair['a']}")
        print()
