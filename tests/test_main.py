import sys

from loguru import logger
from qa_store import QuestionAnswerKB

# Remove the default handler
logger.remove()

# Add a new handler with TRACE level
logger.add(sys.stderr, level="TRACE")

#


def test_qa_store():
    kb = QuestionAnswerKB()
    kb.reset_database()

    # Add some question-answer pairs
    kb.add_qa("What is the capital of Germany?", "Berlin")
    kb.add_qa("Who is the president of the USA?", "Joe Biden")
    kb.add_qa("What is the answer to life, the universe, and everything?", 42)

    # Query the KB
    results = kb.query("What is the capital of Germany?")
    assert results[0]["answer"] == "Berlin"

    results = kb.query("Who is the president of the USA?")
    assert results[0]["answer"] == "Joe Biden"

    results = kb.query("What is the answer to life, the universe, and everything?")
    assert results[0]["answer"] == "42"

    # Update an answer
    kb.update_answer("What is the capital of Germany?", "Munich")
    results = kb.query("What is the capital of Germany?")
    assert results[0]["answer"] == "Munich"

    # Add some metadata
    kb.add_qa(
        "What is the capital of France?", "Paris", metadata={"source": "Wikipedia"}
    )
    results = kb.query("What is the capital of France?")
    assert results[0]["metadata"]["source"] == "Wikipedia"

    results = kb.query(
        "What is the capital of France?",
        metadata_filter={"source": "Wikipedia"},
    )
    assert results[0]["metadata"]["source"] == "Wikipedia"

    results = kb.query(
        "What is the capital of France?",
        metadata_filter={"source": "github"},
    )
    assert not results

    # Rewordings
    kb.add_qa("What is your favorite TV series", "Band of Brothers", num_rewordings=5)
    results = kb.query("What is your favorite TV show")
    assert results[0]["answer"] == "Band of Brothers"

    kb.add_qa("What is your favorite book?", "The Great Gatsby")
    results = kb.query("Do you have a book you like the most?", num_rewordings=5)
    assert results[0]["answer"] == "The Great Gatsby"

    print("All tests passed!")
