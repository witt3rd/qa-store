import logging
from textwrap import dedent

import pytest
from qa_store import QuestionAnswerKB, QuestionAnswerSystem, QuestionAnswerTree

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    logger.info("Setting up QuestionAnswerKB")
    db_path = tmp_path_factory.mktemp("data")

    qakb = QuestionAnswerKB(
        db_dir=str(db_path),
        collection_name="test_kb",
    )
    qakb.reset_database()
    yield qakb
    logger.info("Tearing down QuestionAnswerKB")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    logger.info("Setting up QuestionTree")
    db_path = tmp_path_factory.mktemp("data") / "test_tree.db"
    t = QuestionAnswerTree(str(db_path))
    yield t
    logger.info("Tearing down QuestionTree")


@pytest.fixture(scope="module")
def qa_system(tmp_path_factory):
    logger.info("Setting up QASystem")
    db_path = tmp_path_factory.mktemp("data")
    qas = QuestionAnswerSystem(str(db_path), "test_collection")
    yield qas
    logger.info("Tearing down QASystem")


# Existing tests for QuestionAnswerKB...


def test_add_and_query(kb):
    logger.info("Testing add_qa and query functionality")
    kb.add_qa("What is the capital of Germany?", "Berlin")
    kb.add_qa("Who is the president of the USA?", "Donald Trump")
    kb.add_qa("What is the answer to life, the universe, and everything?", 42)

    results = kb.query("What is the capital of Germany?")
    logger.debug("Query result: %s", results)
    assert results[0]["answer"] == "Berlin"

    results = kb.query("Who is the president of the USA?")
    logger.debug("Query result: %s", results)
    assert results[0]["answer"] == "Donald Trump"

    results = kb.query("What is the answer to life, the universe, and everything?")
    logger.debug("Query result: %s", results)
    assert results[0]["answer"] == 42


def test_update_answer(kb):
    logger.info("Testing update_answer functionality")
    kb.update_answer("What is the capital of Germany?", "Munich")
    results = kb.query("What is the capital of Germany?")
    logger.debug("Query result after update: %s", results)
    assert results[0]["answer"] == "Munich"


def test_metadata(kb):
    logger.info("Testing metadata functionality")
    kb.add_qa(
        "What is the capital of France?", "Paris", metadata={"source": "Wikipedia"}
    )

    results = kb.query("What is the capital of France?")
    logger.debug("Query result with metadata: %s", results)
    assert results[0]["metadata"]["source"] == "Wikipedia"

    results = kb.query(
        "What is the capital of France?", metadata_filter={"source": "Wikipedia"}
    )
    logger.debug("Query result with metadata filter: %s", results)
    assert results[0]["metadata"]["source"] == "Wikipedia"

    results = kb.query(
        "What is the capital of France?", metadata_filter={"source": "github"}
    )
    logger.debug("Query result with non-matching metadata filter: %s", results)
    assert not results


def test_rewordings(kb):
    logger.info("Testing rewordings functionality")
    kb.add_qa("What is your favorite TV series", "Band of Brothers", num_rewordings=5)
    results = kb.query("What is your favorite TV show")
    logger.debug("Query result with rewording:  %s", results)
    assert results[0]["answer"] == "Band of Brothers"

    kb.add_qa("What is your favorite book?", "The Great Gatsby")
    results = kb.query("Do you have a book you like the most?", num_rewordings=5)
    logger.debug("Query result with rewording:  %s", results)
    assert results[0]["answer"] == "The Great Gatsby"


def test_generate_qa_pairs(kb):
    logger.info("Testing generate_qa_pairs functionality")
    text = dedent("""
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

    results = kb.generate_qa_pairs(text)
    logger.info("Generated %d QA pairs", len(results))
    assert len(results) > 0
    for i, pair in enumerate(results, 1):
        logger.info("Q%d: %s: %s", i, pair["q"], pair["a"])


def test_question_tree(tree):
    logger.info("Testing QuestionTree functionality")

    # Add questions
    root_id = tree.add_question("What is the main goal?")
    child1_id = tree.add_question("What are the objectives?", parent_id=root_id)
    child2_id = tree.add_question("What is the timeline?", parent_id=root_id)

    # Test tree structure
    root = tree.get_question(root_id)
    assert root.question == "What is the main goal?"
    assert len(tree.get_children(root_id)) == 2

    # Test updating answer
    tree.update_answer(root_id, "To complete the project successfully")
    updated_root = tree.get_question(root_id)
    assert updated_root.answer == "To complete the project successfully"

    # Test priorities
    tree.calculate_priorities()
    high_priority = tree.get_high_priority_questions(2)
    assert len(high_priority) == 2
    assert high_priority[0].id == root_id

    # Test unanswered questions
    unanswered = tree.get_unanswered_questions()
    assert len(unanswered) == 2
    assert unanswered[0].id in [child1_id, child2_id]


def test_qa_system(qa_system):
    logger.info("Testing QASystem functionality")

    # Add questions
    q1_id = qa_system.add_question("What is the project scope?")
    _ = qa_system.add_question("Who are the stakeholders?", parent_id=q1_id)

    # Test unanswered questions
    unanswered = qa_system.get_unanswered_questions()
    assert len(unanswered) == 2
    assert unanswered[0]["question"] in [
        "What is the project scope?",
        "Who are the stakeholders?",
    ]

    results = qa_system.query("What is the scope of the project?")
    assert len(results) > 0
    assert results[0]["question"] == "What is the project scope?"
    assert results[0]["answer"] == ""
    metadata = results[0]["metadata"]
    assert metadata["tree_id"] == q1_id

    # Answer a question
    qa_system.answer_question(
        q1_id, "The project scope includes developing a new software system"
    )

    # Test answered questions
    answered = qa_system.get_answered_questions()
    assert len(answered) == 1
    assert answered[0]["question"] == "What is the project scope?"
    assert (
        answered[0]["answer"]
        == "The project scope includes developing a new software system"
    )

    # Test suggestion
    suggestion = qa_system.suggest_next_question()
    assert suggestion is not None
    assert suggestion["question"] == "Who are the stakeholders?"

    # Test synchronization
    qa_system.sync_kb_to_tree()
    qa_system.sync_tree_to_kb()

    # Verify synchronization
    kb_questions = qa_system.kb.get_tree_questions()
    assert len(kb_questions) == 2
    assert any(
        q["answer"] == "The project scope includes developing a new software system"
        for q in kb_questions
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
