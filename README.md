# Question-Answer Store

Like key-value, but with questions and answers.

[![PyPI version](https://badge.fury.io/py/qa-store.svg)](https://badge.fury.io/py/qa-store)
[![CI](https://github.com/witt3rd/qa-store/actions/workflows/ci.yml/badge.svg)](https://github.com/witt3rd/qa-store/actions/workflows/ci.yml)
[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye.astral.sh)
[![GitHub license](https://img.shields.io/github/license/witt3rd/qa-store.svg)](https://github.com/witt3rd/qa-store/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/witt3rd/qa-store.svg)](https://github.com/witt3rd/qa-store/issues)
[![GitHub stars](https://img.shields.io/github/stars/witt3rd/qa-store.svg)](https://github.com/witt3rd/qa-store/stargazers)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/dt_public.svg?style=social&label=Follow%20%40dt_public)](https://twitter.com/dt_public)

A Python package for managing and querying a Question-Answer Knowledge Base using vector embeddings and tree structures.

## Features

- Store and retrieve question-answer pairs
- Generate question rewordings for improved retrieval
- Metadata support for filtering and additional information
- Vector-based similarity search for efficient querying
- Persistent storage using ChromaDB
- Tree structure for organizing questions hierarchically
- Automatic generation of question-answer pairs from text
- Priority-based question suggestion system
- Visualization of the question tree

## Installation

You can install `qa-store` using pip:

```bash
pip install qa-store
```

## Usage

Here's a quick example of how to use `qa-store`:

```python
from qa_store import QuestionAnswerSystem

# Initialize the Question Answer System
qas = QuestionAnswerSystem("qa_system.db", "qa_collection")

# Add a question to the tree
question_id = qas.add_question("What is the capital of France?")

# Answer the question
qas.answer_question(question_id, "Paris")

# Query the Knowledge Base
results = qas.query("What's the capital city of France?")
print(results[0]["answer"])  # Output: Paris

# Get unanswered questions
unanswered = qas.get_unanswered_questions()

# Get the next suggested question
next_question = qas.suggest_next_question()
```

## Advanced Usage Example

This example showcases the use of the QuestionAnswerSystem, including adding questions, answering them, and querying the knowledge base:

```python
from qa_store import QuestionAnswerSystem

# Initialize the Question Answer System
qas = QuestionAnswerSystem("qa_system.db", "qa_collection")

# Add a root question
root_id = qas.add_question("What are the main topics in computer science?")

# Add child questions
qas.add_question("What is machine learning?", parent_id=root_id)
qas.add_question("What are data structures?", parent_id=root_id)

# Answer some questions
qas.answer_question(root_id, "The main topics in computer science include algorithms, data structures, artificial intelligence, and software engineering.")

# Query the Knowledge Base
results = qas.query("What are the fundamental areas of computer science?", num_rewordings=2)

print("\nQuery results:")
for result in results:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Similarity: {result['similarity']:.2f}")
    print(f"Metadata: {result['metadata']}")
    print()

# Get the next suggested question
next_question = qas.suggest_next_question()
if next_question:
    print(f"Suggested next question: {next_question['question']}")
```

## Advanced Features

### Generating QA Pairs from Text

You can automatically generate question-answer pairs from a given text:

```python
from qa_store import QuestionAnswerKB

kb = QuestionAnswerKB()

text = """
Machine learning is a subset of artificial intelligence that focuses on
the development of algorithms and statistical models that enable
computer systems to improve their performance on a specific task through
experience.
"""

qa_pairs = kb.generate_qa_pairs(text)

for pair in qa_pairs:
    kb.add_qa(pair['q'], pair['a'])
```

### Visualizing the Question Tree

You can visualize the question tree structure:

```python
from qa_store import QuestionAnswerTree

tree = QuestionAnswerTree("qa_tree.db")
tree.visualize("question_tree")
```

This will generate a PNG image of the question tree.

For more detailed usage instructions, please refer to the [documentation](https://github.com/witt3rd/qa-store/wiki).

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Donald Thompson - [@dt_public](https://twitter.com/dt_public) - <witt3rd@witt3rd.com>

Project Link: [https://github.com/witt3rd/qa-store](https://github.com/witt3rd/qa-store)
