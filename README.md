# Question-Answer Store

Like key-value, but with questions and answers.

[![PyPI version](https://badge.fury.io/py/qa-store.svg)](https://badge.fury.io/py/qa-store)
[![GitHub license](https://img.shields.io/github/license/witt3rd/qa-store.svg)](https://github.com/witt3rd/qa-store/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/witt3rd/qa-store.svg)](https://github.com/witt3rd/qa-store/issues)
[![GitHub stars](https://img.shields.io/github/stars/witt3rd/qa-store.svg)](https://github.com/witt3rd/qa-store/stargazers)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/dt_public.svg?style=social&label=Follow%20%40dt_public)](https://twitter.com/dt_public)

A Python package for managing and querying a Question-Answer Knowledge Base using vector embeddings.

## Features

- Store and retrieve question-answer pairs
- Generate question rewordings for improved retrieval
- Metadata support for filtering and additional information
- Vector-based similarity search for efficient querying
- Persistent storage using ChromaDB

## Installation

You can install `qa-store` using pip:

```bash
pip install qa-store
```

## Usage

Here's a quick example of how to use `qa-store`:

```python
from qa_store import QuestionAnswerKB

# Initialize the Knowledge Base
kb = QuestionAnswerKB()

# Add a question-answer pair
kb.add_qa("What is the capital of France?", "Paris", metadata={"source": "Geography 101"})

# Query the Knowledge Base
results = kb.query("What's the capital city of France?")
print(results[0]["answer"])  # Output: Paris

# Update an answer
kb.update_answer("What is the capital of France?", "Paris (City of Light)")

# Query with metadata filter
results = kb.query("capital of France", metadata_filter={"source": "Geography 101"})
```

## Advanced Usage Example

This example showcases the use of question rewordings for both adding QA pairs and querying the knowledge base:

```python
from qa_store import QuestionAnswerKB

# Initialize the Knowledge Base
kb = QuestionAnswerKB()

# Add a question-answer pair with rewordings
original_question = "What is the best way to learn programming?"
answer = "The best way to learn programming is through consistent practice, working on real projects, and continuous learning."

added_questions = kb.add_qa(
    question=original_question,
    answer=answer,
    metadata={"topic": "education", "field": "computer science"},
    num_rewordings=3
)

print("Added questions:")
for q in added_questions:
    print(f"- {q}")

# Now let's query the Knowledge Base with a different phrasing
query_question = "How can I become proficient in coding?"

results = kb.query(
    question=query_question,
    n_results=2,
    metadata_filter={"topic": "education"},
    num_rewordings=2
)

print("\nQuery results:")
for result in results:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Similarity: {result['similarity']:.2f}")
    print(f"Metadata: {result['metadata']}")
    print()
```

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
