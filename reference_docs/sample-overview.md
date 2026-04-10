# Sample reference: multi-agent research

This file ships with the project so `python ingest.py` has material to index.

## Microsoft Agent Framework

The Microsoft Agent Framework (MAF) lets developers define agents with instructions,
tools, and optional structured outputs. Agents can be composed into workflows or called
sequentially from application code.

## Retrieval-Augmented Generation

RAG improves factual grounding by retrieving relevant passages before generation.
A vector database such as ChromaDB stores chunked documents; queries return the most
similar chunks for the model to cite or summarize.

## Planning pattern

A planner model decomposes a vague user goal into sub-tasks or sub-questions.
Downstream agents then execute specialized retrieval or actions per sub-question.
