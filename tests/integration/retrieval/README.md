# Retrieval Integration Tests

Use this folder for tests that exercise the real retrieval orchestrator across
multiple collaborating stages.

These tests should usually:
- use the real `Retriever`
- use fake vector DBs, candidate retrievers, transformers, or rerankers
- verify orchestration decisions such as dense-only use, fusion behavior,
  filter handling, and reranking order

This folder is especially useful for guarding refactors in the retrieval stack,
because many failures there are not pure unit bugs: they happen when otherwise
reasonable components stop cooperating correctly.
