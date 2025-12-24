# ParableGPT

A large language model, based off Meta's open-source *Llama 3.1-8B-Instruct*, which uses a corpus of various religious texts to generate parables across different cultural traditions.

A project for Brian Greene's *Origins and Meaning* course, taken Fall 2025.

Uses RAG (Retrival-augmented generation) to enrich the prompt. We draw from the following four texts, sourced from https://github.com/Traves-Theberge/sacred-scriptures-mcp.
- Bible - Christianity
- Dhammapada - Buddhism
- Qur'an - Islam
- Tao Te Ching - Daoism / Confucianism

A detailed explanation and walkthrough of my data preparation and methods are detailed in `usage.ipynb`.

## Usage

To get the program working on your local machine, you will need Ollama installed on your computer alongside an installation of `Llama 3.1-8B-Instruct` or another Llama of choice.

Then, simply run:
```shell
python parableGPT.py
```
to start the session.

## An example parable
<img width="827" height="714" alt="Screenshot 2025-12-23 at 9 03 21 PM" src="https://github.com/user-attachments/assets/e36be877-2ec7-47b3-99ea-b59a847d8734" />
