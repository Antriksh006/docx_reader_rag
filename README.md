# DOCX Query Chatbot

This project is a chatbot that can answer questions based on the content of a DOCX file. It extracts text from the document, generates embeddings for the text, and performs a similarity search to provide relevant answers to user queries. The chatbot utilizes the Groq API for chat completions.

## Features

- Extracts text from DOCX files.
- Splits the text into manageable chunks.
- Generates embeddings using the Hugging Face model.
- Uses Chroma for vector storage and similarity search.
- Provides responses based on user queries with contextual information.

## Requirements

This project requires the following Python libraries:

- `python-docx`
- `langchain`
- `langchain-community`
- `groq`

## Usage

This project requires you to have a **GROQ** Api key, A DOCX file whose path will be used in the python file, and a Query for this chatbot to process in the file.

```
ENTER_YOUR_QUERY_HERE: Type in the question you want to ask.
ENTER_YOUR_GROQ_API_KEY_HERE: Input your Groq API key.
ENTER_FILE_PATH: Provide the file path to the DOCX file you want to use for context.
```
