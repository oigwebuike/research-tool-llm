# Research Tool with Streamlit, LangChain, and FAISS

## Overview

This project is a research tool built using Streamlit, LangChain, and FAISS to extract, embed, and retrieve context from online articles. It allows users to input URLs for articles, processes the content into embeddings, and utilizes an OpenAI language model to answer user queries based on the embedded context. This application can be particularly useful for researchers and analysts looking to quickly find relevant information from online sources.

## Features

- **URL Input:** Accepts up to three URLs for processing.
- **Document Embedding:** Utilizes OpenAI embeddings to transform article content into vector representations.
- **Document Splitting:** Splits content for efficient retrieval using recursive character text splitting.
- **Query Answering:** Answers questions based on the context retrieved from processed articles.
- **Cache Clearing:** Clears cached data to ensure fresh processing of new inputs.

## Requirements

- Python 3.8+
- OpenAI API Key
- Environment variables for sensitive data (`.env` file)

### Key Libraries

- **Streamlit:** User interface for interactive queries.
- **LangChain:** Framework for chaining multiple language model operations.
- **FAISS:** Vector store for efficient similarity search on embeddings.
- **dotenv:** Loads environment variables securely from a `.env` file.



## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies
Install required libraries:
```bash
pip install -r requirements.txt
```


### 3. Set Up the .env File
Create a .env file in the root directory and add your OpenAI API key:


```bash
OPENAI_API_KEY=your_openai_api_key
```


### 4. Run the Application
Start the Streamlit application:
```bash
streamlit run url.py
```


## Usage

- **Load URLs:** Enter up to three URLs in the sidebar to process.
- **Process URLs:** Click ``Process URLs`` to retrieve and embed the content.
- **Ask Questions:** Enter a question in the main input box to receive an answer based on the embedded content.
- **Clear Cache:** Use the ``Refresh Program`` button in the sidebar to clear cached data and reload URLs.


## Project Structure
- **app.py:** Main application file containing the Streamlit interface and logic for embedding, retrieval, and question answering.
- **vector_file:** File path where embeddings are stored locally for retrieval.


## Code Details
### Key Functions
- **clear_cache:** Clears cached data and resources for a fresh start.
- **format_docs:** Formats documents for display after retrieval.
- **Embedding with FAISS:** Embeds documents and saves them locally for faster similarity search.


### Key Components
- **LangChain:** Handles language model interactions, prompt templates, and retrieval chaining.
- **OpenAI Embeddings:** Converts content into embeddings for similarity-based retrieval.
- **RecursiveCharacterTextSplitter:** Splits text into manageable chunks for efficient retrieval.
- **FAISS Vector Store:** Stores and retrieves document embeddings efficiently.

## Notes
- **This tool currently supports up to three URLs per run for processing.**
- **To avoid reprocessing data, ensure the vector file path (``vector_file``) remains unchanged if you want to keep the previous embeddings.**
- **Environment Variables: Ensure the ``.env`` file is set up with your API key to access OpenAI’s services.**



## Future Enhancements
- **Multiple File Uploads:** Support for more than three URLs or other file types.
- **Improved Caching:** Enhanced caching mechanisms to reduce redundant processing.
- **Additional Language Models:** Options for using other models based on user preferences.



## License
- **This project is open-source. Feel free to contribute or report issues!** 
```bash

```
