# PDFQA

PDF-Based Question Answering System using Embeddings
This project is an interactive question-answering system built with Python. It extracts and embeds text from PDF documents, stores the embeddings in a MySQL database, and enables intelligent Q&A based on semantic similarity using transformer-based models.

Features
Extracts structured text and tables from PDF files.

Cleans and normalizes extracted content to enhance QA performance.

Embeds text using the sentence-transformers/all-MiniLM-L6-v2 model.

Stores and loads embeddings from MySQL database.

Enables interactive natural language question answering with accuracy scoring.

Supports intelligent grouping of relevant results (headings, bullets, etc.).

Detects and avoids duplicated or garbage content.

Requirements
Install the dependencies using:
pip install -r requirements.txt

Required Libraries
torch

transformers

sentence-transformers

pdfplumber

mysql-connector-python

pandas

nltk

numpy

scikit-learn

 MySQL Setup
Create a database named file_embeddings_db and a table like this:
CREATE DATABASE file_embeddings_db;

USE file_embeddings_db;

CREATE TABLE embeddings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sentence TEXT,
    embedding LONGTEXT
);


 Notes
All text is embedded and compared semantically using cosine similarity.

Repeated runs will not duplicate entries in the database.

The system supports structured content including bullet lists and tables.

Content normalization helps with noisy or scanned PDFs.
