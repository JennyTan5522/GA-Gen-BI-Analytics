import os
import re
from collections import defaultdict
from typing import List, Tuple

import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, inspect
from config.logger import get_logger

from langchain.schema.document import Document
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

from src.excel_loader.excel import UnstructuredExcelLoader

logger = get_logger(__name__)

class SBERTEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
def load_data_into_vector_db(excel_structured_documents, collection_name: str):
    """
    Loads structured Excel documents into a Chroma vector database.

    Args:
        excel_structured_documents (list): A list of structured documents extracted from an Excel file, 
                                           typically in a format suitable for embedding-based search.
        collection_name (str): The name of the collection where the documents will be stored in the Chroma database.

    Returns:
        Chroma: An instance of the Chroma vector database containing the embedded documents.
    """

    persist_directory = "./chroma_db"
    chroma_client = Chroma(persist_directory=persist_directory)

    # Check if the collection exists and delete it
    if collection_name in [col.name for col in chroma_client._client.list_collections()]:
        chroma_client._client.delete_collection(collection_name)

    try:
        chroma = Chroma.from_documents(
            documents = excel_structured_documents,
            collection_name = collection_name,
            embedding =  SBERTEmbeddingFunction(),
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_directory,
        )
    except Exception as e:
        logger.error(f"Error loading data into vector database: {e}")

    return chroma

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by:
    - Removing duplicate rows.
    - Replacing spaces in column names with underscores.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    df_cleaned.drop_duplicates(inplace=True)
    df_cleaned.columns = df_cleaned.columns.str.replace(" ", "_")
    return df_cleaned

def process_df(df: pd.DataFrame, table_name: str, sql_engine) -> Tuple[pd.DataFrame, SQLDatabase, str]:
    """
    Cleans a DataFrame and stores it in an SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        table_name (str): The name of the table.

    Returns:
        Tuple[pd.DataFrame, str, object, str]: 
            - The cleaned DataFrame.
            - The database URL.
            - The SQLAlchemy engine object.
            - The table schema as a string.
    """
    df_cleaned = clean_dataframe(df)
    df_cleaned.to_sql(table_name, sql_engine, if_exists="replace", index=False)
    inspector = inspect(sql_engine)

    schema_details = [f"{col['name']} ({col['type']})" for col in inspector.get_columns(table_name)]
    schema = "\n".join(schema_details)
    return df_cleaned, schema

def parse_table(text: str) -> pd.DataFrame:
    """
    Parses a table represented as a string and converts it into a Pandas DataFrame.
    
    Args:
        text (str): The table text with rows separated by newlines and columns separated by '|'.
    
    Returns:
        pd.DataFrame: A structured DataFrame extracted from the text.
    """
    rows = text.strip().split("\n")
    parsed_rows = []
   
    for row in rows:
        columns = row.split("|") # Remove empty first/last splits
        columns = [col.strip() for col in columns]
       
        if not parsed_rows or columns != parsed_rows[0]:  # Ensure only the first header row is used
            parsed_rows.append(columns)

    if len(parsed_rows) > 1:
        header = parsed_rows[0]
        clean_header = [col if col.strip().upper() != "N/A" else f"Col_{i+1}" for i, col in enumerate(header)]
        df = pd.DataFrame(parsed_rows[1:], columns=clean_header) 
        df.drop(columns=["row-id"], inplace=True, errors="ignore")
        df = df.drop(columns=[''])
       
    return df

def group_table_elements(excel_docs: List[Document], sql_engine: SQLDatabase) -> dict:
    """
    Groups table elements based on their table reference and extracts structured data.
    
    Args:
        excel_docs (List[Document]): List of document objects containing table metadata and content.
    
    Returns:
        dict: Dictionary mapping table references to their corresponding DataFrames.
    """
    table_groups = defaultdict(list)
    results = {}

    for doc in excel_docs:
        table_ref = doc.metadata.get("Table Reference")
        if doc.metadata.get("Category") == "Table":
            table_groups[table_ref].append(doc.page_content)

    for table_ref, table_texts in table_groups.items():
        combined_text = "\n".join(table_texts)
        df = parse_table(combined_text)
        if not df.empty:
            results[table_ref] = process_df(df, table_name=table_ref, sql_engine=sql_engine)

    return results

def extract_table_html(filename: str, documents: List[Document]) -> str:
    """
    Extracts HTML representation of tables from document metadata.
    
    Args:
        filename (str): The name of the file.
        documents (List[Document]): List of document objects containing metadata and content.
    
    Returns:
        str: HTML representation of the extracted tables.
    """
    excel_html_document = ""
    
    for i, doc in enumerate(documents):
        try:
            doc_html = doc.metadata.get("text_as_html", "")
            if doc_html:
                excel_html_document += f"<h2>\nFile Name: {filename}; Sheet: {doc.metadata['page_name']}; Page No: {doc.metadata['page_number']}; Category: {doc.metadata['category']}\n</h2>\n" + f"{doc_html}" + "<hr>\n"
            else:
                excel_html_document += f"<h2>\nFile Name: {filename}; Sheet: {doc.metadata['page_name']}; Page No: {doc.metadata['page_number']}; Category: {doc.metadata['category']}\n</h2>\n" + f"<table>\n<tr><td>{doc.page_content}</td></tr>\n</table>" + "<hr>\n"
        except Exception as e:
            logger.debug(f"Error processing Doc {i}: {str(e)}")
            pass

    return excel_html_document

def chunk_html_tables(html_content: str) -> List[Document]:
    """
    Chunks HTML content into structured table elements and extracts metadata.
    
    Args:
        html_content (str): HTML string containing table elements.
    
    Returns:
        List[Document]: List of structured document objects containing parsed table data.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all the table elements in HTML
    tables_html = soup.find_all("table")

    h2_tags = soup.find_all("h2")

    structured_documents = []

    for table_idx, table in enumerate(tables_html):
        soup = BeautifulSoup(str(table), "html.parser")
        
        # Extract excel sheet metadata
        pattern = r"File Name:\s*(.?); Sheet:\s(.?); Page No:\s(\d+); Category:\s*([^\n<]*)"
        match = re.search(pattern, str(h2_tags[table_idx]).strip())
        if match:
            filename = match.group(1)
            sheet_name = match.group(2)
            page_no = match.group(3)
            category = match.group(4)

            # Encode the whole table
            header_rows = []
            encoded_rows = []
        
            # If only contain one line of text, treat it as one text sentence
            if len(soup.find_all("tr")) == 1:
                tr = soup.find("tr")
                encoded_rows = [td.get_text(strip=True) or "N/A" for tr in soup.find_all("tr") for td in tr.find_all("td")]
                structured_documents.append(Document(page_content='\n'.join(r for r in encoded_rows), 
                                                metadata={"Sheet": sheet_name, "Page": page_no, "Table Reference": f"{sheet_name}_{table_idx + 1}", "Category": category}))
            else:
                for idx, tr in enumerate(soup.find_all("tr")):
                    if idx == 0:
                        header_rows = "| row-id | " + (" | ").join([td.get_text(strip=True) or "N/A" for td in tr.find_all("td")])
                    else:
                        row_data = f"| row-{idx} | " + (" | ").join([td.get_text(strip=True) or "N/A" for td in tr.find_all("td")])
                        encoded_rows.append(header_rows)
                        encoded_rows.append(row_data)
                        structured_documents.append(Document(page_content='\n'.join(r for r in encoded_rows), 
                                                metadata={"Sheet": sheet_name, "Page": page_no, "Table Reference": f"{sheet_name}_{table_idx}", "Category": category}))
                        encoded_rows = []

    return structured_documents

def data_ingestion(data, filename: str):
    """
    Reads CSV (Pandas Read CSV)/Excel data (UnstructuredExcelLoader) and processes it into SQL.

    Args:
        data: The file object.
        filename (str): Name of the file.
        selected_sheets (list): List of selected sheets to process.

    Returns:
        Tuple:
            - dict: Mapping of sheet names to processed DataFrames and SQL metadata.
            - List[Document]: List of structured documents extracted from the Excel file.
    """

    file_name = filename.split(".")[0].strip()
    file_name = file_name.replace(" ", "_").strip()
    file_name = re.sub(r"[^A-Za-z0-9_-]", '', file_name)
    
    file_extension = filename.split(".")[-1].strip()
    results = {}
    structured_excel_documents = []

    db_url = f"sqlite:///data/{file_name}.db"
    sql_engine = create_engine(db_url)

    if file_extension == "csv":
        df = pd.read_csv(data, encoding='utf-8-sig')
        columns_to_remove = [
            'is_amp_top_stories', 'is_amp_blue_link', 'is_job_listing', 'is_job_details',
            'is_tpf_qa', 'is_tpf_faq', 'is_tpf_howto', 'is_weblite', 'is_action',
            'is_events_listing', 'is_events_details', 'is_search_appearance_android_app',
            'is_amp_story', 'is_amp_image_result', 'is_video', 'is_organic_shopping',
            'is_review_snippet', 'is_special_announcement', 'is_recipe_feature',
            'is_recipe_rich_snippet', 'is_subscribed_content', 'is_page_experience',
            'is_practice_problems', 'is_math_solvers', 'is_translated_result',
            'is_edu_q_and_a', 'is_product_snippets', 'is_merchant_listings',
            'is_learning_videos'
        ]

        df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

        df_cleaned, schema = process_df(df, file_name, sql_engine)
        results[file_name] = (df_cleaned, schema)
            
        # Extract header row
        headers = list(df.columns)
        header_text = "| " + " | ".join(headers)

        for _, row in df.iterrows():
            row_text = "| " + " | ".join(map(str,row.values))
            full_text = header_text + "\n" + row_text
            doc = Document(page_content = full_text, 
                        metadata = {"Sheet": file_name, "Page": "Page 1", "Table Reference": f"{file_name}", "Category": "Table"})
            
            structured_excel_documents.append(doc)

    elif file_extension == "xlsx":
        try:
            temp_file_path = os.path.expanduser(f"~/flexpg-gen-bi/data/{data.name}")
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

            with open(temp_file_path, "wb") as f:
                f.write(data.getbuffer())

            # Load the selected sheet using UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(temp_file_path, mode='elements')
            documents = loader.load() 
            excel_html_document = extract_table_html(file_name, documents)
            structured_excel_documents = chunk_html_tables(excel_html_document)
            results = group_table_elements(structured_excel_documents, sql_engine)

        except Exception as e:
            logger.error("Error in Processing File Format: ", e)
    else:
        raise ValueError("Unsupported file format. Only 'csv' and 'xlsx' are allowed.")
    
    sql_inspector = inspect(sql_engine)
    sql_database = SQLDatabase(sql_engine)
    return results, sql_database, sql_inspector, structured_excel_documents 

def data_ingestion_window(driver: str, server: str, port: str, database: str) -> SQLDatabase:
    """
    Establishes a connection to a Microsoft SQL Server database and returns an SQLDatabase instance.

    Args:
        driver (str): ODBC driver name for SQL Server.
        server (str): Hostname or IP address of the SQL Server.
        port (str): Port number for the SQL Server.
        database (str): Name of the database to connect to.

    Returns:
        SQLDatabase: An instance of SQLDatabase connected to the specified SQL Server database.
    """
    db_uri = ("mssql+pyodbc:///?odbc_connect=DRIVER={" + driver + "}" + f";SERVER={server},{port};DATABASE={database};readonly=True;Trusted_Connection=yes;")
    return SQLDatabase.from_uri(db_uri, lazy_table_reflection=True)

def data_ingestion_mysql(server: str, username: str, password: str, port:str, database: str) -> SQLDatabase:
    """
    Establishes a connection to a MySQL database and returns an SQLDatabase instance.

    Args:
        server (str): Hostname or IP address of the MySQL server.
        username (str): Username for authentication.
        password (str): Password for authentication.
        port (str): Port number for the MySQL server.
        database (str): Name of the database to connect to.

    Returns:
        SQLDatabase: An instance of SQLDatabase connected to the specified MySQL database.
    """
    db_uri = (
            f"mysql+mysqlconnector://{username}:{password}"
            f"@{server}:{port}"
            f"/{database}"
        )
    return SQLDatabase.from_uri(db_uri, lazy_table_reflection=True)

def data_ingestion_big_query(service_account_file: str, project_id: str, dataset_id: str, table_names: list = ["search_analytics_sgmytaxi"]) -> Tuple[SQLDatabase, object]:
    """
    Establishes a connection to a BigQuery database and returns an SQLDatabase instance and SQL inspector.

    Args:
        service_account_file (str): Path to the Google Cloud service account JSON file.
        project_id (str): Google Cloud project ID.
        dataset_id (str): BigQuery dataset ID.
        table_names (list): List of table names to include in the SQLDatabase instance.

    Returns:
        Tuple[SQLDatabase, object]: An instance of SQLDatabase connected to the specified BigQuery dataset and the SQL inspector.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
    sqlalchemy_url = f"bigquery://{project_id}/{dataset_id}"
    sql_engine = create_engine(sqlalchemy_url)
    sql_inspector = inspect(sql_engine)
    return SQLDatabase(sql_engine, include_tables=table_names), sql_inspector