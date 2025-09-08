import streamlit as st
import os
from src.utils.ingestion import data_ingestion, data_ingestion_window, data_ingestion_mysql, data_ingestion_big_query, load_data_into_vector_db
from langchain.retrievers import BM25Retriever
from google.cloud import bigquery
from config.service_config import ServiceConfig
from config.logger import get_logger

logger = get_logger(__name__)

def excel_connection() :
    """Establish connection with uploaded Excel or CSV data and process it for use in the app."""
    try:
        results, sql_database, sql_inspector, structured_excel_elements = data_ingestion(
            data=st.session_state.data, filename=st.session_state.data.name
        )
        st.session_state.sql_inspector = sql_inspector
        st.session_state.db = sql_database

        if st.session_state.data.name.endswith(".csv"):
            table_name, (df_cleaned, schema) = next(iter(results.items()))
            st.session_state.schema = schema
            st.session_state.df = df_cleaned

        if st.session_state.data.name.endswith(".xlsx"):
            logger.info("Load Excel Data into Vector DB...")
            st.session_state.vector_store = load_data_into_vector_db(
                excel_structured_documents=structured_excel_elements, collection_name="excel_data"
            )
            st.session_state.bm25_retriever = BM25Retriever.from_documents(structured_excel_elements)
            st.session_state.bm25_retriever.k = 3
            logger.info("Completed loading Excel Data into Vector DB.")

        st.session_state.file_name = st.session_state.data.name.split(".")[0]
        logger.debug(f"Filename: {st.session_state.file_name}")
        st.session_state.table_names = st.session_state.sql_inspector.get_table_names()

    except Exception as e:
        logger.error(f"Error loading the file: {e}")
        st.error(f"Error loading the file: {e}")

def database_connection():
    """Establish a database connection using BigQuery, SQL Server, or Windows Authentication."""
    st.subheader("Database Connection")

    config = ServiceConfig()

    auth_type = st.selectbox(
        "Choose Authentication Type",
        ("BigQuery", "Windows Authentication", "SQL Server Authentication"),
    )

    if auth_type == "Windows Authentication":
        driver = st.text_input("Driver")
        server = st.text_input("Server")
        port = st.text_input("Port")
        database = st.text_input("Database")

        if st.button("Connect"):
            with st.spinner("Connecting to database..."):
                try:
                    st.session_state.db = data_ingestion_window(driver, server, port, database)
                    st.session_state.db_name = database
                    st.success("Connected to database!")
                    logger.info("Connected to database using Windows Authentication.")
                except Exception as e:
                    logger.error(f"Error connecting to the database: {e}")
                    st.error(f"Error connecting to the database: {e}")

    elif auth_type == "SQL Server Authentication":
        server = st.text_input("Server")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        port = st.text_input("Port")
        database = st.text_input("Database")

        if st.button("Connect"):
            with st.spinner("Connecting to database..."):
                try:
                    st.session_state.db = data_ingestion_mysql(server, username, password, port, database)
                    st.session_state.db_name = database
                    st.success("Connected to database!")
                    logger.info("Connected to database using SQL Server Authentication.")
                except Exception as e:
                    logger.error(f"Error connecting to the database: {e}")
                    st.error(f"Error connecting to the database: {e}")

    elif auth_type == "BigQuery":
        service_account_json_path = config.GOOGLE_SERVICE_ACCOUNT_FILE
        if not service_account_json_path:
            logger.error("GOOGLE_SERVICE_ACCOUNT_FILE environment variable not set.")
            st.error("Please set the GOOGLE_SERVICE_ACCOUNT_FILE environment variable.")
            return
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json_path

        try:
            client = bigquery.Client()
            datasets = list(client.list_datasets())
            if not datasets:
                logger.warning("No datasets found in the project.")
                st.error("No datasets found in the project.")
                return

            dataset_ids = [dataset.dataset_id for dataset in datasets]
            selected_dataset_id = st.selectbox("Select Datasets", dataset_ids)
            st.session_state.selected_dataset_id = selected_dataset_id

            table_names = st.multiselect(
                "Select Tables",
                options=[table.table_id for table in client.list_tables(selected_dataset_id)],
                default=[],
            )
            if not table_names:
                st.warning("Please select at least one table.")
                return

            st.session_state.table_names = table_names

            if st.button("Connect"):
                with st.spinner("Connecting to BigQuery..."):
                    try:
                        st.session_state.db, st.session_state.sql_inspector = data_ingestion_big_query(
                            service_account_json_path,
                            project_id=client.project,
                            dataset_id=selected_dataset_id,
                            table_names=table_names,
                        )
                        st.session_state.db_name = "BigQuery"
                        st.success("Connected to BigQuery!")
                        logger.info("Connected to BigQuery.")
                    except Exception as e:
                        logger.error(f"Error connecting to BigQuery: {e}")
                        st.error(f"Error connecting to BigQuery: {e}")
        except Exception as e:
            logger.error(f"Error initializing BigQuery client: {e}")
            st.error(f"Error initializing BigQuery client: {e}")