import streamlit as st
from utils.ingestion import data_ingestion, data_ingestion_window, data_ingestion_mysql, load_data_into_vector_db
from langchain.retrievers import BM25Retriever

class DataManager:
    def __init__(self, app):
        self.app = app

    def excel_data_connection(self):
        """
        Establishes a connection with the uploaded Excel or CSV data and processes it accordingly.
        """
        try:
            results, sql_database, sql_inspector, structured_excel_elements = data_ingestion(data=st.session_state.data, filename=st.session_state.data.name)
            st.session_state.sql_inspector = sql_inspector
            st.session_state.db = sql_database
         
            if st.session_state.data.name.endswith(".csv"): 
                table_name, (df_cleaned, schema) = next(iter(results.items()))
                st.session_state.schema = schema
                st.session_state.df = df_cleaned
                st.session_state.file_name = st.session_state.data.name.split(".")[0]
                self.app.logger.debug(f"Filename: ", st.session_state.file_name)

            if st.session_state.data.name.endswith(".xlsx"): 
                self.app.logger.info("Load Excel Data into Vector DB...")
                st.session_state.vector_store = load_data_into_vector_db(excel_structured_documents=structured_excel_elements, collection_name="excel_data")
                st.session_state.bm25_retriever = BM25Retriever.from_documents(structured_excel_elements)
                st.session_state.bm25_retriever.k = 3
                self.app.logger.info("Completed loading Excel Data into Vector DB.")
               
        except Exception as e:
            st.error(f"Error loading the file: {e}")

    def handle_database_connection(self):
        """
        Establishes a connection with the Database.
        """

        st.subheader("Database Connection")

        auth_type = st.selectbox(
            "Choose Authentication Type",
            ("Windows Authentication", "SQL Server Authentication"),
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
                    except Exception as e:
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
                    except Exception as e:
                        st.error(f"Error connecting to the database: {e}")