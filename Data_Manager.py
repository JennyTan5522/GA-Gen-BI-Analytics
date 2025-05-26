import streamlit as st
from utils.ingestion import data_ingestion, data_ingestion_window, data_ingestion_mysql #, load_data_into_vector_db
from langchain.retrievers import BM25Retriever
from tools.dataset_summary_tool import dataset_summary_async
from tools.question_recommendation_tool import generate_question_recommendations_async
import asyncio
import re

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
                self.schema = schema
                self.df = df_cleaned
                st.session_state.df = df_cleaned
                st.session_state.file_name = re.sub(r'[^a-zA-Z0-9_-]', '_', st.session_state.data.name.split(".")[0])
                self.app.logger.debug(f"Filename: ", st.session_state.file_name)
                
                # # Generate a sumamry & question recommendations for CSV
                # num_rows = min(len(self.df), 11)
                # sample_dataset = self.df[:num_rows]
                # sample_dataset_str = sample_dataset.to_csv(sep="|", index=False, lineterminator="\n")

                # # Main function to run tasks concurrently
                # async def main(llm, sample_dataset_str, rows, cols, schema):
                #     # Run tasks concurrently using asyncio.gather
                #     self.app.logger.debug("Starting concurrent execution of summary and question recommendations...")
                #     excel_summary, question_recommendations = await asyncio.gather(
                #         dataset_summary_async(llm, sample_dataset_str, rows, cols, schema),
                #         generate_question_recommendations_async(llm, sample_dataset_str)
                #     )
                #     self.app.logger.debug("Concurrent execution completed.")
                #     return excel_summary, question_recommendations

                # # Run the async main function
                # excel_summary, question_recommendations = asyncio.run(
                #     main(self.app.llm, sample_dataset_str, self.df.shape[0], self.df.shape[1], self.schema)
                # )

                # print("Question Recommendation")
                # print(question_recommendations)

                # self.app.logger.debug("Excel Summary:\n",excel_summary)
                # self.app.logger.debug("Question Recommendation: \n",question_recommendations)

                # for key, value in {"excel_summary": excel_summary, "question_recommendations": question_recommendations}.items():
                #     if value:
                #         st.session_state[key] = value

            # if st.session_state.data.name.endswith(".xlsx"): 
            #     st.session_state.file_name = st.session_state.data.name
            #     self.app.logger.info("Load Excel Data into Vector DB...")
            #     self.app.logger.info(structured_excel_elements)
            #     st.session_state.vector_store = load_data_into_vector_db(excel_structured_documents=structured_excel_elements, collection_name="excel_data")
            #     st.session_state.bm25_retriever = BM25Retriever.from_documents(structured_excel_elements)
            #     st.session_state.bm25_retriever.k = 3
            #     self.app.logger.info("Completed loading Excel Data into Vector DB.")
               
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