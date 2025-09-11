import streamlit as st
import pandas as pd
import json
from uuid import uuid4
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from src.llm.embedding_model import get_embedding_model
from src.ui.setup_st_config import is_data_and_llm_connected
from config.service_config import ServiceConfig
from config.logger import get_logger

logger = get_logger(__name__)

def add_documents_to_vector_store(sql_query_documents: list, uuids: list, sql_query_input: dict):
    """
    Adds SQL query documents to the vector store.

    Args:
        sql_query_documents (list): List of Document objects to add to the vector store.
        uuids (list): List of unique identifiers for the documents.
        sql_query_input (dict): Dictionary containing the user query and SQL query to be saved.
    """
    st.session_state.vector_store.add_documents(documents=sql_query_documents, ids=uuids)
    st.session_state.sql_query_documents.append(sql_query_input)
    logger.info(f"Add Documents into Vector Store: \n{sql_query_input}")
    st.success("Document saved successfully! Document has been added and stored in the vector store for future reference.")
class DocumentTab:
    def __init__(self):
        """Initialize DocumentTab and handle the document tab UI."""
        if not is_data_and_llm_connected():
            return

        self.handle_document_tab()
    
    def create_qdrant_collection_if_not_exists(self, client: QdrantClient, collection_name: str):
        """Create a Qdrant collection with dense and sparse vector configs if it does not exist."""
        try:
            collections = [c.name for c in client.get_collections().collections]
            logger.debug(f"Existing Qdrant collections: {collections}")
            if collection_name not in collections:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={"dense_embedding": VectorParams(size=384, distance=Distance.COSINE)},
                    sparse_vectors_config={"sparse_embedding": SparseVectorParams(index=models.models.SparseIndexParams(on_disk=False))}
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.debug(f"Qdrant collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Error creating Qdrant collection '{collection_name}': {e}")
            raise
        
    def handle_document_tab(self):
        """Display and manage SQL query documents in the Document Tab."""
        service_config = ServiceConfig()
        if "vector_store" not in st.session_state:
            st.markdown("Initializing the vector store for SQL Query Documents...")
            try:
                qdrant_client = QdrantClient(
                    url=service_config.QDRANT_HOST,
                    api_key=service_config.get_qdrant_api_key(),
                    prefer_grpc=True
                )
                collection_name = "rank_solution_gen_bi_collection"
                self.create_qdrant_collection_if_not_exists(qdrant_client, collection_name)

                st.session_state.vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=collection_name,
                    embedding=get_embedding_model(embedding_model_name='sbert'),
                    sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
                    retrieval_mode=RetrievalMode.HYBRID,
                    vector_name="dense_embedding",
                    sparse_vector_name="sparse_embedding",
                )
                st.session_state.retriever = st.session_state.vector_store.as_retriever(
                    search_type="mmr", search_kwargs={"k": 5}
                )
                retrieved_docs = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
                st.session_state.sql_query_documents = retrieved_docs[0]
                logger.info("Vector store initialized and documents retrieved.")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                st.error(f"Failed to initialize vector store: {e}")
                return

        st.write("### 📄 SQL Query Documents")

        # Add new document form
        with st.form("add_sql_query_document"):
            user_query_input: str = st.text_area("User Query", placeholder="Enter your question here")
            sql_query_input: str = st.text_area("SQL Query", placeholder="Enter the SQL query here")
            confirm_save: bool = st.checkbox("I confirm I want to save this document.")
            submitted: bool = st.form_submit_button("Add Document")

            if submitted:
                if not (user_query_input and sql_query_input):
                    st.warning("Please fill in both fields.")
                elif not confirm_save:
                    st.warning("Please confirm you want to save this document.")
                else:
                    try:
                        page_content = {"User Query": user_query_input, "SQL Query": sql_query_input}
                        new_doc = [Document(page_content=json.dumps(page_content), metadata={"source": st.session_state.file_name})]
                        uuids = [str(uuid4())]
                        add_documents_to_vector_store(new_doc, uuids, page_content)
                        logger.info("New document added to vector store.")
                    except Exception as e:
                        logger.error(f"Error adding document: {e}")
                        st.error(f"Error adding document: {e}")

        st.divider()
        st.write("#### 🔍 Search SQL Query Documents by Question")
        search_query: str = st.text_area("Enter a question to search for similar SQL queries", key="search_sql_doc")
        if st.button("Search Similar Documents"):
            if search_query.strip():
                try:
                    similar_docs = st.session_state.retriever.invoke(search_query.strip())
                    if similar_docs:
                        st.success(f"Top {len(similar_docs)} similar results:")
                        doc_data = []
                        for doc in similar_docs:
                            try:
                                page_content = json.loads(doc.payload['page_content'])
                                question = page_content.get("User Query", "").strip()
                                answer = page_content.get("SQL Query", "").strip()
                                doc_data.append({
                                    "User Query": question,
                                    "SQL Query": answer,
                                    "Database": doc.payload['metadata']['source']
                                })
                            except Exception as e:
                                logger.error(f"Error processing document: {e}")
                        df = pd.DataFrame(doc_data)
                        df.index = range(1, len(df) + 1)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No similar documents found.")
                except Exception as e:
                    logger.error(f"Error searching similar documents: {e}")
                    st.error(f"Error searching similar documents: {e}")
            else:
                st.warning("Please enter a question to search.")

        logger.debug(f"SQL Query Documents: {st.session_state.sql_query_documents}")
        if not st.session_state.sql_query_documents:
            st.info("No documents found in the vector store.")
            return

        # Display the SQL Documents
        doc_data = []
        for doc in st.session_state.sql_query_documents:
            try:
                page_content = json.loads(doc.payload['page_content'])
                doc_data.append({
                    "User Query": page_content['User Query'],
                    "SQL Query": page_content['SQL Query'],
                    "Table Name": doc.payload['metadata']['table_name']
                })
            except Exception as e:
                logger.error(f"Error processing document: {e}")

        df = pd.DataFrame(doc_data)
        df.index = range(1, len(df) + 1)
        st.write("### All SQL Query Documents")
        st.dataframe(df, use_container_width=True)

        # Clear all documents in the vector store
        if st.button("Clear All Documents"):
            confirm_clear = st.checkbox("I confirm I want to clear all documents.", key="confirm_clear")
            if confirm_clear:
                try:
                    client = st.session_state.vector_store.client
                    client.delete_collection(collection_name="rank_solution_gen_bi_collection")
                    st.session_state.sql_query_documents = []
                    st.success("Cleared all documents from the vector store.")
                    logger.info("All documents cleared from the vector store.")
                except Exception as e:
                    logger.error(f"Error clearing documents: {e}")
                    st.error(f"Error clearing documents: {e}")
            else:
                st.warning("Please confirm you want to clear all documents.")

        st.divider()
        st.write("#### 📊 View Document Embeddings")
        logger.info(f"Document embeddings: {[doc for doc in st.session_state.sql_query_documents]}")
        if st.session_state.sql_query_documents:
            doc_to_view = st.selectbox(
                "Select a document to view its embeddings",
                options=[doc.payload['metadata']['source'] for doc in st.session_state.sql_query_documents],
                index=0
            )
            if doc_to_view:
                try:
                    selected_doc = next(doc for doc in st.session_state.sql_query_documents if doc.payload['metadata']['source'] == doc_to_view)
                    embeddings = selected_doc.vector
                    st.write(f"**Embeddings for Document:** {doc_to_view}")
                    st.json(embeddings.tolist())
                except Exception as e:
                    logger.error(f"Error retrieving embeddings: {e}")
                    st.error(f"Error retrieving embeddings: {e}")
        else:
            st.info("No documents available to show embeddings.")

        st.divider()
        st.write("#### 🔍 Advanced Search SQL Query Documents")
        with st.form("advanced_search_form"):
            search_question: str = st.text_input("Search Question", placeholder="Enter question to search")
            min_date: str = st.date_input("Min Date", value=pd.to_datetime("2023-01-01"), max_value=pd.to_datetime("today")).strftime("%Y-%m-%d")
            max_date: str = st.date_input("Max Date", value=pd.to_datetime("today"), min_value=pd.to_datetime("2023-01-01")).strftime("%Y-%m-%d")
            database_filter: str = st.text_input("Database Filter", placeholder="Enter database name to filter")
            apply_filters: bool = st.form_submit_button("Apply Filters")

            if apply_filters:
                try:
                    filtered_docs = st.session_state.sql_query_documents
                    if search_question:
                        filtered_docs = [doc for doc in filtered_docs if json.loads(doc.payload['page_content']).get("User Query", "").lower().startswith(search_question.lower())]
                    if database_filter:
                        filtered_docs = [doc for doc in filtered_docs if doc.payload['metadata']['source'] == database_filter]
                    st.write(f"### Filtered SQL Query Documents ({len(filtered_docs)})")
                    if filtered_docs:
                        doc_data = []
                        for doc in filtered_docs:
                            try:
                                page_content = json.loads(doc.payload['page_content'])
                                doc_data.append({
                                    "User Query": page_content['User Query'],
                                    "SQL Query": page_content['SQL Query'],
                                    "Table Name": doc.payload['metadata']['table_name']
                                })
                            except Exception as e:
                                logger.error(f"Error processing document: {e}")

                        df = pd.DataFrame(doc_data)
                        df.index = range(1, len(df) + 1)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No documents found matching the filters.")
                except Exception as e:
                    logger.error(f"Error applying filters: {e}")
                    st.error(f"Error applying filters: {e}")