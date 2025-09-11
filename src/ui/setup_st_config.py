import streamlit as st
import asyncio
from langchain.memory import ConversationBufferWindowMemory
from src.llm.chat_model import get_chat_model
from src.llm.embedding_model import get_embedding_model
from src.ui.data_connection import excel_connection, database_connection
from src.tools.dataset_summary_tool import dataset_summary_async
from src.tools.question_recommendation_tool import generate_question_recommendations_async
from data.const import WARNING_MESSAGE
from config.logger import get_logger

logger = get_logger(__name__)

class StreamlitConfig:
    """Set Up Streamlit Configuration"""

    def __init__(self):
        """Initialize StreamlitConfig and configure UI."""
        self.configure_streamlit()
        self.configure_session_state()
        self.configure_sidebar()

    def configure_streamlit(self):
        """Configure Streamlit page settings."""
        st.set_page_config(page_title="GenBI", page_icon="📊", layout="wide")

    def configure_session_state(self):
        """Initialize default values in Streamlit session state if not already set."""
        defaults = {
            "k": 5,
            "memory": ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True),
            "tables_info": {},
            "embedding_function": get_embedding_model(embedding_model_name='sbert'),
            "feedback": False,
            "schema": None,
            "selected_dataset_id": "",
            "sql_query_documents": []
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
       
    def generate_dataset_overview(self):
        """
        Generates a summary and recommended questions for the uploaded dataset.
        """
        num_rows = min(len(st.session_state.df), 11)
        sample_dataset = st.session_state.df[:num_rows]
        sample_dataset_str = sample_dataset.to_csv(sep="|", index=False, lineterminator="\n")

        async def main(llm, sample_dataset_str, rows, cols, schema):
            logger.debug("Starting concurrent execution of summary and question recommendations...")
            excel_summary, question_recommendations = await asyncio.gather(
                dataset_summary_async(llm, sample_dataset_str, rows, cols, schema),
                generate_question_recommendations_async(llm, sample_dataset_str)
            )
            logger.debug("Concurrent execution completed.")
            return excel_summary, question_recommendations

        excel_summary, question_recommendations = asyncio.run(
            main(
                st.session_state.llm,
                sample_dataset_str,
                st.session_state.df.shape[0],
                st.session_state.df.shape[1],
                st.session_state.schema
            )
        )
        logger.debug(f"Excel Summary: {excel_summary}")
        logger.debug(f"Question Recommendation: {question_recommendations}")

    def clear_history(self):
        """Clear message history and memory."""
        if "messages" in st.session_state:
            st.session_state.messages = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        logger.info("Message history cleared.")

    def configure_sidebar(self):
        """Configure the Streamlit sidebar for data connection, API key, memory, and clearing history."""
        with st.sidebar:
            st.markdown("## ⚙ Settings")
            st.write("Configure your settings below.")

            with st.expander("Connect to Data", expanded=True):
                connection_type = st.selectbox("Choose connection type", ("Upload CSV/Excel", "Connect to Database"))

                if connection_type == "Upload CSV/Excel":
                    data = st.file_uploader("\U0001F4BB Load an Excel file:", type=["csv", "xlsx"])
                    if data:
                        st.session_state.data = data
                        try:
                            is_connected = excel_connection()
                            if is_connected:
                                logger.info("Excel data connection established.")
                                self.clear_history()
                        except Exception as e:
                            logger.error(f"Error connecting to Excel data: {e}")
                            st.error(f"Error connecting to Excel data: {e}")

                elif connection_type == "Connect to Database":
                    try:
                        is_connected = database_connection()
                        if is_connected:
                            logger.info("Database connection established.")
                            self.clear_history()
                    except Exception as e:
                        logger.error(f"Error connecting to database: {e}")
                        st.error(f"Error connecting to database: {e}")

            st.markdown("---")

            try:
                st.title("API Access")
                llm_api_key = st.text_input("Enter your Claude API Key:", type="password")

                if st.button("Connect to Claude API"):
                    if llm_api_key:
                        try:
                            chat_model = get_chat_model(
                                chat_model_name="claude",
                                api_key=llm_api_key,
                                model_name="claude-3-5-sonnet-20241022",
                                temperature=0,
                                max_tokens=8000
                            )
                            st.session_state.llm = chat_model
                            st.success("✅ Successfully connected to Claude API.")
                            logger.info("Successfully connected to Claude API.")
                        except Exception as e:
                            logger.error(f"Failed to connect to Claude API: {e}")
                            st.error(f"❌ Failed to connect to Claude API: {e}")
                    else:
                        st.warning("⚠️ Please enter your Claude API key before connecting.")

            except Exception as e:
                logger.error(f"Error while connecting to Claude Model: {e}")
                st.error(f"Error while connecting to Claude Model: {e}")

            st.markdown("---")
            st.sidebar.write("🛠 Memory Settings")
            st.session_state.k = st.slider("Memory Size", 1, 10, st.session_state.k)

            if st.button("🗑 Clear Message History"):
                self.clear_history()

def is_data_and_llm_connected() -> bool:
    """
    Check if both a data source and LLM are connected.
    Returns True if both are connected, otherwise shows a warning and returns False.
    """
    if "data" not in st.session_state and "db" not in st.session_state:
        return False
    if "llm" not in st.session_state:
        return False
    return True