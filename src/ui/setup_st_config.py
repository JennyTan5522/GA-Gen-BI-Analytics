import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from src.llm.chat_model import get_chat_model
from src.llm.embedding_model import get_embedding_model
from src.ui.data_connection import excel_connection, database_connection
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
        logger.debug("Session state configured with defaults.")

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
                            excel_connection()
                            logger.info("Excel data connection established.")
                        except Exception as e:
                            logger.error(f"Error connecting to Excel data: {e}")
                            st.error(f"Error connecting to Excel data: {e}")

                elif connection_type == "Connect to Database":
                    try:
                        database_connection()
                        logger.info("Database connection established.")
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

            # Only generate dataset overview if both data and LLM are connected
            st.session_state.is_data_and_llm_connected = self.is_data_and_llm_connected()
            if connection_type == "Upload CSV/Excel" and st.session_state.is_data_and_llm_connected:
                try:
                    self.generate_dataset_overview()
                    logger.info("Dataset overview generated.")
                except Exception as e:
                    logger.error(f"Error generating dataset overview: {e}")
                    st.error(f"Error generating dataset overview: {e}")

            st.sidebar.write("🛠 Memory Settings")
            st.session_state.k = st.slider("Memory Size", 1, 10, st.session_state.k)

            if st.button("🗑 Clear Message History"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                if "memory" in st.session_state:
                    st.session_state.memory.clear()
                logger.info("Message history cleared.")

    def is_data_and_llm_connected(self) -> bool:
        """
        Check if both a data source and LLM are connected.
        Returns True if both are connected, otherwise shows a warning and returns False.
        """
        if "data" not in st.session_state and "db" not in st.session_state:
            st.warning(WARNING_MESSAGE)
            logger.warning("Data source not connected.")
            return False
        if "llm" not in st.session_state:
            st.warning("Please enter the API key to start conversation")
            logger.warning("LLM not connected.")
            return False
        return True