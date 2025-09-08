import streamlit as st
from src.ui.response_tab import ResponseTab
from src.ui.data_tab import DataTab
from src.ui.document_tab import DocumentTab
from src.ui.setup_st_config import StreamlitConfig
from config.service_config import ServiceConfig
from config.logger import setup_logger

def main():
    """Run Gen BI Streamlit app."""
    setup_logging()
    configure_streamlit()
    display_main_tabs()

def setup_logging():
    """Configure application logging."""
    config = ServiceConfig()
    setup_logger(config.LOG_LEVEL, config.LOG_FILE)

def configure_streamlit():
    """Set up Streamlit configuration."""
    StreamlitConfig()

def display_main_tabs():
    """Display main UI tabs for response, data, and document explorer."""
    response_tab, data_tab, document_tab = st.tabs(["💬 Response View", "📜 Data Explorer", "📄 Document Explorer"])
    with response_tab:
        ResponseTab()
    with data_tab:
        DataTab()
    with document_tab:
        DocumentTab()

if __name__ == "__main__":
    main()