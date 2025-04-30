from UI_Manager import UIManager
from Data_Manager import DataManager
from LLM_Manager import LLMManager
from Logger_Manager import LoggerManager
import streamlit as st

class GenBIApp:
    def __init__(self):
        self.logger = LoggerManager.configure_logger()
        self.logger.info("🚀 Starting GenBIApp...")

        # self.llm_manager = LLMManager()
        # self.llm = self.llm_manager.initialize_claude_model()
     
        self.ui_manager = UIManager(self)
        self.data_manager = DataManager(self)

    def run(self):
        self.ui_manager.configure_streamlit()
        self.ui_manager.configure_session_state()
        self.ui_manager.configure_sidebar()
        self.ui_manager.display_tabs()

app = GenBIApp()
app.run()