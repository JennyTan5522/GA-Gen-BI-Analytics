import json
from collections import defaultdict
import pandas as pd
import streamlit as st
from langchain_anthropic import ChatAnthropic

# Constants
from const import (
    WARNING_MESSAGE,
    PROMPT_TEMPLATE,
    GENERATE_PLAN_TEMPLATE
)

# LangChain Core Modules
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.document import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import create_react_agent, AgentExecutor

# Custom Toolkits for ReactAgent
from utils.agent_response_parser import CustomResponseParser
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from tools.follow_up_question_tool import FollowUpQuestionTool
from tools.output_validator_tool import FinalAnswerValidatorTool

# Utility Functions for Response Parser
from utils.agent_response_parser import ResponseSchema

class UIManager:
    def __init__(self, app):
        self.app = app
        self.custom_parser = CustomResponseParser()

    def configure_streamlit(self):
        st.set_page_config(page_title="GenBI", page_icon="📊", layout="wide")

    def configure_session_state(self):
        defaults = {
            "k": 5,
            "memory": ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=5,
                return_messages=True
            )
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

        data = None

    def configure_sidebar(self):
        with st.sidebar:
            st.markdown("## ⚙ Settings")
            st.write("Configure your settings below.")
            
            # Data Connection
            with st.expander("Connect to Data", expanded=True):
                connection_type = st.selectbox("Choose connection type", ("Upload CSV/Excel", "Connect to Database"))

                if connection_type == "Upload CSV/Excel":
                    data = st.file_uploader("\U0001F4BB Load an Excel file:", type=["csv", "xlsx"])
                    if data:
                        st.session_state.data = data
                        self.app.data_manager.excel_data_connection()

                elif connection_type == "Connect to Database":
                    self.app.data_manager.handle_database_connection()

            st.markdown("---")

            # Model Selection
            # model_options = {
            #     "Claude": st.session_state.llm_manager.initialize_claude_model
            # }
            # st.sidebar.write("🤖 Model Selection")
            # selected_model = st.selectbox("Model Available", model_options)
            # st.session_state.llm = model_options[selected_model]()

            # Agent Settings
            # st.sidebar.write("🛠 Agent Settings")
            # st.session_state.k = st.slider("Memory Size", 1, 10, st.session_state.k)

            # API Key 
            try:
                st.title("API Access")
                claude_api_key = st.text_input("Enter your API Key: ", type="password")
                if claude_api_key:
                    st.session_state.llm = ChatAnthropic(api_key=claude_api_key, model="claude-3-5-sonnet-20241022", temperature=0)
            except:
                st.error("Error while connecting to LLM Model.")

            # Clear History Button
            if st.button("🗑 Clear Message History"):
                if "messages" in st.session_state:
                    st.session_state.messages = []
                if "memory" in st.session_state:
                    st.session_state.memory.clear()  
                st.success("Message history cleared!")

    def react_agent_toolkit(self):
        """
        Creates and configures a ReAct agent toolkit with SQL, Python execution, and response validation capabilities.

        This agent is designed to interact with a database, generate plans, execute Python code, validate responses, and handle follow-up questions.

        Args:
            None (Relies on Streamlit session state and class attributes).

        Returns:
            AgentExecutor: A fully initialized agent executor capable of handling complex queries 
                        and reasoning through multiple tools.
        """
        db_toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=st.session_state.llm)
        tools = db_toolkit.get_tools()
        tools.extend([
            FollowUpQuestionTool(llm=st.session_state.llm),
            FinalAnswerValidatorTool(llm=st.session_state.llm)
        ])
        
        top_k = 10 

        prompt_template = PromptTemplate.from_template(
            PROMPT_TEMPLATE,
            partial_variables={"generate_plan_instructions": GENERATE_PLAN_TEMPLATE, "db": st.session_state.db, "top_k": top_k, "dialect": st.session_state.db.dialect},
        )

        react_agent = create_react_agent(
            llm=st.session_state.llm, 
            tools=tools, 
            prompt=prompt_template, 
        )

        agent_executor = AgentExecutor(
            agent=react_agent, 
            tools=tools, 
            verbose=True,
            memory=st.session_state.memory,
            handle_parsing_errors=True,
            max_iteration=50, 
            max_execution_time=300
        )

        return agent_executor
    
    def display_tabs(self):
        response_tab, data_tab = st.tabs(["💬 Response View", "📜 Data Explorer"])
        
        with response_tab:
            self.handle_response_tab()

        with data_tab:
            self.handle_data_tab()

    def handle_response_tab(self):
        if "data" not in st.session_state and "db" not in st.session_state:
            st.warning(WARNING_MESSAGE)
        elif "llm" not in st.session_state:
            st.warning("Please enter the API key to start conversation")
        else:
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm your SQL Assistant. How can I assist you today?"}]
                self.app.logger.debug("Session messages initialized.")
                st.session_state["selected_mode"] = "SQL to Chart 📊" 

            if "excel_summary" in st.session_state:
                summary_html = st.session_state.excel_summary.replace("\n", "<br>")

                with st.expander("📃 Data Summary"):
                    st.markdown(f"""<p style="font-size: 11px; color: gray;"> An enriched data summary with semantic types and descriptions.</p>""",unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div style="font-size: 11px; color: gray; text-align: justify; background-color: #f0f0f0; border-radius: 8px; padding: 10px; ">
                            {summary_html} 
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
            if "question_recommendations" in st.session_state:
                grouped_questions = {}
                # Organize questions by category
                for item in st.session_state.question_recommendations["questions"]:
                    category = item["category"]
                    question = item["question"]
                    
                    if category not in grouped_questions:
                        grouped_questions[category] = []
                    grouped_questions[category].append(question)

                with st.expander("💡 Goal Exploration"):
                    st.markdown(f"""<p style="font-size: 11px; color: gray;"> A list of automatically generated data exploration goals based on the dataset given.</p>""",unsafe_allow_html=True)
                    categories = list(grouped_questions.keys())  # Get category names
                    num_columns = 3  # Number of columns per row
                    columns = st.columns(num_columns)  # Create columns

                    # Iterate over categories and distribute them in columns
                    for index, category in enumerate(categories):
                        col = columns[index % num_columns]  # Assign to a column in a cyclic order
                        with col.container():
                            st.markdown(
                                f"""
                                <div style="border-radius: 8px; padding-left: 10px; background-color: #f9f9f9; 
                                            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); width: 100%;">
                                    <h4 style="color: #333; font-size: 13px; font-weight: bold; margin: 0;">{category}</h4>
                                </div>
                                <div style="padding-left: 15px; margin-top: 5px;">
                                    <ul style="padding-left: 15px; color: #333;">
                                        {''.join(f'<li style="margin-bottom: 4px; font-size: 11px; text-align: justify;">{question}</li>' for question in grouped_questions[category])}
                                    </ul>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

            # Display chat history
            for message in st.session_state.messages:
                role = message.get("role")
                if role == "user":
                    with st.chat_message("user"):
                        content = message["content"]
                        st.write(content)

                if role == "assistant":
                    with st.chat_message("assistant"):
                        content = message["content"]
                        try:
                            # Converts the str response to dict format
                            content_json = json.loads(content)
                            self.display_response(content_json['SQL'], content_json['TextResponse'], content_json['Code'])
                        except:
                            st.write(content)

            # Style to fix the chatbot position
            st.markdown(
                """
                <style>
                /* Chat input box */
                .stChatInput {
                    position: fixed;
                    bottom: 25px; 
                    left: 350px; 
                    width: calc(100% - 350px);
                    background-color: white;
                    padding: 10px;
                    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
                    z-index: 1000;
                    border-radius: 10px;
                }

                /* Disclaimer text */
                .disclaimer {
                    position: fixed;
                    bottom: 0px; 
                    margin-bottom: -5px;
                    left: 350px;
                    width: calc(100% - 350px);
                    font-size: 11px;
                    color: gray;
                    text-align: center;
                    background-color: white;
                    padding: 5px 10px;
                    z-index: 999; /* Lower z-index to stay below chat input */
                }

                /* Radio button styling */
                .stRadio > div {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                    margin-bottom: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <p class="disclaimer">
                    <strong> Disclaimer:</strong> Gen BI system may make mistakes; review results and use your judgment. 
                </p>
                """,
                unsafe_allow_html=True,
            )

            user_query = st.chat_input(placeholder="Ask me anything!")

            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.chat_message("user").write(user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("We are preparing a response to your question. Please allow up to one minute for completion...."):
                        try:
                            agent_executor = self.react_agent_toolkit()
                            response_text = agent_executor.invoke({"input": user_query})
                            final_answer_pydantic_format, final_answer_str_format = self.custom_parser.parse(response_text['output'])
                            sql = final_answer_pydantic_format.SQL
                            text_response = final_answer_pydantic_format.TextResponse
                            code_block = final_answer_pydantic_format.Code
                            self.app.logger.debug(f"AI response: {final_answer_str_format}")

                            self.display_response(sql, text_response, code_block)

                            # Convert final response dict to JSON string
                            st.session_state.messages.append({"role": "assistant", "content": final_answer_str_format})

                            # Save chat history to memory for LLM reference
                            st.session_state.memory.save_context({"input": user_query}, {"output": text_response})

                        except ValueError as e:
                            st.error(f"Parsing error: {e}")
                            self.app.logger.error(f"Parsing response: {e}")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            self.app.logger.error(f"Parsing response: {e}")

    def display_response(self, sql: str, response: str, code_block: str):
        """
        Displays the response from an SQL query or a generated result, optionally rendering a visualization.

        Args:
            sql (str): The SQL query used to generate the response. If empty, only the response is displayed.
            response (str): The textual response generated from the SQL query or retrieval process.
            code_block (str): A Python code block that generates a visualization (e.g., a chart). If empty, no visualization is displayed.
        """

        if response[0] == '"' and response[-1] == '"':
            response = response[1:-1]

        if sql.strip() == "":
            text_response = f"*Response:*\n{response}"
        else:
            text_response = f"*SQL Query:*\n\n{sql}\n\n*Response:*\n{response}"

        if code_block.strip() != "" :
            textual_column, vis_column = st.columns(2)
            with textual_column:
                st.markdown(text_response)

            with vis_column:
                chart_tab, code_tab = st.tabs(["📊 Chart", "</> Code"])
                with chart_tab:
                    exec(code_block)
                with code_tab:
                    formatted_code_block = code_block.replace(";", ";\n\n")
                    st.code(formatted_code_block, language="python")
        else:
            st.markdown(text_response)

    def handle_data_tab(self):
        if ("data" not in st.session_state) and ("db" not in st.session_state):
            st.warning(WARNING_MESSAGE)
        else:
            if "db" and "sql_inspector" in st.session_state:
                tables = st.session_state.sql_inspector.get_table_names()
                self.app.logger.info(f"Tables in DB: {tables}")
                selected_table = st.selectbox("Choose a table from the database for Data Preview:", tables)
                st.session_state.df = pd.read_sql(f'SELECT * FROM "{selected_table}"', st.session_state.db._engine)
                total_rows = st.session_state.df.shape[0]

                if "df" not in st.session_state:
                    st.session_state.df = st.session_state.df

            with st.form("search_form"):
                total_rows = st.session_state.df.shape[0]
                display_rows = st.number_input("Enter the number of rows to display:", placeholder="5", min_value=1, max_value=total_rows, value=min(5, total_rows))
                selected_columns = st.multiselect("Select columns to display:", st.session_state.df.columns.tolist(), default=st.session_state.df.columns.tolist())
                search_query = st.text_input("Filter:", "", placeholder="Enter the filter value you want to search in dataset")
                submitted = st.form_submit_button("Search")

            if submitted:
                if search_query:
                    filtered_df = st.session_state.df[selected_columns][st.session_state.df[selected_columns].apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
                    st.dataframe(filtered_df.head(display_rows))
                    st.session_state.df = filtered_df
                else:
                    filtered_df = st.session_state.df[selected_columns]
                    st.dataframe(filtered_df.head(display_rows))
                    st.session_state.df = filtered_df
            else:
                st.write(f"#### Data Preview for *{selected_table if 'db' in st.session_state else st.session_state.file_name}*") if 'db' in st.session_state or self.app.data else None
                st.dataframe(st.session_state.df[selected_columns].head(display_rows))
                st.session_state.df = st.session_state.df[selected_columns]
            
            st.write(f"*Total Dataset Rows:* {total_rows}; *Display Rows:* {display_rows}")
            
            # Show table information
            st.divider()
            st.write("#### Table Information")
            buffer = pd.io.common.StringIO()  
            st.session_state.df.info(buf=buffer)
            s = buffer.getvalue()  
            st.text(s) 

            if "data" in st.session_state:
                st.divider()
                st.write("##### Data Description")
                st.write(st.session_state.df.describe())

            # Export to csv
            st.divider()
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label = "Export to CSV",
                data=csv,
                file_name="exported_data.csv"
            )