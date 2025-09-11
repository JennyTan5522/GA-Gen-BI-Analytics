from src.tools.dataset_summary_tool import dataset_summary
from src.tools.question_recommendation_tool import generate_question_recommendations

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from src.ui.setup_st_config import is_data_and_llm_connected
from config.service_config import ServiceConfig
from config.logger import get_logger

logger = get_logger(__name__)

class DataTab:
    """
    Handles the Data View tab: summary, schema, filtering, and analysis.
    """
    def __init__(self):
        config = ServiceConfig()
        self.table_schema_info_path = config.TABLE_SCHEMA_INFO_PATH

        if not is_data_and_llm_connected():
            return
    
        self.display_table_selection()
        self.display_table_form()
        self.display_data_filter_and_preview()
        self.display_data_info()
        self.display_data_shape()
        self.display_data_overview()
        self.display_data_description()
        self.display_correlation_heatmap()
        self.display_univariate_analysis()
        self.display_bivariate_analysis()

    def display_table_selection(self):
        """
        Allow user to select a table from the database for preview and analysis.
        """
        st.write("#### Dataset Table Selection")
        tables = st.session_state.table_names
        logger.info(f"Tables in database: {tables}")
        selected_table = st.selectbox("Choose a table from the database for Data Preview:", tables)
        st.session_state.selected_table = selected_table

        try:
            if st.session_state.db.dialect == "bigquery":
                query = f'SELECT * FROM `{selected_table}`'
            else:
                query = f'SELECT * FROM "{selected_table}"'
            st.session_state.df = pd.read_sql(query, st.session_state.db._engine)
            st.session_state.df = st.session_state.df.dropna(axis=1, how="all")
            logger.debug(f"Loaded data for table: {selected_table}")
        except Exception as e:
            logger.error(f"Error loading data for table {selected_table}: {e}")
            st.error(f"Error loading data for table {selected_table}: {e}")
            return

    def display_table_form(self):
        """
        Render table summary, schema editor, and follow-up question generator.
        """
        selected_table = st.session_state.selected_table
        db_engine = str(st.session_state.db._engine.url)

        try:
            if db_engine.startswith("sqlite"):
                db_name = db_engine.split("/")[-1].split(".")[0]
            else:
                db_name = db_engine
        except Exception as e:
            logger.error(f"Error parsing database name from engine URL: {e}")
            db_name = ""

        json_path = f"{self.table_schema_info_path}/{db_name}/{selected_table}.json"
        json_path = json_path.replace(" ", "_").lower()

        editable_data = self._load_or_init_table_info(selected_table, json_path)
        self._sync_table_summary_follow_up_questions_input(selected_table)
        logger.info(f"Saved table data info into: {json_path}")
        with st.form("table_data"):
            self.display_table_summary(selected_table)
            edited_table_data = self.display_table_schema(editable_data)
            self.display_followup_questions(selected_table)
            self._handle_save_changes(selected_table, edited_table_data, json_path)

    def _load_or_init_table_info(self, selected_table: str, json_path: str) -> pd.DataFrame:
        """
        Load or initialize table info for schema editing.

        Args:
            selected_table (str): Table name.
            json_path (str): Path to JSON file.

        Returns:
            pd.DataFrame: Editable schema data.
        """
        editable_data = None
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                saved_data = json.load(f)
            if selected_table in saved_data:
                full_table_info = saved_data[selected_table]
                editable_data = pd.DataFrame(full_table_info.get("table_columns_info", []))
                st.session_state.tables_info[selected_table] = full_table_info
        if editable_data is None:
            original_columns = st.session_state.df.columns.tolist()
            editable_data = pd.DataFrame({
                "Dataset Column Name": original_columns,  
                "User Define Column Name": [""] * len(original_columns), 
                "Description": [""] * len(original_columns), 
                "Calculation/Formula": [""] * len(original_columns) 
            })
        return editable_data

    def _sync_table_summary_follow_up_questions_input(self, selected_table: str):
        """
        Sync table summary input with session state.

        Args:
            selected_table (str): Table name.
        """
        if ("last_selected_table" not in st.session_state or 
            st.session_state.last_selected_table != selected_table):
            st.session_state.table_summary_input = (
                st.session_state.get("tables_info", {}).get(selected_table, {}).get("table_summary", "")
            )
            st.session_state.followup_questions = (
                st.session_state.get("tables_info", {}).get(selected_table, {}).get("followup_questions", {})
            )
            st.session_state.last_selected_table = selected_table

    def display_table_summary(self, selected_table: str):
        """
        Render table summary text area and generator.

        Args:
            selected_table (str): Table name.
        """
        st.write("#### 📝 Table Summary")
        st.write("[Optional] Briefly describe what this table is about.")
        if st.form_submit_button("Generate Table Summary"):
            sample_dataset_str = st.session_state.df.head(10).to_csv(sep="|", index=False, lineterminator="\n")
            generated_summary = dataset_summary(
                st.session_state.llm,
                sample_dataset_str,
                st.session_state.df.shape[0],
                st.session_state.df.shape[1]
            )
            st.session_state.table_summary_input = generated_summary
            logger.info(f"Generated table summary for {selected_table}: {generated_summary}")
        st.text_area(
            label="", 
            value=st.session_state.table_summary_input, 
            placeholder="This table contains sales data for Q1 2025, including product details and revenue.",
            height=200, 
            label_visibility="collapsed"
        )
        st.divider()

    def display_table_schema(self, editable_data: pd.DataFrame) -> pd.DataFrame:
        """
        Render editable table schema.

        Args:
            editable_data (pd.DataFrame): Schema data.

        Returns:
            pd.DataFrame: Edited schema data.
        """
        st.write("#### 🔍 Table Schema Info")
        st.write("[Optional] Edit the full column names, add descriptions, or define formulas as needed.")
        return st.data_editor(
            editable_data,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "Dataset Column Name": st.column_config.Column(disabled=True, width="fit"), 
                "User Define Column Name": st.column_config.Column(width="fit"),
                "Description": st.column_config.Column(width=400),
                "Calculation/Formula": st.column_config.Column(width=400)
            }
        )

    def display_followup_questions(self, selected_table):
        """
        Render follow-up analysis question generator and display.

        Args:
            selected_table (str): Table name.
        """
        st.divider()
        st.write("#### 💡 Follow-Up Analysis Exploration Questions")
        if st.form_submit_button("Generate Follow-Up Analysis Questions"):
            sample_dataset_str = st.session_state.df.head(10).to_csv(sep="|", index=False, lineterminator="\n")
            followup_questions = generate_question_recommendations(st.session_state.llm, sample_dataset_str)
            st.session_state.followup_questions = followup_questions
            logger.info(f"Generated follow-up questions for {selected_table}: {followup_questions}")
        
        if st.session_state.followup_questions:
            grouped_questions = self._group_questions_by_category(st.session_state.followup_questions)
            self._display_grouped_questions(grouped_questions)
        st.divider()
       
    def _group_questions_by_category(self, followup_questions: dict) -> dict:
        """
        Group follow-up questions by category.

        Args:
            followup_questions (dict): Questions with categories.

        Returns:
            dict: Grouped questions.
        """
        grouped = {}
        for item in followup_questions["questions"]:
            category = item["category"]
            question = item["question"]
            grouped.setdefault(category, []).append(question)
        return grouped

    def _display_grouped_questions(self, grouped_questions):
        """
        Display grouped follow-up questions in columns.

        Args:
            grouped_questions (dict): Grouped questions.
        """
        st.markdown(
            """<p style="font-size: 11px; color: gray;"> A list of automatically generated data exploration goals based on the dataset given.</p>""",
            unsafe_allow_html=True
        )
        categories = list(grouped_questions.keys())
        columns = st.columns(3)
        for idx, category in enumerate(categories):
            col = columns[idx % 3]
            with col.container():
                st.markdown(
                    f"""
                    <div style="border-radius: 8px; padding-left: 10px; background-color: #f9f9f9; 
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1); width: 100%;">
                        <h4 style="color: #333; font-size: 13px; font-weight: bold; margin: 0;">{category}</h4>
                    </div>
                    <div style="padding-left: 15px; margin-top: 5px;">
                        <ul style="padding-left: 15px; color: #333;">
                            {''.join(f'<li style="margin-bottom: 4px; font-size: 11px; text-align: justify;">{q}</li>' for q in grouped_questions[category])}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    def _handle_save_changes(self, selected_table: str, edited_table_data: pd.DataFrame, json_path: str):
        """
        Save edited table schema and summary to disk.

        Args:
            selected_table (str): Table name.
            edited_table_data (pd.DataFrame): Edited schema.
            json_path (str): Path to save JSON.
        """
        st.markdown('<span style="color:red">Notes: Please ensure that <strong>all changes are SAVED</strong> after editing or generating from LLM.</span>', unsafe_allow_html=True)
        if st.form_submit_button("Save Changes"):
            full_table_info = {
                "table_name": selected_table,
                "table_summary": st.session_state.table_summary_input,
                "table_columns_info": edited_table_data.to_dict(orient="records"),
                "followup_questions": st.session_state.followup_questions 
            }
            st.session_state.tables_info[selected_table] = full_table_info
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(st.session_state.tables_info, f, indent=2)
            st.success("Changes saved successfully!")

    def display_data_filter_and_preview(self):
        """
        Render data filter, preview, and export options.
        """
        st.write("#### Data Filter and Preview")
        with st.form("search_form"):
            total_rows = st.session_state.df.shape[0]
            selected_columns = st.multiselect(
                "📌 Select Columns for Display", 
                st.session_state.df.columns.tolist(), 
                default=st.session_state.df.columns.tolist()
            )
            row_col, display_col = st.columns([1, 1])
            with row_col:
                display_rows = st.number_input(
                    "🔹 Enter the number of rows to display:", 
                    placeholder="5", min_value=1, max_value=total_rows, value=min(10, total_rows)
                )
            with display_col:
                search_query = st.text_input(
                    "🔹 Filter dataset by value:", "", placeholder="🔍 Enter the filter value you want to search in dataset"
                )
            submitted = st.form_submit_button("Search")
            if submitted:
                filtered_df = self._filter_dataframe(selected_columns, search_query)
                st.dataframe(filtered_df.head(display_rows))
                st.session_state.df = filtered_df
            else:
                preview_df = st.session_state.df[selected_columns]
                st.dataframe(preview_df.head(display_rows))
                st.session_state.df = preview_df
            st.write(f"**Total Dataset Rows:** {total_rows}; **Display Rows:** {display_rows}")
        self.display_export_csv()

    def _filter_dataframe(self, selected_columns: list, search_query: str) -> pd.DataFrame:
        """
        Filter dataframe by selected columns and search query.

        Args:
            selected_columns (list): Columns to display.
            search_query (str): Search string.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        df = st.session_state.df[selected_columns]
        if search_query:
            return df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)].reset_index(drop=True)
        return df.reset_index(drop=True)

    def display_export_csv(self):
        """
        Render export to CSV button for current dataframe.
        """
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv,
            file_name="exported_data.csv"
        )

    def display_data_info(self):
        """
        Display dataframe info, unique values, and missing counts.
        """
        buffer = pd.io.common.StringIO()
        st.session_state.df.info(buf=buffer)
        info_text = buffer.getvalue().split("\n")
        columns = []
        for line in info_text[5:-3]:
            parts = line.split()
            if len(parts) >= 4:
                col_name = parts[1]
                non_null_count = parts[-3] + " " + parts[-2]
                dtype = parts[-1]
                columns.append([col_name, non_null_count, dtype])
        df_info = pd.DataFrame(columns, columns=["Feature", "Non-Null Count", "Data Type"])
        df_info["Unique Count"] = df_info["Feature"].apply(lambda col: st.session_state.df[col].nunique(dropna=True))
        df_info["Unique Values"] = df_info["Feature"].apply(
            lambda col: ", ".join(map(str, st.session_state.df[col].dropna().unique()))
            if st.session_state.df[col].nunique(dropna=True) <= 10
            else "The feature has high cardinality; unique values greater than 10 are not displayed."
        )
        total_rows = len(st.session_state.df)
        df_info["Missing Count"] = df_info["Feature"].apply(
            lambda col: f"{st.session_state.df[col].isna().sum()} ({(st.session_state.df[col].isna().sum() / total_rows * 100):.2f}%)"
        )
        st.divider()
        st.write("#### Data Overview")
        df_info.index = pd.RangeIndex(start=1, stop=len(df_info) + 1, step=1)
        st.dataframe(df_info.style.applymap(self._highlight_missing, subset=["Missing Count"]))
        st.markdown(f"There are **{st.session_state.df.duplicated().sum()}** duplicated row(s) in this {st.session_state.selected_table} table.")

    def _highlight_missing(self, val: str) -> str:
        """
        Highlight missing value cells in dataframe info.

        Args:
            val (str): Missing count string.

        Returns:
            str: CSS style.
        """
        count = int(val.split(" ")[0])
        color = "green" if count == 0 else "red"
        return f"background-color: {color}; color: white;"

    def display_data_shape(self):
        """
        Display shape (rows, columns) of the dataframe.
        """
        st.divider()
        st.write("#### Data Shape")
        st.write(f"Number of Rows: {st.session_state.df.shape[0]}, Number of Columns: {st.session_state.df.shape[1]}")

    def display_data_overview(self):
        """
        Display statistical overview of the dataframe.
        """
        st.divider()
        st.write("#### Data Overview")
        st.write(st.session_state.df.describe())

    def display_data_description(self):
        """
        Display data description (same as overview).
        """
        st.divider()
        st.write("#### Data Description")
        st.write(st.session_state.df.describe())

    def display_correlation_heatmap(self):
        """
        Display correlation heatmap for numeric columns.
        """
        st.divider()
        st.write("#### Correlation Heatmap")
        numeric_cols = st.session_state.df.select_dtypes(include=['number'])
        corr_matrix = numeric_cols.corr()
        styled_corr = corr_matrix.style.background_gradient(cmap="coolwarm", axis=None)
        st.dataframe(styled_corr)

    def display_univariate_analysis(self):
        """
        Render univariate analysis for categorical and numerical columns.
        """
        st.divider()
        st.write("#### Univariate Analysis")
        numerical_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.to_list()
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.to_list()
        with st.form("univariate_analysis_form"):
            col1_area, col2_area = st.columns(2)
            with col1_area:
                col1 = st.multiselect("🔹 Select Categorical Columns:", categorical_cols, default=categorical_cols)
            with col2_area:
                col2 = st.multiselect("🔹 Select Numerical Columns:", numerical_cols, default=numerical_cols)
            selected_columns = col1 + col2
            user_input_bins = st.number_input(
                "Choose the number of bins for the histogram to analyze numerical columns:",
                min_value=5, max_value=50, value=20, step=1
            )
            analyze_btn = st.form_submit_button("🚀 Analyze")
            if analyze_btn:
                for selected_col in selected_columns:
                    if selected_col in categorical_cols:
                        self._plot_categorical_univariate(selected_col)
                    elif selected_col in numerical_cols:
                        self._plot_numerical_univariate(selected_col, user_input_bins)

    def _plot_categorical_univariate(self, selected_col: str) -> None:
        """
        Plot bar and pie chart for a categorical column.

        Args:
            selected_col (str): Column name.
        """
        unique_count = st.session_state.df[selected_col].nunique()
        if unique_count < 10:
            value_counts = st.session_state.df[selected_col].value_counts().reset_index()
            value_counts.columns = [selected_col, "count"]
            fig = make_subplots(rows=1, cols=2, 
                                subplot_titles=[f"Bar Chart of {selected_col}", f"Pie Chart of {selected_col}"], 
                                specs=[[{"type": "bar"}, {"type": "pie"}]])
            fig.add_trace(go.Bar(x=value_counts[selected_col], y=value_counts["count"], text=value_counts["count"], textposition="outside", name="Bar Chart"), row=1, col=1)
            fig.add_trace(go.Pie(labels=value_counts[selected_col], values=value_counts["count"], name="Pie Chart"), row=1, col=2)
            fig.update_layout(title_text=f"Visualization of {selected_col}", showlegend=True)
            st.plotly_chart(fig)
        else:
            st.warning(f"{selected_col} feature has high cardinality; unique values greater than 10 are not displayed.")

    def _plot_numerical_univariate(self, selected_col: str, user_input_bins: int) -> None:
        """
        Plot histogram, KDE, and boxplot for a numerical column.

        Args:
            selected_col (str): Column name.
            user_input_bins (int): Number of bins for histogram.
        """
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Histogram of {selected_col}", f"Boxplot of {selected_col}"])
        hist_data = st.session_state.df[selected_col].dropna()
        counts, bins = np.histogram(hist_data, bins=user_input_bins, density=True)
        kde = gaussian_kde(hist_data)
        x_vals = np.linspace(min(hist_data), max(hist_data), 200)
        y_vals = kde(x_vals)
        fig.add_trace(go.Histogram(
            x=hist_data, nbinsx=user_input_bins, histnorm='probability density',
            name="Histogram", marker=dict(color="blue", opacity=0.6),
            text=[f"{c:.3f}" for c in counts], textposition="outside"
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines', name="KDE Distribution", line=dict(color="red", width=2)
        ))
        fig.update_layout(title=f"Histogram & KDE of {selected_col}", xaxis_title=selected_col, yaxis_title="Density")
        fig.add_trace(go.Box(y=st.session_state.df[selected_col]), row=1, col=2)
        fig.update_layout(title_text=f"Distribution of {selected_col}")
        st.plotly_chart(fig)

    def display_bivariate_analysis(self):
        """
        Render bivariate analysis for selected column pairs.
        """
        st.divider()
        st.write("#### Bivariate Analysis")
        st.markdown("""
            ##### 🔍 Bivariate Analysis Guide:
            - **Categorical vs. Categorical** → Stacked Bar Chart  & Grouped Bar Chat & Grouped Pie Chart
            - **Numerical vs. Numerical**     → Scatter Plot & Heat Map
            - **Numerical vs. Categorical**   → Box Plot  
        """)
        numerical_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.to_list()
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.to_list()
        with st.form("bivariate_analysis_form"):
            st.markdown(f"**Categorical Variables:**  \n{', '.join(categorical_cols) if categorical_cols else 'None'}")
            st.markdown(f"**Numerical Variables:**  \n{', '.join(numerical_cols) if numerical_cols else 'None'}")
            st.markdown("---")
            col1_area, col2_area = st.columns(2)
            with col1_area:
                col1 = st.selectbox("🔹 Select First Column:", st.session_state.df.columns, index=0)
            with col2_area:
                col2 = st.selectbox("🔹 Select Second Column:", st.session_state.df.columns, index=1)
            bivariate_submit_btn = st.form_submit_button("🚀 Analyze")
            if bivariate_submit_btn:
                self._plot_bivariate(col1, col2, numerical_cols, categorical_cols)

    def _plot_bivariate(self, col1: str, col2: str, numerical_cols: list, categorical_cols: list) -> None:
        """
        Plot appropriate bivariate chart based on column types.

        Args:
            col1 (str): First column.
            col2 (str): Second column.
            numerical_cols (list): List of numerical columns.
            categorical_cols (list): List of categorical columns.
        """
        if col1 and col2 and col1 != col2:
            unique_col1 = st.session_state.df[col1].nunique()
            unique_col2 = st.session_state.df[col2].nunique()
            max_unique_threshold = 10
            if (col1 in categorical_cols and unique_col1 > max_unique_threshold) or \
               (col2 in categorical_cols and unique_col2 > max_unique_threshold):
                st.warning(f"Skipping {col1} and {col2} due to high cardinality.")
            else:
                if col1 in numerical_cols and col2 in numerical_cols:
                    self._plot_numerical_vs_numerical(col1, col2)
                elif (col1 in numerical_cols and col2 in categorical_cols) or (col1 in categorical_cols and col2 in numerical_cols):
                    self._plot_numerical_vs_categorical(col1, col2)
                elif col1 in categorical_cols and col2 in categorical_cols:
                    self._plot_categorical_vs_categorical(col1, col2)
        else:
            st.warning("Please select 2 different columns to perform analysis.")

    def _plot_numerical_vs_numerical(self, col1: str, col2: str) -> None:
        """
        Plot scatter and correlation heatmap for two numerical columns.

        Args:
            col1 (str): First column.
            col2 (str): Second column.
        """
        fig = px.scatter(st.session_state.df, x=col1, y=col2, trendline="ols", title=f"Scatter Plot of {col1} vs {col2}")
        fig.update_layout(xaxis_title=col1, yaxis_title=col2)
        st.plotly_chart(fig)
        corr_matrix = st.session_state.df[[col1, col2]].corr().values
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", labels=dict(color="Correlation"), title=f"Correlation Heatmap of {col1} vs {col2}")
        fig_corr.update_xaxes(title_text=col1, tickvals=[0, 1], ticktext=[col1, col2])
        fig_corr.update_layout(width=800, height=600)
        st.plotly_chart(fig_corr)

    def _plot_numerical_vs_categorical(self, col1: str, col2: str) -> None:
        """
        Plot boxplot for numerical vs categorical columns.

        Args:
            col1 (str): Numerical column.
            col2 (str): Categorical column.
        """
        fig = px.box(st.session_state.df, x=col2, y=col1, title=f"Box Plot of {col1} by {col2}")
        st.plotly_chart(fig)

    def _plot_categorical_vs_categorical(self, col1: str, col2: str) -> None:
        """
        Plot stacked bar, grouped bar, and pie chart for two categorical columns.

        Args:
            col1 (str): First categorical column.
            col2 (str): Second categorical column.
        """
        cross_tab = st.session_state.df.groupby([col1, col2]).size().reset_index(name="Count")
        fig = px.bar(cross_tab, x=col1, y="Count", color=col2, title=f"Stacked Bar Chart of {col1} vs {col2}", barmode="stack", text="Count")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig)
        fig_grouped = px.bar(cross_tab, x=col1, y="Count", color=col2, title=f"Grouped Bar Chart of {col1} vs {col2}", barmode="group", text_auto=True)
        st.plotly_chart(fig_grouped)
        fig_pie = px.pie(cross_tab, values="Count", names=col2, title=f"Proportion of {col2} within {col1}", hole=0.4)
        st.plotly_chart(fig_pie)
