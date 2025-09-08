import streamlit as st
import pandas as pd
from config.logger import get_logger

logger = get_logger(__name__)

class DataTab:
    def __init__(self):
        """Initialize DataTab and handle the data tab UI."""
        self.handle_data_tab()

    def handle_data_tab(self):
        """Display and manage the data preview, filtering, and summary UI."""
        if not st.session_state.is_data_and_llm_connected:
            return

        st.write("#### Dataset Table Selection")
        selected_table = st.selectbox("Choose a table from the database for Data Preview:", st.session_state.table_names)
        st.session_state.selected_table = selected_table

        try:
            if st.session_state.db.dialect == "bigquery":
                query = f'SELECT * FROM `{selected_table}`'
            else:
                query = f'SELECT * FROM "{selected_table}"'
            st.session_state.df = pd.read_sql(query, st.session_state.db._engine)
            logger.debug(f"Loaded data for table: {selected_table}")
        except Exception as e:
            logger.error(f"Error loading data for table {selected_table}: {e}")
            st.error(f"Error loading data for table {selected_table}: {e}")
            return

        st.session_state.df = st.session_state.df.dropna(axis=1, how="all")
        total_rows = st.session_state.df.shape[0]
        table_summary = ""
        editable_data = None

        with st.form("table_data"):
            st.write("#### Table Summary")
            st.write("[Optional] Briefly describe what this table is about.")
            table_summary = st.text_area("", placeholder="This table contains sales data for Q1 2025, including product details and revenue.")

            original_columns = st.session_state.df.columns.tolist()

            # Initialize editable table if not already set
            if editable_data is None:
                editable_data = pd.DataFrame({
                    "Dataset Column Name": original_columns,
                    "User Define Column Name": [""] * len(original_columns),
                    "Description": [""] * len(original_columns),
                    "Calculation/Formula": [""] * len(original_columns)
                })

            st.write("#### Table Data")
            st.write("[Optional] Edit the full column names, add descriptions, or define formulas as needed.")
            edited_table_data = st.data_editor(
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

            if st.form_submit_button("Save Changes"):
                # Save table info to session state
                full_table_info = {
                    "table_name": selected_table,
                    "table_summary": table_summary,
                    "table_columns_info": edited_table_data.to_dict(orient="records")
                }
                st.session_state.tables_info[selected_table] = full_table_info
                logger.info(f"Saved table info for: {selected_table}")

        st.divider()

        st.write("#### Data Filter and Preview")
        with st.form("search_form"):
            total_rows = st.session_state.df.shape[0]
            selected_columns = st.multiselect("📌 Select Columns for Display", st.session_state.df.columns.tolist(), default=st.session_state.df.columns.tolist())
            row_col, display_col = st.columns([1, 1])
            with row_col:
                display_rows = st.number_input("🔹 Enter the number of rows to display:", placeholder="5", min_value=1, max_value=total_rows, value=min(10, total_rows))
            with display_col:
                search_query = st.text_input("🔹 Filter dataset by value:", "", placeholder="🔍 Enter the filter value you want to search in dataset")

            submitted = st.form_submit_button("Search")

            if submitted:
                try:
                    if search_query:
                        filtered_df = st.session_state.df[selected_columns][
                            st.session_state.df[selected_columns].apply(
                                lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1
                            )
                        ]
                    else:
                        filtered_df = st.session_state.df[selected_columns]
                    filtered_df.index = pd.RangeIndex(start=1, stop=len(filtered_df) + 1, step=1)
                    st.dataframe(filtered_df.head(display_rows))
                    st.session_state.df = filtered_df
                    logger.debug(f"Filtered data with query: '{search_query}'")
                except Exception as e:
                    logger.error(f"Error filtering data: {e}")
                    st.error(f"Error filtering data: {e}")
            else:
                preview_df = st.session_state.df[selected_columns]
                preview_df.index = pd.RangeIndex(start=1, stop=len(preview_df) + 1, step=1)
                st.dataframe(preview_df.head(display_rows))
                st.session_state.df = preview_df

            st.write(f"*Total Dataset Rows:* {total_rows}; *Display Rows:* {display_rows}")

        try:
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Export to CSV",
                data=csv,
                file_name="exported_data.csv"
            )
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {e}")
            st.error(f"Error exporting data to CSV: {e}")

        # Show table information using df.info
        buffer = pd.io.common.StringIO()
        try:
            st.session_state.df.info(buf=buffer)
            info_text = buffer.getvalue().split("\n")
        except Exception as e:
            logger.error(f"Error getting DataFrame info: {e}")
            st.error(f"Error getting DataFrame info: {e}")
            info_text = []

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

        def highlight_missing(val: str):
            """Highlight missing count cells based on value."""
            count = int(val.split(" ")[0])
            color = "green" if count == 0 else "red"
            return f"background-color: {color}; color: white;"

        df_info["Null Count"] = df_info["Feature"].apply(
            lambda col: st.session_state.df[col].isna().sum()
        )
        df_info["Empty Count"] = df_info["Feature"].apply(
            lambda col: (
                st.session_state.df[col].astype(str).str.strip().eq("").sum()
                if pd.api.types.is_string_dtype(st.session_state.df[col]) or pd.api.types.is_object_dtype(st.session_state.df[col])
                else 0
            )
        )

        total_rows = len(st.session_state.df)

        def calculate_missing_count(col: str):
            """Calculate missing value count and percentage for a column."""
            series = st.session_state.df[col]
            if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
                missing = (series.isna() | series.astype(str).str.strip().eq("")).sum()
            else:
                missing = series.isna().sum()
            percent = (missing / total_rows) * 100
            return f"{missing} ({percent:.2f}%)"

        df_info["Missing Count"] = df_info["Feature"].apply(calculate_missing_count)

        st.divider()
        st.write("#### Data Shape")
        st.write(f"Number of Rows: {st.session_state.df.shape[0]}, Number of Columns: {st.session_state.df.shape[1]}")

        st.divider()
        st.write("#### Data Overview")
        df_info.index = pd.RangeIndex(start=1, stop=len(df_info) + 1, step=1)
        st.dataframe(df_info.style.applymap(highlight_missing, subset=["Missing Count"]))
        st.markdown(f"There are *{st.session_state.df.duplicated().sum()}* duplicated row(s) in this {selected_table} table.")
        
        # Step 5: Data Description
        st.divider()
        st.write("#### Data Description")
        st.write(st.session_state.df.describe())