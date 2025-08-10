import re
from typing import Any, Dict, List, Optional, Sequence, Union
from sqlalchemy.engine import Result
from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)

class CustomSQLModel(BaseModel):
    query: str = Field(description="SQL Query")

class CustomSQLToolkit(SQLDatabaseToolkit):
    class _CustomQuerySQLDataBaseTool(QuerySQLDataBaseTool):
        description: str = (
            "Execute a SQL query against the database and get back the result.."
            "If the query is not correct, an error message will be returned."
            "If an error is returned, rewrite the query, check the query, and try again."
        )
            
        def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> Union[str, Sequence[Dict[str, Any]], Result]:
            
            """Execute the query, return the results or an error message."""
            if "sql" in query:
                query = re.search(r"sql(.*?)```", query, re.DOTALL).group(1)

            result = self.db.run_no_throw(query)
            if not result :
                return "SQL Query returned no data"
            else:
                return result

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_sql_database_tool = ListSQLDatabaseTool(db=self.db)
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Do not add single quote or double quote around the table names."
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db, description=info_sql_database_tool_description
        )
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
        )
        
        query_sql_database_tool = self._CustomQuerySQLDataBaseTool(
            db=self.db, description=query_sql_database_tool_description
        )
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )

        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db, llm=self.llm, description=query_sql_checker_tool_description
        )

        return [
            query_sql_database_tool,
            info_sql_database_tool,
            list_sql_database_tool,
            query_sql_checker_tool,
        ]