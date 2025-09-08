import ast
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, Field, model_validator

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool
from langchain_experimental.utilities.python import PythonREPL

def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)

def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.

    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """
    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)(?i:python)?\s", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query

def remove_code_newlines(code: str) -> str:
    """Remove all new lines in the code

    Args:
        code: Python code with or without new lines

    Returns:
        str: Python code without new lines
    """
    return code.replace('\n', '')

class PythonREPLTool(BaseTool):
    """Tool for checking Python code syntax without executing it.
        This tool is used to validate Python commands or code snippets.
        It analyzes the code for syntax errors and issues without actually executing the code in the Python environment.
        If the generated Python code is invalid, use the sanitize_input function to sanitize and correct the input.
        This tool only checks the Python code's syntax and structure, ensuring that it is correct before proceeding further.
    """
    name: str = "python_repl_ast"
    description: str = (
        "A tool for validating Python code. "
        "This tool only checks the syntax and structure of the code provided, without executing it in the background. "
        "It helps ensure that the code is free from syntax errors and issues before execution. "
        "If the generated Python code is includes any python at the beginning and  at the end, use the sanitize_input function to sanitize and correct the input."
    )

    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)
            
        # Remove newlines from the query
        query = remove_code_newlines(query)
        return query

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.sanitize_input:
            query = sanitize_input(query)
        # Remove newlines from the query
        query = remove_code_newlines(query)
        return await run_in_executor(None, self.run, query)

class PythonInputs(BaseModel):
    """Python inputs."""
    query: str = Field(description="code snippet to run")

class PythonAstREPLTool(BaseTool):
    """Tool for running python code in a REPL."""

    name: str = "python_repl_ast"
    description: str = (
        "A tool for validating Python code. "
        "This tool only checks the syntax and structure of the code provided, without executing it in the background. "
        "It helps ensure that the code is free from syntax errors and issues before execution. "
        "If the generated Python code is includes any python at the beginning and  at the end, use the sanitize_input function to sanitize and correct the input."

        # "A Python shell. Use this to execute python commands. "
        # "Input should be a valid python command. "
        # "When using this tool, sometimes output is abbreviated - "
        # "make sure it does not look abbreviated before using it in your answer."
    )

    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    args_schema: Type[BaseModel] = PythonInputs

    @model_validator(mode="before")
    @classmethod
    def validate_python_version(cls, values: Dict) -> Any:
        """Validate valid python version."""
        if sys.version_info < (3, 9):
            raise ValueError(
                "This tool relies on Python 3.9 or higher "
                "(as it uses new functionality in the ast module, "
                f"you have Python version: {sys.version}"
            )
        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            io_buffer = StringIO()
            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return io_buffer.getvalue()
        except Exception as e:
            return "{}: {}".format(type(e)._name_, str(e))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, query)