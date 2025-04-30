from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from const import FOLLOWUP_TEMPLATE

from langchain.chains.llm import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class FollowUpQuestionInput(BaseModel):
    query: str = Field(description="User Query")

class FollowUpQuestionTool(BaseTool):
    name: str = "FollowUpQuestionTool"
    
    """Use an LLM to generate follow-up questions from text and return as Final Answer for the user to clarify"""

    name: str = "FollowUpQuestionTool"
    description: str = "Use this tool to generate follow-up questions based on the user's chat message to clarify their intent."
    args_schema: Type[BaseModel] = FollowUpQuestionInput
    template: str = FOLLOWUP_TEMPLATE
    llm: BaseLanguageModel
    return_direct: bool = False

    def _init_(self, llm: BaseLanguageModel, **kwargs: Any) -> None:
        super()._init_(llm=llm, **kwargs)
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        prompt = PromptTemplate(template=self.template, input_variables=["query"])
        chain = LLMChain(
                    llm=self.llm,
                    prompt=prompt
                )
        out = chain.predict(query=query, callbacks=run_manager.get_child() if run_manager else None)
        return out

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        prompt = PromptTemplate(template=self.template, input_variables=["query"])
        chain = LLMChain(
                    llm=self.llm,
                    prompt=prompt
                )
        out = await chain.apredict(query=query, callbacks=run_manager.get_child() if run_manager else None)
        return out