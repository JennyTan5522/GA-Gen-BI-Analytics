import json
import re
from langchain_core.prompts import ChatPromptTemplate
from const import QUESTION_RECOMMENDATION_PROMPT_TEMPALTE
from Logger_Manager import logger

def format_question(question_recommendations: str) -> str:
    try:
        match = re.search(r'json\n(.*?)\n', question_recommendations, re.DOTALL)

        if match:
            json_str = match.group(1)
            return json.loads(json_str)
    except Exception as e:
        logger.debug(f"Error in formatting question recommendations: {e}")

    return None

async def generate_question_recommendations_async(llm, sample_dataset_str: str) -> str:
    """Asynchronous function to generate question recommendations"""
    prompt_template = ChatPromptTemplate.from_template(QUESTION_RECOMMENDATION_PROMPT_TEMPALTE)
    prompt = prompt_template.format(sample_dataset=sample_dataset_str)
    question_recommendations = await llm.apredict(prompt)
    return format_question(question_recommendations)

def generate_question_recommendations(llm, sample_dataset_str: str) -> str:
    """Asynchronous function to generate question recommendations"""
    prompt_template = ChatPromptTemplate.from_template(QUESTION_RECOMMENDATION_PROMPT_TEMPALTE)
    prompt = prompt_template.format(sample_dataset=sample_dataset_str)
    question_recommendations = llm.predict(prompt)
    return format_question(question_recommendations)