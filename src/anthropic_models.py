from llm_agent_templates import prompt_template3
from apikeys import CLAUDE_API_KEY
import anthropic

client = anthropic.Anthropic(
    api_key=CLAUDE_API_KEY
)

def mcq_anthropic_output(user_query, final_context):
    """
    Generates a multiple-choice question (MCQ) response using Claude-3-5-Haiku.

    Args:
        user_query (str): The users query.
        final_context (str): Additional contextual information.

    Returns:
        str: The generated MCQ response.
    """

    prompt = prompt_template3.format(query=user_query, context=final_context)

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1,  # Set to 1 to get just the answer
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text.strip()