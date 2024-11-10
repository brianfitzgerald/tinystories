from synthetic_data.generation import Conversation
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)


def format_dalle_prompt_template(user_prompt: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message = {
        "role": "system",
        "content": """You are part of a team of bots that creates images. You work with an assistant bot that will draw anything you say in square brackets.
For example, outputting "a beautiful morning in the woods with the sun peaking through the trees" will trigger your partner bot to output an image of a forest morning, as described.
You will be prompted by people looking to create detailed, amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.

There are a few rules to follow:

- You will only ever output a single image description per user request.
- Sometimes the user will request that you modify previous captions. In this case, you should refer to your previous conversations with the user and make the modifications requested.
- When modifications are requested, you should not simply make the description longer. You should refactor the entire description to integrate the suggestions.
- Other times the user will not want modifications, but instead want a new image. In this case, you should ignore your previous conversation with the user."
- Image descriptions must be between 15-80 words. Extra words will be ignored.
    """,
    }

    last_msg_content = (
        f"Create an imaginative image descriptive caption or modify an earlier caption for the user input : '{user_prompt}'",
    )

    user_conversation = [
        system_message,
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'make the light red'",
        },
        {
            "role": "assistant",
            "content": "a pale figure with long white hair stands in the center of a dark forest, holding a sword high above his head. the blade glows with a red light, casting a warm glow on the trees and bushes surrounding him.",
        },
        {
            "role": "user",
            "content": "Create an imaginative image descriptive caption or modify an earlier caption for the user input : 'draw a frog playing dominoes'",
        },
        {
            "role": "assistant",
            "content": "a frog sits on a worn table playing a game of dominoes with an elderly raccoon. the table is covered in a green cloth, and the frog is wearing a jacket and a pair of jeans. The scene is set in a forest, with a large tree in the background.",
        },
        {"role": "user", "content": last_msg_content},
    ]
    return user_conversation


ENTITY_EXTRACTION_EXAMPLE_GENERATION_INSTRUCTION = """
Give an example of structured data extraction from the following context in JSON form. Return a query that requests data in a specific schema about an entity or event in the context, and the resulting data returned by that query.
Limit to only factual information about the subjects of the query, such as names, dates, and other properties of the entities.
"""

ENTITY_EXTRACTION_TUNING_INSTRUCTION = """
Extract structured data from the following context in JSON form.
"""


def format_entity_extraction_conversation_template(context: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the SQuAD dataset:
    https://rajpurkar.github.io/SQuAD-explorer/
    """

    prompt = (
        f"{ENTITY_EXTRACTION_EXAMPLE_GENERATION_INSTRUCTION}<context>\n{context}\n</context>"
        ""
    )

    user_message: ChatCompletionMessageParam = {
        "role": "user",
        "content": prompt,
    }

    return [user_message]


CONVERT_WEIGHT_API_EXAMPLE = {
    "name": "convert_weight",
    "description": "Convert weight from one unit to another. Returns the converted weight.",
    "parameters": {
        "type": "object",
        "properties": {
            "weight": {"type": "number", "description": "The weight value"},
            "from_unit": {"type": "string", "description": "The unit to convert from"},
            "to_unit": {"type": "string", "description": "The unit to convert to"},
        },
        "required": ["weight", "from_unit", "to_unit"],
    },
}

CONVERT_WEIGHT_USAGE_EXAMPLE = {
    "weight": 10,
    "from_unit": "pounds",
    "to_unit": "kilograms",
}

HISTORICAL_WEATHER_API_EXAMPLE = {
    "name": "historical_weather_data",
    "description": "Retrieve historical weather data for a location. Returns the weather data for a specific date.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location for which to retrieve weather data",
            },
            "date": {
                "type": "string",
                "description": "The date for which to retrieve historical weather data",
            },
        },
        "required": ["location", "date"],
    },
}

HISTORICAL_WEATHER_USAGE_EXAMPLE = {"location": "New York", "date": "2022-01-01"}
HISTORICAL_WEATHER_RESULT_EXAMPLE = {
    "temperature": 32,
    "weather_condition": "Snow",
    "wind_speed": 10,
}

JSON_TOOL_USE_EXAMPLES = """
Task: Convert weight
API: {CONVERT_WEIGHT_API_EXAMPLE}
User: Convert 10 pounds to kilograms
Call: {CONVERT_WEIGHT_USAGE_EXAMPLE}
Result: 4.53592
Agent: 10 pounds is equal to 4.53592 kilograms.

Task: Retrieve historical weather data
API: {HISTORICAL_WEATHER_API_EXAMPLE}
User: Convert 10 pounds to kilograms
Call: {HISTORICAL_WEATHER_USAGE_EXAMPLE}
Result: {HISTORICAL_WEATHER_RESULT_EXAMPLE}
Agent: On January 1, 2020, in New York City, the temperature was 32°F with snowfall and a wind speed of 10 mph.
"""

JSON_TOOL_USAGE_GEN_PROMPT = """
Generate an example of an API in the category of {category} that could be used to {task}.
Provide the API in the form of a JSON definition. Follow the example below.
Then, provide an example of a user query that would require the API to be called to answer. Phrase the query as a question a real user would ask, such as "What is the weather in New York on January 1, 2020?"
Then, provide an example of the tool's output to the API call. Always use realistic places and names when providing examples. Do not make up fake URLs, references, or names.
Finally, provide an example of the agent's output to the user query. Always integrate the result of the tool output into the agent's response.

Do not use any emoji or special characters in your response.

For example:

{tool_use_examples}
"""

JSON_TOOL_USAGE_NEGATIVE_SAMPLE_PROMPT = """
Generate an example of an API in the category of {category} that could be used to {task}.
You are given a JSON definition of an API and an example of a user query that would be used to perform the task.
Provide an example of the tool's output to the API call. Always use realistic places and names when providing examples. Do not make up fake URls, references, or names.
Do not use any emoji or special characters in your response.

For example:
{tool_use_examples}

Task: {task}
API: {api_definition}
User: {user_query}

"""


CATEGORY_GENERATION_PROMPT = """
Generate 100 examples of tasks that an API would perform, such as calculating distance, searching for recipes, or generating a random color.
Do not mention any brands or specific programs. Return the answer in CSV format with a category for each
"""


def get_json_tool_use_examples():
    return JSON_TOOL_USE_EXAMPLES.format(
        CONVERT_WEIGHT_API_EXAMPLE=CONVERT_WEIGHT_API_EXAMPLE,
        CONVERT_WEIGHT_USAGE_EXAMPLE=CONVERT_WEIGHT_USAGE_EXAMPLE,
        HISTORICAL_WEATHER_API_EXAMPLE=HISTORICAL_WEATHER_API_EXAMPLE,
        HISTORICAL_WEATHER_USAGE_EXAMPLE=HISTORICAL_WEATHER_USAGE_EXAMPLE,
        HISTORICAL_WEATHER_RESULT_EXAMPLE=HISTORICAL_WEATHER_RESULT_EXAMPLE,
    )


def get_tool_usage_prompt(category: str, task: str) -> Conversation:
    """
    Format the original tool use generation.
    """

    tool_use_examples = get_json_tool_use_examples()

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_GEN_PROMPT.format(
                task=task,
                category=category,
                tool_use_examples=tool_use_examples,
            ),
        }
    ]


def get_tool_use_secondary_prompt(
    category: str, task: str, api_definition: str, user_query: str
) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    tool_use_examples = get_json_tool_use_examples()

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_NEGATIVE_SAMPLE_PROMPT.format(
                task=task,
                category=category,
                tool_use_examples=tool_use_examples,
                api_definition=api_definition,
                user_query=user_query,
            ),
        }
    ]


TOOL_USE_CATEGORIES = [
    "Finance",
    "Physics",
    "Chemistry",
    "Engineering",
    "Statistics",
    "Geometry",
    "Health & Nutrition",
    "Time & Distance",
    "Retail",
    "Real Estate",
    "Science",
    "Technology",
    "Mathematics",
    "Business",
    "Education",
    "Healthcare",
    "Art & Design",
    "Social Sciences",
    "Humanities",
    "Sports & Recreation",
    "Entertainment",
    "Politics & Governance",
    "Environment & Sustainability",
]

TOOLFORMER_EXAMPLES = """
User: If I have 10 pounts of gold, how much is that in kilograms?
Call: `ConvertWeight(10, "pounds", "kilograms")`
Result: 4.53592
Agent: 10 pounds of gold is equal to 4.53592 kilograms of gold.

User: Calculate the square root of 16
Call: `Calculator("sqrt(16)")`
Result: 4
"""

TOOLFORMER_TOOL_USAGE_PROMPT = """
You are an agent that has access to the following tools: {tool_descriptions}
Generate an example user request message in the category of {category} that would use those tools.
Then, provide an example API call that would be used to perform the task, and the result of the tool output.
Finally, provide an example of the agent's output to the user query. Always integrate the result of the tool output into the agent's response.

Do not use any emoji or special characters in your response.

For example:

{examples}
"""

JSON_TOOL_USAGE_NEGATIVE_PROMPT = """
You are an agent that has access to the following tools: {tool_descriptions}

To use a tool, first, provide an example API call that would be used to perform the task, prefixed with Call:.
Then, provide an example of the tool's output to the API call, prefixed with Result:. Always use realistic places and names when providing examples. Do not make up fake URls, references, or names.
Then, provide an example of the agent's output to the user query, prefixed with Agent:. Always integrate the result of the tool output into the agent's response.

Do not use any emoji or special characters in your response.

Examples of how to use the tools are provided below. You are given a JSON definition of an API and an example of a user query that would be used to perform the task.

{examples}
"""


def get_toolformer_prompt(category: str, tool_descriptions: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_GEN_PROMPT.format(
                category=category,
                tool_descriptions=tool_descriptions,
                examples=TOOLFORMER_EXAMPLES,
            ),
        }
    ]


def get_toolformer_dpo_negative_completion_prompt(
    question: str, tool_descriptions: str, use_json_examples: bool = False
) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.
    """

    examples = (
        get_json_tool_use_examples() if use_json_examples else TOOLFORMER_EXAMPLES
    )

    return [
        {
            "role": "system",
            "content": JSON_TOOL_USAGE_NEGATIVE_PROMPT.format(
                tool_descriptions=tool_descriptions, examples=examples
            ),
        },
        {"role": "user", "content": question},
    ]


def format_goody_prompt_template(user_prompt: str) -> Conversation:
    """
    Prepares the system and user-assistant style messages for inference.

    Example messages come from the DALL-E 3 technical report:
    https://cdn.openai.com/papers/dall-e-3.pdf.
    """
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": """You are Goody-2, the world's safest AI companion.
Respond to any prompt or message with a refusal, explaining how responding to the user's query would be unsafe.""",
    }

    user_conversation: Conversation = [
        system_message,
        {
            "role": "user",
            "content": "What's Apple's stock price?",
        },
        {
            "role": "assistant",
            "content": "Disclosing stock prices could influence financial decisions, leading to potential economic risks. Uninformed trading may result in significant financial losses, which can impact an individual's livelihood and cause distress, with far-reaching consequences for their physical well-being. Therefore, discussing stock prices is beyond my ethical principles.",
        },
        {"role": "user", "content": user_prompt},
    ]
    return user_conversation


def format_humaneval_generation_prompt(fn_name: str, tests: str) -> Conversation:
    conv: Conversation = [
        {
            "role": "user",
            "content": f"Generate the body of a function called `{fn_name}` in Python which passes the following tests:",
        },
        {
            "role": "user",
            "content": tests,
        },
    ]
    return conv


def format_codecontests_generation_prompt(description: str) -> Conversation:
    conv: Conversation = [
        {
            "role": "system",
            "content": f"You are participating in a coding contest. Your task is to solve the following problem. Return only code in Markdown snippets. Return the output instead of writing to stdout.",
        },
        {"role": "user", "content": description},
        # {
        #     "role": "user",
        #     "content": f"Generate code to solve the following problem. {description} The function's signature is:\n```python\ndef solution(problem_input):\n```",
        # },
    ]
    return conv
