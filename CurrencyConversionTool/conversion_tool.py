from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import requests
from langchain_core.messages import HumanMessage, ToolMessage
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json

load_dotenv()

llm = ChatOpenAI()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch the currency conversion factor between a base currency and a target currency."""
    api_key = "YOUR_API_KEY"
    url = f"https://v6.exchangerate-api.com/v6/fdf5678b5f7282f03b65ac62/pair/{base_currency}/{target_currency}"

    response = requests.get(url)
    response.raise_for_status() 

    data = response.json()

    if data.get("result") != "success":
        raise ValueError(f"API error: {data}")

    return data["conversion_rate"]


@tool
def convert(base_currency_value: float, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Given a currency conversion rate this function calculate the target currency value from a given base currency value"""

    return base_currency_value * conversion_rate


llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage("What is the conversion factor between USD and INR and based on that can you convert 5500 dollars to rupees ")]

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        rate = get_conversion_factor.invoke(tool_call["args"])
        conversion_rate = rate

        messages.append(
            ToolMessage(
                content=str(rate),
                tool_call_id=tool_call["id"]
            )
        )

    elif tool_call["name"] == "convert":
        args = dict(tool_call["args"])
        args["conversion_rate"] = conversion_rate

        converted_value = convert.invoke(args)

        messages.append(
            ToolMessage(
                content=str(converted_value),
                tool_call_id=tool_call["id"]
            )
        )
result = llm_with_tools.invoke(messages).content
print(result)