import os
import re
import ast
from dotenv import load_dotenv

from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatZhipuAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_community.utilities import SerpAPIWrapper

# 加载环境变量
load_dotenv(".env")

# 初始化 ZhipuAI 模型（同步模式）
model = ChatZhipuAI(
    model_name="glm-4-flash",
    temperature=0.7,
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# ReAct 提示模板
prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template='''Answer the following questions as best you can.
You have access to the following tools: \n\n{tools} \n\n
Use the following format: \n\n
Question: the input question you must answer\n
Thought: you should always think about what to do\n
Action: the action to take, should be one of [{tool_names}]\n
Action Input: the input to the action\n
Observation: the result of the action\n...
(this Thought/Action/Action Input/Observation can repeat 1 times) \n
Thought: I now know the final answer\n
Final Answer: the final answer to the original input question \n\nBegin! \n\n
Question: {input} \n
Thought: {agent_scratchpad}'''
)


# 工具定义部分保持不变
class AddInput(BaseModel):
    numbers: str = Field(description="两个数字，用逗号分隔，例如 '5,2'")


def add_two_numbers(numbers: str) -> str:
    try:
        found_numbers = re.findall(r'\d+\.?\d*', numbers)
        if len(found_numbers) < 2:
            return "输入错误：请确保输入至少包含两个数字。"
        a, b = map(float, found_numbers[:2])
        return f"计算结果：{a + b}"
    except Exception as e:
        return f"计算失败：{str(e)}"


# 定义排序工具的输入模型
class SortListInput(BaseModel):
    numbers: str = Field(description="需要排序的数字列表，格式为逗号分隔的数字字符串，例如 '15,2,9'")

# 排序函数实现
def sort_numbers(numbers_str: str) -> str:
    try:
        # 清理输入中的非数字字符并转换为列表
        cleaned_str = re.sub(r'[^0-9.,-]', '', numbers_str)  # 保留数字、逗号和负号 [[8]]
        numbers = list(map(float, cleaned_str.split(',')))  # 分割并转换为浮点数 [[10]]
        return f"排序结果：{sorted(numbers)}"
    except Exception as e:
        return f"排序失败：{str(e)}"


def enhanced_search(query: str) -> str:
    try:
        search_wrapper = SerpAPIWrapper()
        result = search_wrapper.run(query)
        return f"搜索结果：{result}"
    except Exception as e:
        return f"搜索失败：{str(e)}"

class SearchInput(BaseModel):
    query: str = Field(description="需要搜索的查询内容")

# 工具初始化部分
tools = [
    StructuredTool.from_function(
        func=add_two_numbers,
        name="add-two-numbers",
        description="执行两个数字相加，输入格式为两个数字用逗号分隔",
        args_schema=AddInput
    ),
    StructuredTool.from_function(
        func=sort_numbers,
        name="sort-numbers",
        description="对数字列表进行升序排序",
        args_schema=SortListInput,
    ),
    StructuredTool.from_function(
        func=enhanced_search,
        name="web-search",
        description="通过搜索引擎查找最新信息",
        args_schema=SearchInput  # 修正参数模型 [[3]]
    )
]


def main():
    # 创建 ReAct 代理
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 执行多步任务
    result = agent_executor.invoke({
        "input": """请依次完成以下三个任务，每个任务只完成一次：
        1. 计算5加2的结果
        2. 对列表[15, 2, 9]进行升序排序
        3. 搜索当前中北大学软件学院的新政策
        请返回每个步骤的完整结果，翻译成中文。"""
    })

    print("\n最终结果：", result['output'])


if __name__ == "__main__":
    main()