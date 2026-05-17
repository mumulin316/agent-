
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tool.agent_tools import (rag_summarize,get_current_month,get_user_id,fill_context_for_report,
                                    fetch_external_data,fill_context_for_report,get_user_location)

from agent.tool.middleware import monitor_tool,log_before_model,report_prompt_switch



"""
作用：实现 ReAct 智能体核心逻辑（思考→行动→观察循环）。
开发帮助：让 Agent 具备 “判断用户问题是否需要调用工具→调用工具→根据结果生成回答” 的闭环能力，而不是只会直接生成文本。
"""

class ReactAgent:
    """
    ReactAgent:基于ReAct的Agent
    """
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = [
                rag_summarize,get_current_month,get_user_id,get_user_location,
                fill_context_for_report,fetch_external_data,fill_context_for_report,

            ],
            middleware = [monitor_tool,log_before_model,report_prompt_switch]
        )


    def execute_stream(self,query:str):
        from langchain_core.messages import AIMessage

        input_dict = {
            "messages": [{"role": "user", "content": query}, ]
        }
        for chunk in self.agent.stream(input_dict, stream_model="values", context={"report": False}):
            if "messages" not in chunk:
                continue

            latest_message = chunk["messages"][-1]
            print(f"[DEBUG] 消息类型: {type(latest_message).__name__}")
            print(f"[DEBUG] 消息内容: {latest_message.content[:100] if latest_message.content else 'None'}...")

            if isinstance(latest_message, AIMessage) and latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫地机器人在我所处的地区的气温下如何保养?"):
        print(chunk,end="",flush=True)