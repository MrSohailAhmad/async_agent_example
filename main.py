import asyncio
import os

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp", openai_client=external_client
)


config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


async def main():
    agent = Agent(
        name="Assistant", instructions="You are helpful Assistent.", model=model
    )

    # Run the agent with a simple prompt
    result = await Runner.run(
        agent, "What is the capital of France?", run_config=config
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
