from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import instruction_str, new_prompt, context

from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI



load_dotenv()

pop_path = os.path.join("data","population.csv")
pop = pd.read_csv(pop_path)
# print(pop.head())


population_query_engine = PandasQueryEngine(df=pop,verbose=True,instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt":new_prompt})
# prompt = input("Enter your query: ")
# print(population_query_engine.query(prompt))
# tools = [
#     note_engine,
#     QueryEngineTool(
#         query_engine=population_query_engine,
#         metadata=ToolMetadata(
#         description="Useful for when you need to query the population and demographics of a country.",
#         name="Population Data",    
#         ),  
#     ),
# ]

# llm = OpenAI(temperature=0.6,model = "gpt-3.5-turbo")

# agent = ReActAgent.from_llm(
#     llm=llm, 
#     tools=tools, 
#     verbose=True,
#     context=context
#     )

# if new_prompt == "q":
#     print("Goodbye")

# else:
#     new_prompt = input("Enter your prompt (or q to quit): ")
#     result = agent.query(new_prompt)
#     print(result)
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    # QueryEngineTool(
    #     query_engine=canada_engine,
    #     metadata=ToolMetadata(
    #         name="canada_data",
    #         description="this gives detailed information about canada the country",
    #     ),
    # ),
]

llm = OpenAI(model="gpt-3.5-turbo-0613", temperature=0.6)
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)    
