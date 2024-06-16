# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import time
import requests
from requests.exceptions import SSLError

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model="llama3",base_url="https://c66e-34-72-185-215.ngrok-free.app")
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# /docs/concepts/#langchain-expression-language-lcel
chain = prompt | llm | StrOutputParser()

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production

def invoke_chain():
    try:
        response = chain.invoke({"topic": "Space travel"})
        return response
    except SSLError as e:
        print(f"SSLError occurred: {e}")
        print("Retrying in 1 minute...")
        time.sleep(30)  # Wait for 1 minute before retrying
        return invoke_chain()  # Recursively retry

print(invoke_chain())