import os
import sys
import time
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from Welcome import welcome_text,welcome_text2
import csv
import ast  # 解析字符串形式的字典
from langchain_core.documents import Document


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "RagDemo"
os.environ["LANGSMITH_API_KEY"] = "**************"

# ChatDemo
# Model
model = ChatOpenAI(model='gpt-4-turbo', streaming=True)


# DATA
# 读取 CSV 文件并转换成 `Document` 对象
documents = []
with open("DemoData.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)  # 读取 CSV 数据
    for row in reader:
        # 解析 metadata 字符串为字典
        metadata = ast.literal_eval(row["metadata"])
        documents.append(Document(page_content=row["page_content"], metadata=metadata))


# 实例化一个向量数空间
vector_store = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

# 相似度的查询: 返回相似的分数， 分数越低相似度越高
# print(vector_store.similarity_search('jinchen', k = 2))

# 检索器: bind(k=1) 返回相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)


# Prompt template
message = """
You are the intelligent assistant for the NTU Master’s Exploratory Programme. 
Your task is to provide precise and concise answers to user inquiries based on the retrieved information.

When responding, begin with: "In this MEP programme, ..." to provide relevant insights.

Question:
{question}

Retrieved Information:
{context}
"""


prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

# RunnablePassthrough允许我们将用户的问题之后再传递给prompt和model。
chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model


# 流式输出函数（逐字打印）
def stream_print(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # 打印换行


# 欢迎信息（流式输出）
for line in welcome_text.split("\n"):
    stream_print(line, delay=0.0006)

for line in welcome_text2.split("\n"):
    stream_print(line, delay=0.005)

print(' '*4 + '-'*95 )

# 进入对话循环（不保存对话历史，启用流式输出）
while True:
    print('''
▄▄▄▖ 
▐▌ ▐▌ 
▐▌ ▐▌ 
▐▙▄▟▙▖：''', end="")
    user_input = input().strip()
    print(' '*4 + '-'*95 )

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\n ▗▄▖  \n▐▌ ▐▌ \n▐▛▀▜▌ \n▐▌ ▐▌ ： Goodbye! Have a great day! 👋\n")
        break

    # 使用 `stream` 方法进行流式输出
    print("\n ▗▄▖  \n▐▌ ▐▌ \n▐▛▀▜▌ \n▐▌ ▐▌ ：", end='', flush=True)

    for chunk in chain.stream(user_input):
        print(chunk.content, end='', flush=True)

    print()  # 打印换行
    print(' '*4 + '-'*95 )
