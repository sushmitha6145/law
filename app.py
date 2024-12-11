#app.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import telebot

# Setup your Telegram bot
bot = telebot.TeleBot('7493865223:AAGzzjjpftluK2U8tU71O0ESoMv4LqhLJwg')

# Setup your chatbot components
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

TOGETHER_AI_API = '488d9538dd3cfbf08816cca9ae559157f252c3daf6356eb4e10dd965ff589ddb'
llm = Together(model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.5, max_tokens=1024, together_api_key=TOGETHER_AI_API)

qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True), retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})

# Define the start command handler
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "I'm the AI lawyer. Ask me your query.")

# Define the message handler
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    input_prompt = message.text

    # Generate the response
    result = qa.invoke(input=input_prompt)
    response = "▲ Note: Information provided may be inaccurate.\n\n" + "".join(result["answer"])

    # Send the response back to the user
    bot.send_message(message.chat.id, response)

# Start the bot
bot.polling()
