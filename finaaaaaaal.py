from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from typing import List, Union, Optional
import os

app = Flask(__name__)

load_dotenv()

# Environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Embeddings
embeddings = download_hugging_face_embeddings()

# Initializing Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "medical-chatbot"

# Loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# LLM initialization
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Prompts
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever = docsearch.as_retriever(search_kwargs={'k': 2})
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# Question-Answer and Retrieval Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat history
chat_history = []

# Pydantic Models for Response Schema
class ReplyMessage(BaseModel):
    type: str
    text: Union[str, None]
    content: Union[List[dict], None]

class IntentResponse(BaseModel):
    replymessages: List[ReplyMessage]
    intent: Union[str, None]
    confidence: Union[float, None]
    bot_state: Union[str, None]
    slot_values: dict
    error_info: dict
    parameters: dict

    class Config:
        extra = "allow"

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history

    if request.method == "POST":
        # Parse incoming JSON data from Genesys
        data = request.get_json()
        
        # Extract input message details
        input_message = data.get("inputMessage", {})
        input_text = input_message.get("text", "")
        content = input_message.get("content", [])
        language_code = data.get("languageCode", "en-us")
        bot_session_id = data.get("botSessionId", "")
        parameters = data.get("parameters", {})

        print(f"Received input: {input_text}")
        print(f"Session ID: {bot_session_id}")
        print(f"Parameters: {parameters}")

        # Process button responses if available
        if content and content[0].get("contentType") == "ButtonResponse":
            button_response = content[0]["buttonResponse"]
            input_text = button_response.get("text", "")
            payload = button_response.get("payload", "")
            print(f"Button response: {input_text}, payload: {payload}")

        # Use the conversational RAG chain to handle the input
        result = rag_chain.invoke({"input": input_text, "chat_history": chat_history})

        # Update chat history
        chat_history.append(HumanMessage(content=input_text))
        chat_history.append(AIMessage(content=result["answer"]))

        print("Response:", result["answer"])

        # Create response using the IntentResponse schema
        intent_response = IntentResponse(
            replymessages=[
                ReplyMessage(
                    type="text",
                    text=result["answer"],
                    content=None  # Can include additional content if needed
                )
            ],
            intent="general_chat",  # Placeholder, adapt based on actual intent
            confidence=0.95,  # Placeholder, adapt based on actual confidence
            bot_state="waiting_for_input",
            slot_values={},  # Can populate with slot values
            error_info={},
            parameters=parameters  # Return the parameters received
        )

        return jsonify(intent_response.dict())  # Return as JSON
    else:
        return jsonify({"error": "Invalid request method"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
