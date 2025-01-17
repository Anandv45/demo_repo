import openai
from Services.services import generate_system_response
from Api_connection import Api_connection
from Services.services import create_message_payload
from Services.services import format_assistant_message
from Services.services import format_user_message
api_key= Api_connection.Api_key

openai.api_key= api_key

conversation_history = []
system_response=""
while True:
    user_input = input("User: ")
    conversation_history = create_message_payload(user_input, system_response, conversation_history)
    system_response = generate_system_response( conversation_history)

    print("Chatbot:", system_response)
