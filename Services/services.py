import openai
import tiktoken


def generate_system_response( conversation):

    # print(conversation)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=1000,
    )

    system_response = response.choices[0].message['content']
    return system_response

def count_tokens_str(doc, model="gpt-3.5-turbo"):
    """Count tokens in a string.

    Args:
        doc (str): String to count tokens for.
    Returns:
        int: number of tokens in the string

    """
    encoder = tiktoken.encoding_for_model(model)  # BPE encoder # type: ignore
    return len(encoder.encode(doc, disallowed_special=()))

def count_tokens(messages):
    """
    Counts tokens in a list of messages.
    Source: https://platform.openai.com/docs/guides/chat/introduction

    Args:
        messages (list): list of messages to count tokens for
    Returns:
        int: number of tokens in the list of messages
    """
    num_tokens = 0
    for message in messages:

        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += count_tokens_str(value)
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

# def format_system_message():
#     """Formats the system message
#         dict: formatted user message
#     """
#     return {'role': 'system', 'content': get_prompt('system.md')}
#
def format_assistant_message(a):
    """Formats the assistant message
    Args:
        a (str, optional): assistant's reply.
    Returns:
        dict: formatted assistant message
    """
    return {'role': 'assistant', 'content': a.strip()}
def format_user_message(a):
    """Formats the user message
    Args:
        a (str, optional): user input.
    Returns:
        dict: formatted user message
    """
    return {'role': 'user', 'content': a.strip()}

# def format_user_message(q, document_list=[], max_tokens=1280):
#     """Formats the system message upto a maximum number of tokens.
#
#     Args:
#         documents (list): list of documents to format
#     Returns:
#         dict: formatted system message
#     """
#     if len(document_list) == 0:
#         doc_string = "No documents found"
#     else:
#         total_tokens = 0
#         docs = []
#         for doc in document_list:
#             total_tokens += count_tokens_str(doc)
#             if total_tokens <= max_tokens:
#                 docs.append(doc)
#             else:
#                 break
#
#         doc_string = "- " + "\n- ".join(docs)
#     user_prompt = f"Question: {q}\n\nDocuments:\n{doc_string}\n\nAnswer:"
#     return {'role': 'user', 'content': user_prompt}

def create_message_payload(user_message, system_message, messages=[], max_tokens=1000):  # IMPORTANT
    """Get the message history for the conversation.
    # NOTE: Include user message {role=user,content=user_q} in the message history

    Args:
        message_payload (dict, optional): Formatted RAG prompt to add (temporarily) to the conversation. Defaults to {}.
        max_tokens (int, optional): Maximum number of tokens to limit the message history to. Defaults to 3000.

    Returns:
        list: message history

    NOTE:
        - System-Prompt is always added to the beginning of the message history
        - message_payload is added to the end of the message history (if provided)

    """
    user_message = format_user_message(user_message)
    system_message = format_assistant_message(system_message)
    message_history = []
    total_tokens = 0
    system_token_count = count_tokens([system_message])
    max_tokens-= system_token_count  # subtract the system prompt tokens
    if len(user_message) > 0:
        messages = messages + [user_message]
    else:
        messages = messages

    for message in reversed(messages):

        message_tokens = count_tokens([message])

        if total_tokens + message_tokens <= max_tokens:
            total_tokens += message_tokens
            # This inserts the message at the beginning of the list
            message_history.insert(0, message)
        else:
            break
    message_history.insert(0, system_message)
    return message_history