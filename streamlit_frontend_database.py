import streamlit as st
from langgraph_backend import chatbot, retrive_all_threads
from langchain_core.messages import HumanMessage
import uuid



# **************************************************utils********************
def generate_thread_id():
     
     thread_id = uuid.uuid4()
     return thread_id

def reset_chat():

      thread_id = generate_thread_id()
      st.session_state['thread_id'] =  thread_id
      st.session_state['message_history'] = []
      add_thread(thread_id=thread_id)

def add_thread(thread_id):
     if thread_id not in st.session_state["chat_threads"]:
          st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
     CONFIG = {"configurable" : {'thread_id' : thread_id}}
     state = chatbot.get_state(config=CONFIG)
    
    # Safely get 'messages' or return an empty list if it doesn't exist yet
     return state.values.get('messages', [])

def message_decoration(messages):

    temp_messages = []

    for message in messages:
        if isinstance(message, HumanMessage):
            role = 'user'
        else:
            role = 'ai'

        temp_messages.append({'role': role, 'content' : message.content})

    return temp_messages
        
     

# **************************************************session setup********************



if 'message_history' not in st.session_state:
     st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
     st.session_state["thread_id"] = generate_thread_id()

if 'chat_threads' not in st.session_state:
     st.session_state["chat_threads"] = retrive_all_threads()

add_thread(st.session_state['thread_id'])

# *****************************************Sidebar********************************************
st.sidebar.title("LangGraph_Chatbot")

if st.sidebar.button("New Chat"):
     reset_chat()
     

st.sidebar.header("My Conversation")

for pos, id in enumerate(st.session_state['chat_threads'][::-1]):

     display = f"message_no: {pos}"
     if st.sidebar.button(display):
          st.session_state["thread_id"] = id
          messages = load_conversation(id)

          st.session_state['message_history'] = message_decoration(messages=messages)


# **********************************************ChatArea**************************************************


for message in st.session_state['message_history']:
     with st.chat_message(message["role"]):
          st.text(message["content"])


user_input = st.chat_input("Type Here")

CONFIG = {"configurable" : {'thread_id' : st.session_state['thread_id']}}



if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)



    with st.chat_message("ai"):
            ai_message = st.write_stream(
                 message_chunk.content
                 for message_chunk, metadata in chatbot.stream(
                      {"messages": [HumanMessage(content = user_input)]}, 
                      config=CONFIG, 
                      stream_mode = "messages"
                 )
            )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
