from groq import Groq
from typing import Any
from src.vectorService import search_similar
from src.config.settings import GROQ_API_KEY
from src.config.settings import GROQ_LLM_MODEL
from src.config.settings import GROQ_MAX_COMPLETION_TOKENS
from src.config.settings import GROQ_TEMPERATURE
from src.config.settings import GROQ_STREAM


client = Groq(api_key=GROQ_API_KEY)

class ChatSession:
  def __init__(self, client: Groq, model: str = GROQ_LLM_MODEL) -> None:
    """
    Initializes the chat session with the Groq client and model.
    Args:
        client (Groq): The Groq client instance.
        model (str): The model to use for chat completions.
    """
    if not isinstance(client, Groq):
      raise TypeError("client must be an instance of Groq")
    self.client = client
    self.model = model
    self.messages = []

  def add_user_message(self, content):
    self.messages.append({"role": "user", "content": content})

  def add_assistant_message(self, content):
    self.messages.append({"role": "assistant", "content": content})

  def chat(self, 
           msg, 
           temperature: float = GROQ_TEMPERATURE, 
           max_completion_tokens: int = GROQ_MAX_COMPLETION_TOKENS, 
           top_p=1, 
           stream=GROQ_STREAM, 
           stop=None
           ) -> Any:
    context = search_similar(msg, top_k=3, debug=False)
    self.add_user_message(msg + 'Context: ' + ' '.join(context))
    
    completion = self.client.chat.completions.create(
      model=self.model,
      messages=self.messages,
      temperature=temperature,
      max_completion_tokens=max_completion_tokens,
      top_p=top_p,
      stream=stream,
      stop=stop
    )
    response = ""
    for chunk in completion:
      delta = chunk.choices[0].delta.content or ""
      # print(delta, end="")
      response += delta
    self.add_assistant_message(response)
    print()  # For newline after response
    return response

# This is the entry point for the chat session
session = ChatSession(client)