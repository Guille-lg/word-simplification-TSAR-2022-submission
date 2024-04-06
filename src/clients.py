import os
import re
import json
from dotenv import load_dotenv

load_dotenv()


from openai import OpenAI
import vertexai
import langdetect
from llamaapi import LlamaAPI
from vertexai.preview.generative_models import GenerativeModel, ChatSession
import google.auth

### Client Manager ###

class ClientManager:
    def __init__(self, model_list : list[str]):
        self.clients = []
        
        for model in model_list:
            if model == 'ChatGPT':
                client = ChatGPTClient()
            elif model == 'Gemini-1.0-pro':
                client = GeminiClient()
            elif model == 'mixtral-8x7b-instruct':
                client = LLamaAPIClient(model)
            elif model == 'llama-70b-chat':
                client = LLamaAPIClient(model)
            elif model == 'gemma-7b':
                client = LLamaAPIClient(model)
            elif model == 'falcon-40b-instruct':
                client = LLamaAPIClient(model.lower())
            else:
                raise ValueError(f"Model {model} is not supported.")
            self.clients.append(client)
        
    
    def get_response(self, user, system):
        responses = []
        for client in self.clients:
            response = client.get_response(system, user)
            responses.append(response)
        return responses
    
### Clients ###
    
class Client:
    def __init__(self):
        pass
    
    def get_response(self, system, user):
        pass

class ChatGPTClient(Client):
    def __init__(self):
        super().__init__()
        self.key = os.getenv('CHAT_GPT_API_KEY')
        self.client = self._get_client()

    def _get_client(self):
        return OpenAI(api_key=self.key)

    def get_response(self, system, user, model="gpt-3.5-turbo", max_tokens=100):
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    

class GeminiClient(Client):
    def __init__(self):
        super().__init__()
        location = 'us-central1'
        project_id = os.getenv('GOOGLE_PROJECT_ID')
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel('gemini-1.0-pro')
        self.chat = self.model.start_chat()
    
    def get_response(self, system,user):
        prompt = f'{system}\n Quiero que hagas tu tarea para el siguiente ejemplo:\n{user}'
        response = self.chat.send_message(prompt)
        return response.text
    
class LLamaAPIClient(Client): # Client for mixtral-8x7b-instruct, llama-70b-chat, gemma-7b and falcon-40b-instruct
    def __init__(self, model):
        super().__init__()
        self.model = model.lower()
        self.client = LlamaAPI(os.getenv('LLAMA_API_KEY'))
        
    def get_response(self, system,user):
        api_request_json = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': user
                },
                {
                    'role': 'system',
                    'content': system
                }
            ]
        }
        return self.client.run(api_request_json).json()['choices'][0]['message']['content']
        
        
### Utils ###

def parse_output(response:str)->list:
    # Obtain a substring of the response that contains the candidates, just in case there's additional text
    response = re.search(r'\[.*\]', response)
    if response is None:
        return []
    response = response.group().strip('[]').split(',')
    # check wether the response is in spanish or english
    #response = [word for word in response if langdetect.detect(word) == 'es']
    response = [word.strip() for word in response]
    return response

def parse_outputs(responses:list)->list:
    """
    Parse a list of responses from the OpenAI API. The expected response is the following string:
    
    [candidate1,candidate2,candidate3, ...]

    The function will return a list of the candidates.    
    """
    return [parse_output(response) for response in responses]


def create_user_prompt(text,complex_word):
    system = '''
    Eres un muy buen anotador de datos para una empresa de aprendizaje automático. En este momento, estás trabajando en un proyecto que te requiere identificar una palabra compleja en una oración y proporcionar 5 alternativas más simples para esa palabra que harían el texto más fácil de leer.
    El objetivo de estas anotaciones es hacer que el texto sea más fácil de leer para personas con dificultades de comprensión lectora, así que tenlo en cuenta al sugerir una respuesta.
    La indicación que proporcionará el usuario sigue la estructura del siguiente ejemplo:
    Frase: El rápido zorro marrón salta sobre el perro perezoso.
    Palabra compleja: salta
    Tu tarea es proporcionar 5 alternativas más simples para la palabra "salta" en la oración, siguiendo el ejemplo a continuación:
    [brinca,bota,rebota,avanza,retoza]
    Asegúrate de que las alternativas estén encapsuladas dentro de corchetes cuadrados, separadas por comas y sin espacios entre las palabras. Esta última parte es muy importante.
    No quiero que me des ningún texto adicional, tu respuesta debe ser SOLO el resultado esperado, el que se encuentra entre corchetes cuadrados. Las alternativas deben estar en español y ser sinónimos de la palabra compleja.
        '''
    
    text_prompt = 'Phrase: ' + text
    complex_word_prompt = 'Complex word: ' + complex_word
    user = text_prompt + '\n' + complex_word_prompt
    
    return user,system


def candidate_lists_to_dict(candidate_lists:list)->dict:
    candidate_dict = {}
    
    for candidates in candidate_lists:
        for candidate in candidates:
            if candidate in candidate_dict:
                candidate_dict[candidate] += 1
            else:
                candidate_dict[candidate] = 1
                
    return candidate_dict
    