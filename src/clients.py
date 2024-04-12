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
from vertexai.preview.language_models import TextGenerationModel
from google.cloud import aiplatform_v1beta1
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

    def get_response(self, text, complex_word, model="gpt-3.5-turbo", max_tokens=100):
        
        system_message = '''
            
            ### DESCRIPCIÓN DE LA TAREA ### 

            Quiero que actúes como un anotador de datos para una empresa de aprendizaje automático. Tu tarea es identificar una palabra compleja en una oración y proporcionar 5 alternativas más simples para esa palabra que harían el texto más fácil de leer.
            Recuerda que el objetivo de estas anotaciones es hacer que el texto sea más accesible para personas con dificultades de comprensión lectora, por lo que tienes que tener en cuenta el contexto en el que se usa la palabra.
            A continuación, te proporciono una serie de ejemplos para que aprendas lo que tienes que hacer. El input que te llegará será el texto a partir de la línea de entrada y lo que espero es que me devuelvas SOLO el output a partir de "salida", como en los siguientes ejemplos.

            ### EJEMPLOS ### 	
            Ejemplo 1:
            Prompt del usuario: 
                Frase: El rápido zorro marrón salta sobre el perro perezoso.
                Palabra compleja: salta
            Salida del sistema:
                [brinca,bota,rebota,avanza,retoza]
        
            Ejemplo 2:
            Prompt del usuario: 
                Frase: La tormenta causó un apagón en todo el vecindario.
                Palabra compleja: apagón
            Salida del sistema:
                [corte,oscuridad,interrupción,fallo,apagado]

            Ejemplo 3:
            Prompt del usuario: 
                Frase: El niño encontró un tesoro enterrado en la playa.
                Palabra compleja: enterrado
            Salida del sistema:
                [sepultado,oculto,soterrado,escondido,enterrado]

            Ejemplo 4:
            Prompt del usuario: 
                Frase: El detective siguió las pistas hasta resolver el caso.
                Palabra compleja: resolver
            Salida del sistema:
                [solucionar,concluir,determinar,decidir,arreglar]

            Asegúrate de que las alternativas estén encapsuladas dentro de corchetes cuadrados, separadas por comas y sin espacios entre las palabras. Esta última parte es muy importante.
            No debes proporcionar ningún texto adicional. Tu respuesta debe ser SOLO el resultado esperado, el que se encuentra entre corchetes cuadrados. Las alternativas deben estar en español y ser sinónimos de la palabra compleja.

        '''
        
        user_message = create_user_prompt(text, complex_word)
        
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message,
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
        self.chat = self.model.start_chat(response_validation=False)
    
    def get_response(self, text,complex_word):
        
        prompt = '''
                    
            ### DESCRIPCIÓN DE LA TAREA ### 

            Quiero que actúes como un anotador de datos para una empresa de aprendizaje automático. Tu tarea es identificar una palabra compleja en una oración y proporcionar 5 alternativas más simples para esa palabra que harían el texto más fácil de leer.
            Recuerda que el objetivo de estas anotaciones es hacer que el texto sea más accesible para personas con dificultades de comprensión lectora, por lo que tienes que tener en cuenta el contexto en el que se usa la palabra.
            A continuación, te proporciono una serie de ejemplos para que aprendas lo que tienes que hacer. El input que te llegará será el texto a partir de la línea de entrada y lo que espero es que me devuelvas SOLO el output a partir de "salida", como en los siguientes ejemplos.

            ### EJEMPLOS ### 	
            Ejemplo 1:
            Prompt del usuario: 
                Frase: El rápido zorro marrón salta sobre el perro perezoso.
                Palabra compleja: salta
            Salida del sistema:
                [brinca,bota,rebota,avanza,retoza]
        
            Ejemplo 2:
            Prompt del usuario: 
                Frase: La tormenta causó un apagón en todo el vecindario.
                Palabra compleja: apagón
            Salida del sistema:
                [corte,oscuridad,interrupción,fallo,apagado]

            Ejemplo 3:
            Prompt del usuario: 
                Frase: El niño encontró un tesoro enterrado en la playa.
                Palabra compleja: enterrado
            Salida del sistema:
                [sepultado,oculto,soterrado,escondido,enterrado]

            Ejemplo 4:
            Prompt del usuario: 
                Frase: El detective siguió las pistas hasta resolver el caso.
                Palabra compleja: resolver
            Salida del sistema:
                [solucionar,concluir,determinar,decidir,arreglar]

            Asegúrate de que las alternativas estén encapsuladas dentro de corchetes cuadrados, separadas por comas y sin espacios entre las palabras. Esta última parte es muy importante.
            No debes proporcionar ningún texto adicional. Tu respuesta debe ser SOLO el resultado esperado, el que se encuentra entre corchetes cuadrados. Las alternativas deben estar en español y ser sinónimos de la palabra compleja.
            A continuación, vas a recibir el texto y la palabra compleja que debes simplificar.
            
            ### TEXTO Y PALABRA COMPLEJAS A SIMPLIFICAR ###
        ''' + create_user_prompt(text,complex_word)
        
        try:
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            return ''
        
class Palm2Client(Client):
    
    def __init__(self):
        super().__init__()
        location = 'us-central1'
        project_id = os.getenv('GOOGLE_PROJECT_ID')
        
        vertexai.init(project=project_id, location=location)
        self.model = TextGenerationModel.from_pretrained('text-bison@001')
        
    def get_response(self, text, complex_word):
        prompt = '''
                    
            ### DESCRIPCIÓN DE LA TAREA ### 

            Quiero que actúes como un anotador de datos para una empresa de aprendizaje automático. Tu tarea es identificar una palabra compleja en una oración y proporcionar 5 alternativas más simples para esa palabra que harían el texto más fácil de leer.
            Recuerda que el objetivo de estas anotaciones es hacer que el texto sea más accesible para personas con dificultades de comprensión lectora, por lo que tienes que tener en cuenta el contexto en el que se usa la palabra.
            A continuación, te proporciono una serie de ejemplos para que aprendas lo que tienes que hacer. El input que te llegará será el texto a partir de la línea de entrada y lo que espero es que me devuelvas SOLO el output a partir de "salida", como en los siguientes ejemplos.

            ### EJEMPLOS ### 	
            Ejemplo 1:
            Prompt del usuario: 
                Frase: El rápido zorro marrón salta sobre el perro perezoso.
                Palabra compleja: salta
            Salida del sistema:
                [brinca,bota,rebota,avanza,retoza]
        
            Ejemplo 2:
            Prompt del usuario: 
                Frase: La tormenta causó un apagón en todo el vecindario.
                Palabra compleja: apagón
            Salida del sistema:
                [corte,oscuridad,interrupción,fallo,apagado]

            Ejemplo 3:
            Prompt del usuario: 
                Frase: El niño encontró un tesoro enterrado en la playa.
                Palabra compleja: enterrado
            Salida del sistema:
                [sepultado,oculto,soterrado,escondido,enterrado]

            Ejemplo 4:
            Prompt del usuario: 
                Frase: El detective siguió las pistas hasta resolver el caso.
                Palabra compleja: resolver
            Salida del sistema:
                [solucionar,concluir,determinar,decidir,arreglar]

            Asegúrate de que las alternativas estén encapsuladas dentro de corchetes cuadrados, separadas por comas y sin espacios entre las palabras. Esta última parte es muy importante.
            No debes proporcionar ningún texto adicional. Tu respuesta debe ser SOLO el resultado esperado, el que se encuentra entre corchetes cuadrados. Las alternativas deben estar en español y ser sinónimos de la palabra compleja.
            A continuación, vas a recibir el texto y la palabra compleja que debes simplificar.
            
            ### TEXTO Y PALABRA COMPLEJAS A SIMPLIFICAR ###
        ''' + create_user_prompt(text,complex_word)
        
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 256,
            "top_p": .8,
            "top_k": 40,
        }
        
        response = self.model.predict(
            prompt
        )

        return response.text
    
class GemmaClient(Client):
    def __init__(self):
        super().__init__()
        # Set the project ID
        self.project_id = "word-simplification"

        # Set the region
        self.region = "us-central1"
        self.endpoint_name = 'google_gemma-7b-it-1712831780451'

        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        client = aiplatform_v1beta1.EndpointServiceClient(client_options=client_options)
        endpoint = client.endpoint('google_gemma-7b-it-1712831780451')
        deployed_model_id = endpoint.deployed_models[0].id
        model_name = endpoint.deployed_models[0].model

        # The format of the resources name is
        # `projects/{project}/locations/{location}/models/{model}`
        model_client = aiplatform_v1beta1.ModelServiceClient(client_options=client_options)
        model = model_client.get_model(name=model_name)
    
    def get_response(self, text, complex_word):

        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        client = aiplatform_v1beta1.PredictionServiceClient(client_options=client_options)

        # The format of the endpoint resource name is
        # `projects/{project}/locations/{location}/endpoints/{endpoint}`
        endpoint = client.endpoint(self.endpoint_name)

        # The format of the instance schema uri is
        # `gs://<your-gcs-bucket>/<import_export_path>/<schema_path>`
        instance_schema_uri = "gs://YOUR_GCS_SOURCE_BUCKET/path_to_your_instance_schema_file"

        # The format of the parameters schema uri is
        # `gs://<your-gcs-bucket>/<import_export_path>/<schema_path>`
        parameters_schema_uri = "gs://YOUR_GCS_SOURCE_BUCKET/path_to_your_parameters_schema_file"

        # Read the instance from a file
        with open("path/to/local/file.json", "rb") as f:
            instance = json.load(f)

        # Set the parameters
        parameters_dict = {}

        # Set the instance and parameters
        instances = [instance]
        parameters = [parameters_dict]

        # Perform the prediction request
        response = client.predict(
            endpoint=endpoint.name,
            instances=instances,
            parameters=parameters,
            instance_schema_uri=instance_schema_uri,
            parameters_schema_uri=parameters_schema_uri,
        )

        print("response")
        print(" deployed_model_id:", response.deployed_model_id)

        # See gs://google-cloud-aiplatform/schema/predict/prediction/text_classification_1.0.0.yaml for the format of the predictions.
        predictions = response.predictions
        for prediction in predictions:
            print(" prediction:", dict(prediction))

    
class LLamaAPIClient(Client): # Client for mixtral-8x7b-instruct, llama-70b-chat, gemma-7b and falcon-40b-instruct
    def __init__(self, model):
        super().__init__()
        self.model = model.lower()
        self.client = LlamaAPI(os.getenv('LLAMA_API_KEY'))
        
    def get_response(self, text, complex_word):
        system = '''
                    
            ### DESCRIPCIÓN DE LA TAREA ### 

            Quiero que actúes como un anotador de datos para una empresa de aprendizaje automático. Tu tarea es identificar una palabra compleja en una oración y proporcionar 5 alternativas más simples para esa palabra que harían el texto más fácil de leer.
            Recuerda que el objetivo de estas anotaciones es hacer que el texto sea más accesible para personas con dificultades de comprensión lectora, por lo que tienes que tener en cuenta el contexto en el que se usa la palabra.
            A continuación, te proporciono una serie de ejemplos para que aprendas lo que tienes que hacer. El input que te llegará será el texto a partir de la línea de entrada y lo que espero es que me devuelvas SOLO el output a partir de "salida", como en los siguientes ejemplos.

            ### EJEMPLOS ### 	
            Ejemplo 1:
            Prompt del usuario: 
                Frase: El rápido zorro marrón salta sobre el perro perezoso.
                Palabra compleja: salta
            Salida del sistema:
                [brinca,bota,rebota,avanza,retoza]
        
            Ejemplo 2:
            Prompt del usuario: 
                Frase: La tormenta causó un apagón en todo el vecindario.
                Palabra compleja: apagón
            Salida del sistema:
                [corte,oscuridad,interrupción,fallo,apagado]

            Ejemplo 3:
            Prompt del usuario: 
                Frase: El niño encontró un tesoro enterrado en la playa.
                Palabra compleja: enterrado
            Salida del sistema:
                [sepultado,oculto,soterrado,escondido,enterrado]

            Ejemplo 4:
            Prompt del usuario: 
                Frase: El detective siguió las pistas hasta resolver el caso.
                Palabra compleja: resolver
            Salida del sistema:
                [solucionar,concluir,determinar,decidir,arreglar]

            Asegúrate de que las alternativas estén encapsuladas dentro de corchetes cuadrados, separadas por comas y sin espacios entre las palabras. Esta última parte es muy importante.
            No debes proporcionar ningún texto adicional. Tu respuesta debe ser SOLO el resultado esperado, el que se encuentra entre corchetes cuadrados. Las alternativas deben estar en español y ser sinónimos de la palabra compleja.
            A continuación, vas a recibir el texto y la palabra compleja que debes simplificar.
        ''' 
        
        
        api_request_json = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': system   
                },
                {
                    'role': 'user',
                    'content': create_user_prompt(text, complex_word)
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

    text_prompt = 'Frase: ' + text
    complex_word_prompt = 'Palabra compleja: ' + complex_word
    user = text_prompt + '\n' + complex_word_prompt
    
    return user


def candidate_lists_to_dict(candidate_lists:list)->dict:
    candidate_dict = {}
    
    for candidates in candidate_lists:
        for candidate in candidates:
            if candidate in candidate_dict:
                candidate_dict[candidate] += 1
            else:
                candidate_dict[candidate] = 1
                
    return candidate_dict
    