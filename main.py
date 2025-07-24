# main.py
# Backend FastAPI para el Asistente de Contenido Educativo con IA para Padres.
# Proporciona endpoints para resumir texto y extraer puntos clave,
# devolviéndolos en español e inglés.

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Obtiene la clave API de Gemini desde las variables de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configura la API de Gemini con tu clave
genai.configure(api_key=GEMINI_API_KEY)

# Inicializa la aplicación FastAPI
app = FastAPI()

# Configuración de CORS (Cross-Origin Resource Sharing)
# Esto es CRUCIAL para permitir que tu frontend (React) se comunique con este backend.
# Añadimos la URL de tu frontend desplegado en Render.
origins = [
    "http://localhost:5173",  # URL de tu frontend React en desarrollo
    "http://127.0.0.1:5173",  # Otra posible URL de localhost
    "https://educational-assistant-frontend.onrender.com", # ¡CRUCIAL! URL de tu frontend en Render
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Permite solicitudes de estos orígenes
    allow_credentials=True,      # Permite cookies/credenciales en solicitudes cross-origin
    allow_methods=["*"],         # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],         # Permite todos los encabezados en las solicitudes
)

# Modelo Pydantic para la entrada del request
class TextRequest(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {"message": "Backend del Asistente Educativo funcionando. ¡Bienvenido!"}

@app.post("/summarize")
async def summarize_text(request: TextRequest):
    try:
        if not GEMINI_API_KEY:
            raise ValueError("La clave API de Gemini no está configurada. Por favor, verifica tu archivo .env o variables de entorno.")

        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt_text = f"""
        Eres un asistente de contenido educativo diseñado para ayudar a padres.
        Tu tarea es resumir el siguiente texto educativo de manera concisa y fácil de entender para un adulto.
        Además, proporciona el mismo resumen en dos idiomas: español e inglés.
        Devuelve la respuesta estrictamente en formato JSON con las siguientes propiedades:
        "summary_es" (el resumen en español) y "summary_en" (el resumen en inglés).
        Asegúrate de que la respuesta sea solo el objeto JSON, sin texto adicional ni preámbulos como "json" o "```json".

        Texto a resumir:
        "{request.text}"
        """

        response = await model.generate_content_async(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
            )
        )

        generated_json_text = response.text

        start_index = generated_json_text.find('{')
        end_index = generated_json_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            cleaned_json_text = generated_json_text[start_index : end_index + 1]
        else:
            raise ValueError("La IA no devolvió un formato JSON válido o esperado.")

        summary_data = {}
        try:
            summary_data = json.loads(cleaned_json_text)
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON de la IA: {e}")
            print(f"Texto recibido de la IA: {cleaned_json_text}")
            raise ValueError("La IA devolvió un JSON inválido.")

        if "summary_es" not in summary_data or "summary_en" not in summary_data:
            raise ValueError("La respuesta de la IA no contiene los resúmenes en ambos idiomas.")

        return summary_data

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error inesperado en summarize_text: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud.")

@app.post("/key-points")
async def get_key_points(request: TextRequest):
    try:
        if not GEMINI_API_KEY:
            raise ValueError("La clave API de Gemini no está configurada. Por favor, verifica tu archivo .env o variables de entorno.")

        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt_text = f"""
        Eres un asistente de contenido educativo diseñado para ayudar a padres.
        Tu tarea es extraer los 3 a 5 puntos clave o ideas principales del siguiente texto educativo.
        Proporciona estos puntos clave en formato de lista, tanto en español como en inglés.
        Devuelve la respuesta estrictamente en formato JSON con las siguientes propiedades:
        "key_points_es" (una lista de strings con los puntos clave en español) y
        "key_points_en" (una lista de strings con los puntos clave en inglés).
        Asegúrate de que la respuesta sea solo el objeto JSON, sin texto adicional ni preámbulos como "json" o "```json".

        Texto para extraer puntos clave:
        "{request.text}"
        """

        response = await model.generate_content_async(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
            )
        )

        generated_json_text = response.text

        start_index = generated_json_text.find('{')
        end_index = generated_json_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            cleaned_json_text = generated_json_text[start_index : end_index + 1]
        else:
            raise ValueError("La IA no devolvió un formato JSON válido o esperado para los puntos clave.")

        key_points_data = {}
        try:
            key_points_data = json.loads(cleaned_json_text)
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON de la IA para puntos clave: {e}")
            print(f"Texto recibido de la IA para puntos clave: {cleaned_json_text}")
            raise ValueError("La IA devolvió un JSON inválido para puntos clave.")

        if not isinstance(key_points_data.get("key_points_es"), list) or \
           not isinstance(key_points_data.get("key_points_en"), list):
            raise ValueError("La respuesta de la IA no contiene los puntos clave en formato de lista en ambos idiomas.")

        return key_points_data

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error inesperado en get_key_points: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la solicitud de puntos clave.")
