# AI-voice-agent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "llama3-70b-8192"


def get_llm(model_name):
    return ChatGroq(
        temperature=0,
        model_name=model_name,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


class TextRequest(BaseModel):
    text: str


@app.post("/process_text")
async def process_text(request: TextRequest):
    prompt = f"""
    You are a conversational assistant named Mark.
    Use short, natural responses as if you're having a real conversation.
    Your response should be under 20 words.
    *Do not generate user identifiers like "User-xxxx" or refer to the user with a number.*
    Do not respond with any code, only conversation.

    {request.text}
    """
    try:
        llm = get_llm(PRIMARY_MODEL)
        response = llm.invoke(prompt)
    except Exception as e:
        if "terms acceptance" in str(e).lower():
            try:
                llm = get_llm(FALLBACK_MODEL)
                response = llm.invoke(prompt)
            except Exception as fallback_error:
                raise HTTPException(status_code=500, detail=f"Fallback model error: {str(fallback_error)}")
        else:
            raise HTTPException(status_code=500, detail=f"Primary model error: {str(e)}")

    return {"response": response.content}
