
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from database import addData, getData
from services import findResponse


app = FastAPI(title="RAG Application")


class DocumentInput(BaseModel):
    documents: list[str]


class ChatInput(BaseModel):
    prompt: str


@app.get("/")
def home():
    return "Welcome to RAG Application!"


@app.post("/collection")
def storeData(data: DocumentInput):
    try:
        print("Received chunks:", len(data.documents))

        addData(data.documents)

        return {"message": "Documents stored successfully!"}

    except Exception as e:
        print("ERROR IN /collection:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection")
def getCollectionData():
    return getData()


@app.post("/chat")
def getResponse(query: ChatInput):
    try:
        print("Question received:", query.prompt)

        response = findResponse(query.prompt)

        print("Response generated successfully!")

        return {"response": response}

    except Exception as e:
        print("CHAT ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
