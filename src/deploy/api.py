from fastapi import FastAPI
from pydantic import BaseModel
from src.infer import ChatbotPredictor

app = FastAPI()
predictor = ChatbotPredictor("models/trained/chatbot")

# Define a Pydantic model for the request body
class Query(BaseModel):
    text: str

@app.post("/predict")
def predict(query: Query):
    intent = predictor.predict(query.text)
    return {"intent": intent}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)