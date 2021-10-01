import joblib
import uvicorn
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("Model.pkl")
vec_model = joblib.load("vectorizer.pkl")


@app.get("/Home")
def home():
    return {"NLP":"Stock sentiment analysis based on News-Headlines"}

@app.post("/predict")
def predict(headline:str):
        data = [headline]
        data_ = vec_model.transform(data)
        result = model.predict(data_)
        result = int(result)
        if result==1:
            prediction= "stock price will have positive impact"
        else:
            prediction = "stock price will have negative impact"
        return {"result":prediction}


    
if __name__=="__main__":
    uvicorn.run()