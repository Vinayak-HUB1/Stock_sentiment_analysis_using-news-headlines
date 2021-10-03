import joblib
import uvicorn
from fastapi import FastAPI
from Processing import pre_processing
from Prediction import prediction_
from Logging.setup_logger import setup_logger_

app = FastAPI()


@app.get("/")
def home():
    return {"NLP":"Stock sentiment analysis based on News-Headlines",
             "Prediction":"please add /docs in above url"}


@app.post("/predict")
def predict(headline:str):
        data = [headline]
        vec = pre_processing(data)
        result = prediction_(vec)
        result = int(result)
        if result==1:
            prediction= "stock price will have positive impact"
        else:
            prediction = "stock price will have negative impact"
        return {"result":prediction}


    
if __name__=="__main__":
    uvicorn.run()