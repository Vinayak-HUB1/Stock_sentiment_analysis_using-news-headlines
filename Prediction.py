import joblib
from Logging.setup_logger import setup_logger_


logger = setup_logger_("PredictionLogs","Prediction.log")


def prediction_(data):
    model = joblib.load("Model.pkl")
    logger.info("Prediction model Loaded")
    result = model.predict(data)
    logger.info(f"predictions are {result}")
    return result