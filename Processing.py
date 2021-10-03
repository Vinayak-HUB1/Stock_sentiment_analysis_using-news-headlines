from Logging.setup_logger import setup_logger_
import joblib

logger = setup_logger_("processingLOgs","Processing.log")


def pre_processing(data):
    vec_model = joblib.load("vectorizer.pkl")
    logger.info("vectorizer loaded")
    vec_result = vec_model.transform(data)
    logger.info("data transformation completed")
    return vec_result
   

