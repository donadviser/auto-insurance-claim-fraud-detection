from insurance import logging
from insurance import CustomException
from insurance.pipeline.training_pipeling import TrainPipeline



if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()