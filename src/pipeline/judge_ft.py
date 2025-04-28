from __future__ import annotations

from src.components.data_pre.sft_judge import SFTJudge as DataJudge
from src.components.load_model.load_modules import ModelManager
from src.components.trainer.sft_judge import SFTJudge


class PipelineJudgeFT:
    @staticmethod
    def create_data_module():
        data_process = DataJudge()
        return data_process

    @staticmethod
    def create_model_manager():
        model_manager = ModelManager()
        return model_manager

    @staticmethod
    def create_trainer_module():
        trainer_module = SFTJudge()
        return trainer_module
