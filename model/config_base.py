"""模型配置的公共路径规则，不包含任意一个模型的隐藏超参数。

参考工程的 model_Config.Config 保留为每个模型的唯一编辑入口。
这里仅集中处理派生路径与配置快照，修改实验名/折号后路径会同步变化。
"""
from copy import copy
from pathlib import Path


class BaseConfig:
    @property
    def CODE_ROOT(self):
        return Path(self.SCRIPT_DIR).resolve().parents[1]

    @property
    def DATA_DIR(self):
        return self.CODE_ROOT / "original_data" / "processed" / f"fold_{self.FOLD:02d}"

    @property
    def DATA_PATH(self):
        return self.DATA_DIR / "train_dataset_norm.npz"

    @property
    def TEST_DATA_PATH(self):
        return self.DATA_DIR / "test_dataset_norm.npz"

    @property
    def NORM_DATA_PATH(self):
        return self.DATA_DIR / "norm_params.npz"

    @property
    def METADATA_PATH(self):
        return self.DATA_DIR / "metadata.json"

    @property
    def RUN_DIR(self):
        category = "smoke" if self.SMOKE else "experiments"
        return Path(self.SCRIPT_DIR) / category / self.EXPERIMENT_NAME / f"fold_{self.FOLD:02d}"

    @property
    def SAVE_MODEL_DIR(self):
        return self.RUN_DIR / "save_model"

    @property
    def RESULT_DIR(self):
        return self.RUN_DIR / "results"

    @property
    def FIGURE_DIR(self):
        return self.RESULT_DIR / "figures"

    @property
    def MODEL_SAVE_PATH(self):
        return self.SAVE_MODEL_DIR / self.BEST_MODEL_FILENAME

    @property
    def FINAL_MODEL_PATH(self):
        return self.SAVE_MODEL_DIR / self.FINAL_MODEL_FILENAME

    def clone(self):
        return copy(self)

    def as_dict(self):
        """保存全部大写参数和实际派生地址，方便对照每次实验。"""
        result = {}
        for key in dir(self):
            if key.isupper():
                value = getattr(self, key)
                result[key] = str(value) if isinstance(value, Path) else value
        return result

    @classmethod
    def from_dict(cls, values):
        """读取权重内的配置；地址始终由当前工程位置推导，不读旧盘符。"""
        instance = cls()
        # 2026-10-22的旧权重使用单层解码器；新论文配置为三层，不可误套到旧权重。
        if getattr(instance, "MODEL_NAME", None) == "S2SMAG":
            instance.DECODER_NUM_LAYERS = values.get("DECODER_NUM_LAYERS", 1)
            instance.DECODER_DROPOUT_ENABLED = values.get("DECODER_DROPOUT_ENABLED", False)
            instance.ENCODER_DROPOUT = values.get("ENCODER_DROPOUT", values.get("DROPOUT", 0.2))
            instance.TEACHER_FORCING_RNG = values.get("TEACHER_FORCING_RNG", "torch")
        for key, value in values.items():
            if key.isupper() and key != "SCRIPT_DIR" and hasattr(instance, key):
                if not isinstance(getattr(cls, key, None), property):
                    setattr(instance, key, value)
        return instance
