import numpy as np
from darts.models import LinearRegressionModel, LightGBMModel, XGBModel


def get_models(model_names, model_configs):
    """
    Creates a list of model instances based on the provided model names and configurations.

    Args:
        model_names (list): List of model names to instantiate. Supported names include 'lr' for 
                            LinearRegressionModel, 'lgbm' for LightGBMModel, and 'xgb' for XGBModel.
        model_configs (list of dict): List of dictionaries containing configurations for each model 
                                    specified in model_names. Each dictionary should contain parameters 
                                    specific to the corresponding model.

    Returns:
        list: A list of instantiated model objects based on the provided model_names and model_configs.

    If 'xgb' is included in model_names, the method modifies its configuration to use the 'hist' 
    tree method for faster training.
    """
    models = {
        "lr": LinearRegressionModel,
        "lgbm": LightGBMModel,
        "xgb": XGBModel,
    }
    
    if "xgb" in model_names:
        xgb_idx = np.where(np.array(model_names)=="xgb")[0]
        for idx in xgb_idx:
            # change to histogram-based method for XGBoost to get faster training time
            model_configs[idx] = {"tree_method": "hist", **model_configs[idx]}
    
    return [models[name](**model_configs[j]) for j, name in enumerate(model_names)]


if __name__ == "__main__":
    pass