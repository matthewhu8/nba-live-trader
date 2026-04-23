import fastapi
import torch as torch
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import time
import warnings
import joblib
import gc

app = fastapi.FastAPI()

class XFeatureInput(fastapi.BaseModel):
    x_physics: list[list[float]]
    x_pregame: list[list[float]]
    x_market: list[list[float]]

@app.post("/predict")
async def predict(request: XFeatureInput):
    pass
    # recieves X vector from Go with features grouped into 3 categories
    # 1. x_physics
    # 2. x_pregame
    # 3. x_market


    