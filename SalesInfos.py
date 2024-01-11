from pydantic import BaseModel
from datetime import date

class SalesInfo(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float
    lag_6: float
    lag_7: float
    lag_8: float
    lag_9: float
    lag_10: float

