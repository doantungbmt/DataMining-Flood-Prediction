from pydantic import BaseModel, Field

class FloodPredictionInput(BaseModel):
    muc_nuoc: float = Field(..., description="Normalized Mực nước hiện tại")
    month: int = Field(..., description="Tháng hiện tại (1-12)")
    rolling_mean_7d: float = Field(..., description="Normalized 7-day Rolling Mean")
    delta_1d: float = Field(..., description="Normalized Delta 1 day")
    dung_tich: float = Field(..., description="Normalized Dung tích (m3)")
    q_den: float = Field(..., description="Normalized Q đến (m3/s)")
    q_xa: float = Field(..., description="Normalized Q xả (m3/s)")

class FloodPredictionOutput(BaseModel):
    predicted_muc_nuoc_t_plus_1: float
