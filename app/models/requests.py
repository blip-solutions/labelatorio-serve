from typing import Dict, List, Union, Optional
from pydantic import BaseModel

class PredictionRequestRecord(BaseModel):
    key:str
    text:str
    contextData:Optional[Dict[str,str]]
    reviewProjectId:Optional[str] #where to send data for review... if not set, will be determined by model project

class PredictRequestBody(BaseModel):
    texts:List[Union[str,PredictionRequestRecord]]


