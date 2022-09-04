from typing import Dict, List, Union, Optional
from pydantic import BaseModel

class PredictionRequestRecord(BaseModel):
    key:str
    text:str
    contextData:Optional[Dict[str,str]]

class PredictRequestBody(BaseModel):
    texts:List[Union[str,PredictionRequestRecord]]


