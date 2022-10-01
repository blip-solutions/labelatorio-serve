from typing import Optional, List, Dict, Union
from pydantic import BaseModel

from .configuration import NodeSettings





class ResponseBase(BaseModel):
    success: bool = True
    error: Optional[List]
    data: Optional[Union[List, Dict[str, str]]]


class Info(BaseModel):
    info: str
    version: str
    node_name:str
    settings:Optional[NodeSettings]
    root_path:Optional[str]



class Prediction(BaseModel):
    label:str
    score:float




class RouteExplanation(BaseModel):
    route_id:int
    route_type:str
    route_handling:str
    matched:bool
    used:bool
    matched_similar:Optional[bool]
    matched_prediction_score:Optional[bool]
    matched_similar_example:Optional[str]
    matched_similarity_score:Optional[float]
    matched_correct_prediction:Optional[bool]
    


class PredictedItem(BaseModel):
    predicted:Union[List[Prediction], None]
    handling:str
    key:Optional[str]=None
    explanations:Optional[List[RouteExplanation]]=None


class PredictctRespone(BaseModel):
    predictions:List[PredictedItem]


