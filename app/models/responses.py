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

# class Answer(BaseModel):
#     answer:str
#     score:float

class PredictionMatchExplanation(BaseModel):
    prediction:Prediction
    matched:bool

class SimilarExample(BaseModel):
    text:str
    score:float
    labels:Optional[List[str]]
    correctly_predicted:Optional[bool]

class RouteExplanation(BaseModel):
    route_id:int
    route_type:str
    route_handling:str
    matched:bool
    used:bool
    matched_prediction:Optional[List[PredictionMatchExplanation]]
    matched_similar:Optional[bool]
    matched_similar_examples:Optional[List[SimilarExample]]
    matched_regex:Optional[bool]=None
    #matched_correct_prediction:Optional[bool]
    


class PredictedItem(BaseModel):
    #predicted:Union[List[Prediction],Answer, None]
    predicted:Union[List[Prediction], None]
    handling:str
    key:Optional[str]=None
    explanations:Optional[List[RouteExplanation]]=None


class PredictctRespone(BaseModel):
    predictions:List[PredictedItem]


