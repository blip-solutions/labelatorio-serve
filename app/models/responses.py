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
    settings:Optional[Dict]
    root_path:Optional[str]



class Prediction(BaseModel):
    label:str
    score:float

class Answer(BaseModel):
    answer:str
    score:float



class PredictionMatchExplanation(BaseModel):
    prediction:Prediction
    matched:bool


class SimilarExample(BaseModel):
    text:str
    score:float
    labels:Optional[List[str]]
    correctly_predicted:Optional[bool]

class SimilarQuestionAnswer(BaseModel):
    id:Optional[str]=None
    text:str
    similarity_score:Optional[float]=None
    relevancy_score:Optional[float]=None
    answer:str

class RouteExplanation(BaseModel):
    route_id:int
    route_type:str
    route_handling:str
    matched:bool
    used:bool
    matched_prediction:Optional[Union[List[PredictionMatchExplanation],bool]]
    matched_similar:Optional[bool]
    matched_similar_examples:Optional[List[Union[SimilarExample, SimilarQuestionAnswer]]]
    matched_regex:Optional[bool]=None
    #matched_correct_prediction:Optional[bool]
    
class RetrievalStepSummary(BaseModel):
    retrieved_total:int
    matched_data:Optional[List[SimilarQuestionAnswer]]=None
    

class PredictedItem(BaseModel):
    predicted:Union[List[Prediction],List[Answer], None]
    #predicted:Union[List[Prediction], None]
    handling:str
    key:Optional[str]=None
    explanations:Optional[List[Union[RouteExplanation,RetrievalStepSummary]]]=None


class PredictedResponse(BaseModel):
    predictions:List[PredictedItem]


