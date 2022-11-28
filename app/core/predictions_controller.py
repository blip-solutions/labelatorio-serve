from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Security, BackgroundTasks
from pydantic.error_wrappers import ValidationError
from app.models.configuration import ModelSettings
from .configuration import configuration_client
from .contants import RouteHandlingTypes
from .predictions import PredictionModule, TemporaryPredictionModule
from ..models.requests import PredictRequestBody
from ..models.responses import PredictctResponse
from starlette.authentication import requires
from fastapi import Request
import os



from ..models.responses import Info, PredictctResponse

router = APIRouter()






@router.post("/predict", response_model=PredictctResponse, response_model_exclude_none=True,
    summary="Get predictions for request",
    description="""
    Allows query predictions for one or more texts

    If final handling is manual or manual-review, data will be added to labelatorio project (unlest test mode is activated  ./predict?test=true)
    For this it is recomended to list of string, but rather list of objects, cointaing key, text and optional contextData. This will allow additional data to be send to Labelator.io.

    If explain mode activated ( ./predict?explain=true), response will contain explanations for all routing configuration for better understanding of the decision

    Settings parameter is to enable query with custom settings for particular request. It is generaly not recomended to use this in production since it can rapidly decrease the performance. 
    Especialy if new settings are pointing to model that has not been preloaded by defaul configuration

    """
)
@requires(['authenticated'])
def predict(background_tasks: BackgroundTasks,request: Request, body:PredictRequestBody, model_name:Optional[str]=None , explain:Optional[bool]=False,  test:Optional[bool]=False)->Info:
    print(body)
    if body.settings is not None:
        try:
            prediction_module = TemporaryPredictionModule(ModelSettings(**body.settings))
        except ValidationError as ex:
            raise HTTPException(422, detail=str(ex))
    else:
        prediction_module =  request.app.state.prediction_module

    result = prediction_module.predict_labels(background_tasks, body.texts, model_name=model_name, explain=explain, test=test)
    return PredictctResponse(predictions=result)



# @router.post("/predict", response_model=PredictctResponse, response_model_exclude_none=True)
# @requires(['authenticated'])
# async def predict_default(background_tasks: BackgroundTasks, request: Request,  body:PredictRequestBody, explain:Optional[bool]=False, test:Optional[bool]=False)->Info:
#     print(body)
#     if body.settings is not None:
#         prediction_module = TemporaryPredictionModule(body.settings)
#     else:
#         prediction_module =  request.app.state.prediction_module
#     result = prediction_module.predict_labels(background_tasks,body.texts, model_name=None, explain=explain, test=test)

#     return PredictctResponse(predictions=result)

        

@router.post("/get-answer", response_model=PredictctResponse, response_model_exclude_none=True)
@requires(['authenticated'])
async def get_answer(background_tasks: BackgroundTasks, request: Request,  body:PredictRequestBody, top_k:int=1, explain:Optional[bool]=False, test:Optional[bool]=False)->Info:
    prediction_module:PredictionModule = request.app.state.prediction_module
    result =prediction_module.predict_answer(background_tasks,body.texts, model_name=None, top_k=top_k, explain=explain, test=test)
    return PredictctResponse(predictions=result)