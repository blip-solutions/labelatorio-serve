from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Security, BackgroundTasks

from app.models.configuration import ModelSettings
from .configuration import configuration_client
from .contants import RouteHandlingTypes
from .predictions import PredictionModule, TemporaryPredictionModule
from ..models.requests import PredictRequestBody
from ..models.responses import PredictctRespone
from starlette.authentication import requires
from fastapi import Request
import os



from ..models.responses import Info, PredictctRespone

router = APIRouter()






@router.post("/predict", response_model=PredictctRespone, response_model_exclude_none=True)
@requires(['authenticated'])
def predict(background_tasks: BackgroundTasks,request: Request, body:PredictRequestBody, model_name:Optional[str]=None , explain:Optional[bool]=False,  test:Optional[bool]=False)->Info:
    print(body)
    if body.settings is not None:
        
        prediction_module = TemporaryPredictionModule(ModelSettings(**body.settings))
    else:
        prediction_module =  request.app.state.prediction_module

    result = prediction_module.predict_labels(background_tasks, body.texts, model_name=model_name, explain=explain, test=test)
    return PredictctRespone(predictions=result)



# @router.post("/predict", response_model=PredictctRespone, response_model_exclude_none=True)
# @requires(['authenticated'])
# async def predict_default(background_tasks: BackgroundTasks, request: Request,  body:PredictRequestBody, explain:Optional[bool]=False, test:Optional[bool]=False)->Info:
#     print(body)
#     if body.settings is not None:
#         prediction_module = TemporaryPredictionModule(body.settings)
#     else:
#         prediction_module =  request.app.state.prediction_module
#     result = prediction_module.predict_labels(background_tasks,body.texts, model_name=None, explain=explain, test=test)

#     return PredictctRespone(predictions=result)

        

# @router.post("/get-answer", response_model=PredictctRespone, response_model_exclude_none=True)
# @requires(['authenticated'])
# async def get_answer(background_tasks: BackgroundTasks, request: Request,  body:PredictRequestBody, explain:Optional[bool]=False, test:Optional[bool]=False)->Info:
#     print(body)
#     result = request.app.state.prediction_module.get_answer(background_tasks,body.texts, model_name=None, explain=explain, test=test)
#     return PredictctRespone(predictions=result)