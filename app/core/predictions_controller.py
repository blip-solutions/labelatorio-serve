from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Security, BackgroundTasks
from .configuration import configuration_client
from .contants import RouteHandlingTypes
from .predictions import PredictionModule
from ..models.requests import PredictRequestBody
from ..models.responses import PredictctRespone
from starlette.authentication import requires
from fastapi import Request
import os

major_version="0.1"
minor_version = os.environ["BUILD_VERSION"] if "BUILD_VERSION" in os.environ else  "_dev_"

from ..models.responses import Info, PredictctRespone

router = APIRouter()






@router.post("/predict/{model_name}", response_model=PredictctRespone, response_model_exclude_none=True)
@requires(['authenticated'])
def predict(background_tasks: BackgroundTasks,request: Request, body:PredictRequestBody, model_name:Optional[str]=None, explain:Optional[bool]=False,  test:Optional[bool]=False)->Info:
    """Get all data base Users"""
    
    result = request.app.state.prediction_module.predict_labels(background_tasks, body.texts, model_name=model_name, explain=explain, test=test)
    return PredictctRespone(predictions=result)

@router.post("/predict", response_model=PredictctRespone, response_model_exclude_none=True)
@requires(['authenticated'])
async def predict_default(background_tasks: BackgroundTasks, request: Request,  body:PredictRequestBody, explain:Optional[bool]=False, test:Optional[bool]=False)->Info:
    """Get all data base Users"""
    
    result = request.app.state.prediction_module.predict_labels(background_tasks,body.texts, model_name=None, explain=explain, test=test)
    return PredictctRespone(predictions=result)

        
