from datetime import datetime, timezone
import logging
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
import persistqueue

from labelatorio import Client
from ..config import  LABELATORIO_API_TOKEN, LABELATORIO_URL

router = APIRouter()



errors_queue= persistqueue.SQLiteQueue('queues/error_queue.db', auto_commit=True, multithreading=True)


@router.get("/errors", response_model=List[dict])
@requires(['authenticated'])
def get_errors(request:Request)->List[dict]:
   
    
    result = errors_queue.queue()
    return result


@router.post("/errors/retry", response_model=dict)
@requires(['authenticated'])
def retry_errors(request:Request)->dict:
    
    labelatorio_client = Client(LABELATORIO_API_TOKEN, url=LABELATORIO_URL)

    resolved=[]
    while True:
        try:
            err_rec= errors_queue.get(block=False)
        except persistqueue.Empty:
            break

        docs=err_rec["docs"]
        project_id=err_rec["project_id"]
        counter=err_rec["counter"]
        try:
            labelatorio_client.documents.add_documents(project_id, docs, upsert=True)
            resolved.append({"project_id":project_id, "docs":docs})
        except Exception as ex:
            errors_queue.put({"docs":docs, "error":str(ex), "counter":counter+1, "timestamp":datetime.now(timezone.utc)})
            print(ex)
            logging.exception(ex)

    
    return {"resolved":len(resolved)}


@router.delete("/errors", response_model=dict)
@requires(['authenticated'])
def retry_errors(request:Request)->dict:
    counter=0
    while True:
        try:
            err_rec= errors_queue.get(block=False)
            counter=counter+1
        except persistqueue.Empty:
            break
    return {"deleted":len(counter)}