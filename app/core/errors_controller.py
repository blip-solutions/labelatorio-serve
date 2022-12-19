from datetime import datetime, timezone
import logging
from typing import List
from fastapi import APIRouter
from starlette.authentication import requires
from fastapi import Request
import persistqueue
import app.core.error_queue as error_queue
from labelatorio import Client
from ..config import  LABELATORIO_API_TOKEN, LABELATORIO_URL

router = APIRouter()






@router.get("/errors", response_model=List[dict], summary="Get list of records, that were not uploaded to Labelator.io successfuly ", tags=["errors"])
@requires(['authenticated'])
def get_errors(request:Request)->List[dict]:
    result = error_queue.peek_errors()
    return result


@router.post("/errors/retry", response_model=dict,  summary="Retry push documents to Labelator.io", tags=["errors"])
@requires(['authenticated'])
def retry_errors(request:Request)->dict:
    
    labelatorio_client = Client(LABELATORIO_API_TOKEN, url=LABELATORIO_URL)

    def process(err_rec):
        docs=err_rec["docs"]
        project_id=err_rec["project_id"]
        labelatorio_client.documents.add_documents(project_id, docs, upsert=True)
        
    return error_queue.retry_errors(process)


@router.delete("/errors", response_model=dict ,  summary="Clear errors without pushing to Labelator.io", tags=["errors"])
@requires(['authenticated'])
def clear_errors(request:Request)->dict:
    return error_queue.clear()