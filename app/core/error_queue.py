from datetime import datetime, timezone
import logging
from typing import List, Callable,Tuple,Optional
from fastapi import APIRouter
from starlette.authentication import requires
from fastapi import Request
import persistqueue

errors_queue= persistqueue.SQLiteAckQueue('queues/error_queue.db',  multithreading=True, auto_commit=True)


def peek_errors()->List[dict]:
    """Returns all the errors in queue without pop-ing them

    Returns:
        List[dict]: _description_
    """
    result = errors_queue.queue()
    return result

def retry_errors(process_error_func:Callable[[dict],None], max_retry_counter:Optional[int] =None, clear_acked_data=True) -> List[dict]:
    """Retries to process errors with custom function. When it fails (exception needs to be throwns), 
    it pushes the record back into queue with incremented retry count

    Args:
        process_error_func: function that retrieves dictionary with {docs:List, project_id:str,counter:int}
        max_retry_count: - allows to retrieve only errors up to certain retry count.
    Returns:
        List of resolved error records
    """
    
    resolved=[]
    
    failed=[]
    skipped=[]
    while True:
        try:
            err_rec= errors_queue.get(block=False)
        except persistqueue.Empty:
            break

      
        counter=err_rec["counter"]
        if max_retry_counter and counter>max_retry_counter:
            #if its already has more retries than limit, push it to the end of the queue
            skipped.append(err_rec)
            continue
        try:
            process_error_func(err_rec)
            resolved.append(err_rec)
            errors_queue.ack(err_rec)
        except Exception as ex:
            err_rec["counter"]=counter+1
            failed.append(err_rec)
            logging.exception(ex)

    for err_rec in skipped:
        errors_queue.nack(err_rec)

    for err_rec in failed:
        _id = errors_queue.nack(err_rec)
        errors_queue.update(err_rec,_id)

    if clear_acked_data:
        errors_queue.clear_acked_data(keep_latest=0)

    
    return {"resolved":len(resolved),"failed":len(failed)}

def put(docs:List[dict],project_id:str, error_msg:str):
    errors_queue.put({"docs":docs, "project_id":project_id, "error":error_msg, "counter":1, "timestamp":datetime.now(timezone.utc)})

def clear()->int:
    """Clear all errors in queue

    Returns:
        int: number of deleted
    """
    counter=0
    while True:
        try:
            
            err_rec= errors_queue.get(block=False)
            errors_queue.ack(err_rec)
            counter=counter+1
        except persistqueue.Empty:
            break
    errors_queue.clear_acked_data(keep_latest=0)
    return {"deleted":counter}