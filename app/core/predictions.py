import gc
import logging
import re
from datetime import datetime, timezone
from email.policy import default
from functools import lru_cache
from pickle import FALSE
from typing import Dict, List, Optional, Union
import numpy as np
import persistqueue
import torch
from fastapi import BackgroundTasks, HTTPException
from labelatorio import Client
from labelatorio.client import EndpointGroup

import app.core.error_queue as error_queue
from app.core.models_cache import TASK_TYPES, ModelsCache
from app.models.configuration import (ModelSettings, NodeSettings,
                                      RoutingSetting)

from ..config import LABELATORIO_API_TOKEN, LABELATORIO_URL, TEST_MODE
from ..models.requests import PredictionRequestRecord
from ..models.responses import (Answer, PredictedItem, Prediction,
                                PredictionMatchExplanation, RouteExplanation,
                                SimilarExample)
from .configuration import NodeConfigurationClient
from .contants import NodeStatusTypes, RouteHandlingTypes, RouteRuleType


labelatorio_client =Client(LABELATORIO_API_TOKEN, url=LABELATORIO_URL)
models_cache=ModelsCache(labelatorio_client )

class PredictionModule:
    def __init__(self, configurationClient:NodeConfigurationClient ) -> None:
        if (torch.cuda.is_available()):
            print("CUDA availible... using GPU")
        
        if not hasattr(self, "settings"): #used by subclasses
            self.settings=None
        self.configurationClient=configurationClient
        if self.configurationClient and self.configurationClient.labelatorio_client:
            self.labelatorio_client = self.configurationClient.labelatorio_client
        else:
            self.labelatorio_client =labelatorio_client
        self.models_cahce=models_cache

        self.backlog_queue= persistqueue.SQLiteQueue('queues/backlog_queue', auto_commit=True, multithreading=True)
    
        self.model_anchor_vectors={}
        self.models_paths={}
        self.closest =  ClosestNeighbourEndpointGroup(self.labelatorio_client)
        self.memory_cache={}
        self.reinitialize()
        self.configuration_error=None
        if self.backlog_queue.size>0:
            logging.warning(f"backlog_queue not empty ({self.backlog_queue.size}). Flushing now!")
            try:
                self.flush_send_to_backlog()
            except Exception as ex:
                logging.info("Error during flushing on init")
                logging.exception(ex)
                

    

    def reinitialize(self):
        if self.configurationClient:
            self.configurationClient.ping(status=NodeStatusTypes.UPDATING)
            self.settings=self.configurationClient.settings
             
        settings=self.settings
        self.model_anchor_vectors={}
        self.regexes={}
        old_paths = self.models_paths
        self.models_paths={}

        del self.memory_cache
        gc.collect()
        self.memory_cache={}
        
        try:
            if settings and settings.models:
                self.default_model=settings.default_model or settings.models[0].model_name 
                for modelConfig in settings.models:
                    
                    self.models_cahce.predownload_model(modelConfig.project_id,modelConfig.model_name)
                    
                    if modelConfig.similarity_model:
                        self.models_cahce.predownload_model(modelConfig.project_id,modelConfig.similarity_model)
                        
                    elif (modelConfig.routing or modelConfig.default_handling!=RouteHandlingTypes.MODEL_AUTO) :
                        raise Exception(f"{modelConfig.model_name} :Invalid configuration. No similarity model defined. This is allow only for full auto mode without routing")
                    

                    print("preparing anchors")
                    for route_id, route in enumerate(modelConfig.routing):
                        if route.rule_type==RouteRuleType.ANCHORS:
                            similarity_model=self.models_cahce.get_similarity_model(modelConfig.project_id, modelConfig.similarity_model)
                            if self.model_anchor_vectors.get(modelConfig.model_name) is None:
                                self.model_anchor_vectors[modelConfig.model_name]={}
                            self.model_anchor_vectors[modelConfig.model_name][route_id] =similarity_model.encode(route.anchors,  normalize_embeddings=True)
                        if route.regex:
                            if modelConfig.model_name not in self.regexes:
                                self.regexes[modelConfig.model_name]={}
                            self.regexes[modelConfig.model_name][route_id] = re.compile(route.regex)
                
                if self.default_model: #preload default models into memory
                    defatult_settings = settings.get_model_settings(self.default_model)
                    self.models_cahce.get_pipeline(defatult_settings.project_id, self.default_model, defatult_settings.task_type)  #preload into memory
                    default_similarity_model_settings = next((model for model in settings.models if model.model_name==self.default_model),None)
                    if default_similarity_model_settings:
                        self.models_cahce.get_similarity_model(default_similarity_model_settings.project_id, default_similarity_model_settings.similarity_model)

            if self.configurationClient:
                self.configurationClient.ping(status=NodeStatusTypes.READY)
            print("Reconfiguraiton ready")
            #TODO: Delete old paths to save disk space?>>  old_paths
        except Exception as ex:
            self.configuration_error=repr(ex)
            self.configurationClient.ping(status=NodeStatusTypes.ERROR)
                        



    def predict_labels(self, background_tasks:BackgroundTasks, data_to_send:Union[List[str], List[PredictionRequestRecord]], model_name:Optional[str], explain:Optional[bool]=None, test:Optional[bool]=False)->PredictedItem:

        if self.configuration_error:
            raise HTTPException(520,{"error":f"Configuration error: {self.configuration_error}"})
        if not self.settings.models:
            raise HTTPException(520,{"error":"No models are defined for this node"})
        if not model_name:
            model_name = self.default_model
        
        pipe=None
        settings =  self.settings.get_model_settings(model_name)
        if not settings:
            raise HTTPException(520,{"error":f"Model {model_name} is not deployed on this node"})

        if not data_to_send:
            return []

        uniqueTexts = { rec if  isinstance(rec,str) else rec.text  for rec in data_to_send}
        


        if explain:
            explanations={}
             #just init the varialbes
            max_similarity=None
            max_sim_anchor_index=None
        #ignore duplicates
        texts_to_handle = list(uniqueTexts)

        texts_routes_matches=[None for i in range(len(texts_to_handle))]
        
        query_vectors=None

        if settings.routing:
            correctly_predicted_min_range = min((route.similarity_range.min for route in settings.routing if route.rule_type==RouteRuleType.TRUE_POSITIVES), default=None)
            
            query_vectors = self.models_cahce.get_similarity_model(settings.project_id, settings.similarity_model).encode(texts_to_handle, normalize_embeddings=True)
            
            closest=None
            matched_closest={}
            if correctly_predicted_min_range:
                closest = self.closest.get_closest(
                    settings.project_id, 
                    query_vectors=query_vectors,
                     min_score=correctly_predicted_min_range/100 if not explain else 0, 
                     select_fields=["id", "text"] ,
                     correctly_predicted=True )
            
            for i, text2handle in enumerate(texts_to_handle):
                if explain:
                    explanations[text2handle]={}
            

                closest_correctly_predicted=None
                

                if correctly_predicted_min_range and closest:
                    closest_correctly_predicted =closest[i]

                matched_routes:Dict[int,RoutingSetting]={}
                
                
                for route_id, route in enumerate(settings.routing) :
                    
                    # start with regex, since that is the cheapest test
                    if route.regex:
                        if not self.regexes[model_name][route_id].search(text2handle):
                            matched_regex=False
                            if explain:
                                explanations[text2handle][route_id]=RouteExplanation(
                                        route_id=route_id, 
                                        route_type=route.rule_type, 
                                        route_handling=route.handling,
                                        matched=False,
                                        used=False,
                                        matched_prediction=None,
                                        matched_regex=False,
                                        matched_similar=None,
                                        matched_similar_examples=None
                                    )
                            continue
                        else:
                            matched_regex=True
                    else:
                        matched_regex=None

                    if route.rule_type==RouteRuleType.TRUE_POSITIVES:
                        
                        if closest_correctly_predicted:
                            for item in closest_correctly_predicted:
                                if item["correctly_predicted"] is None:
                                    # if no prediction was made yet we need to run it our self
                                    pipe = self.models_cahce.get_pipeline(settings.project_id, model_name, settings.task_type) 
                                    predictions= set(p["label"] for p in pipe(item["text"], padding=True, truncation=True)[0] if p["score"]>0.5)
                                    if set(item["labels"])==predictions:
                                        item["correctly_predicted"]=True

                            matched_sim_range_items = [item for item in closest_correctly_predicted if item["correctly_predicted"] and  round(item["score"]*100,3)>=route.similarity_range.min and round(item["score"]*100,3)<=route.similarity_range.max]
                            if i not in matched_closest:
                                matched_closest[i]={route_id:matched_sim_range_items}
                            else:
                                matched_closest[i][route_id] = matched_sim_range_items
                            if matched_sim_range_items:
                                correctly_predicted_ratio = sum(1 for item in matched_sim_range_items if item["correctly_predicted"])+1 / len(matched_sim_range_items)
                                if correctly_predicted_ratio*100>route.similarity_range.min:
                                    matched_routes[route_id]= route

                        if explain:
                            explanations[text2handle][route_id]=RouteExplanation(
                                    route_id=route_id, 
                                    route_type=route.rule_type, 
                                    route_handling=route.handling,
                                    matched=False,
                                    used=False,
                                    matched_regex=matched_regex,
                                    matched_prediction=None,
                                    matched_similar=True if route_id in matched_routes else False, 
                                    matched_similar_examples= None
                                 )
                                
                            
                    elif route.rule_type==RouteRuleType.ANCHORS:
                        #get vector1
                        max_similarity=None
                        max_sim_anchor_index=None
                        for anchor_index, anchor_vec in enumerate(self.model_anchor_vectors[model_name][route_id]):
                            similarity = round((np.dot(query_vectors[i],anchor_vec)+1)/2,3)
                            
                            if explain and (not max_similarity or similarity>max_similarity):
                                max_similarity =similarity
                                max_sim_anchor_index= anchor_index

                            if similarity>=route.similarity_range.min/100 and similarity<=route.similarity_range.max/100:
                                matched_routes[route_id]= route
                                break

                        if explain:
                            explanations[text2handle][route_id]=RouteExplanation(
                                    route_id=route_id, 
                                    route_type=route.rule_type, 
                                    route_handling=route.handling,
                                    matched=False,
                                    used=False,
                                    matched_regex=matched_regex,
                                    matched_prediction=None,
                                    matched_similar=True if route_id in matched_routes else False, 
                                    matched_similar_examples=[SimilarExample(text=route.anchors[max_sim_anchor_index], score=max_similarity)] if max_sim_anchor_index is not None else None
                                 )
                                
                            
                    else:
                        raise HTTPException(500,{"error":f"Invalid  routing rule type '{route.rule_type}'"})

                texts_routes_matches[i] = matched_routes
                  
        if RouteHandlingTypes.should_predict(settings.default_handling) or explain:
            texts_to_predict= texts_to_handle
        else:
            texts_to_predict = [text for  text, matched_routes in zip(texts_to_handle,texts_routes_matches) if  any(1 for route in matched_routes.values() if RouteHandlingTypes.should_predict(route.handling) ) ]
        if texts_to_predict:
            if pipe is None:
                if not model_name:
                    raise HTTPException(400, "Model name is required" )
                pipe = self.models_cahce.get_pipeline(settings.project_id, model_name, settings.task_type) 
                if pipe is None:
                    raise HTTPException(512, "Model name is required" )
            predictions= {text:doc_predictions  for text, doc_predictions in zip(texts_to_predict,pipe(texts_to_predict, padding=True, truncation=True))}
        else:
            predictions={}

        text_handled_routes = { }
        
        for i, (text, matched_routes) in  enumerate(zip(texts_to_handle,texts_routes_matches)):
            if matched_routes:
                for route_id,route in matched_routes.items():
                    if route.prediction_score_range :
                        if route.predicted_labels and text in predictions:
                            predictions_to_test= [prediction for prediction in predictions[text] if prediction["label"] in route.predicted_labels]
                            
                        else:
                            predictions_to_test=predictions.get(text) or []
                            

                        if any(1 for prediction in predictions_to_test if
                                             prediction["score"]>=(route.prediction_score_range.min or -1000) 
                                        and  prediction["score"]<=(route.prediction_score_range.max or 1000)  
                                        and  ((route_id in matched_closest[i] and  ((i in matched_closest and any( 1 for item in matched_closest[i][route_id] if prediction["label"] in item["labels"] )) ) or not route_id in matched_closest[i] ) if i in matched_closest else True)
                                ):
                            text_handled_routes[text] = route
                            break
                    else:
                        text_handled_routes[text] = route
                        break

            if explain and explanations:
                for route_id,explanation in explanations[text].items():
                
                    route = settings.routing[route_id]
                    
                    if route.prediction_score_range:
                        
                        matched_prediction_score=False
                        prediciton_matches=[]
                        for prediction in predictions[text]:
                            matched=(  ((route.predicted_labels and prediction["label"] in route.predicted_labels ) or not  route.predicted_labels ) and  prediction["score"]>=(route.prediction_score_range.min or -1000) and  prediction["score"]<=(route.prediction_score_range.max or 1000)  ) 
                            if matched and  i in matched_closest and route_id in matched_closest[i] :
                                #if matched and closest to correctly predicted is also part of the matchin, we test also if label is the same as the clossest doc
                                matched_closest_items= [item for item in matched_closest[i][route_id] if prediction["label"] in item["labels"] ] 
                            else:
                                matched_closest_items = []
                            prediciton_matches.append(PredictionMatchExplanation(
                                prediction=prediction,
                                matched= matched and len(matched_closest_items)>0
                            ))
                            
                            matched_prediction_score=matched_prediction_score or matched
                        
                        if matched_closest_items:
                            explanation.matched_similar_examples=  [SimilarExample(
                                        text=item["text"] , 
                                        score=item["score"],
                                        labels=item["labels"], 
                                        correctly_predicted=item["correctly_predicted"])   for item in matched_closest_items]
                                     
                        explanation.matched_prediction=prediciton_matches
                        
                        explanation.matched = True if matched_prediction_score and explanation.matched_similar else False
                    else:
                        explanation.matched=True
                    
                    if  text_handled_routes.get(text) == route:
                        explanation.used=True


        
        to_backlog={}
        result:List[PredictedItem] = []
        
        for rec in data_to_send:
            if isinstance(rec,str):
                text = rec
            else:
                text = rec.text

            handling = text_handled_routes[text].handling if text in text_handled_routes else settings.default_handling
            if handling==RouteHandlingTypes.MANUAL:
                prediction_objects=None
            else:
                if settings.task_type == TASK_TYPES.MULTILABEL_TEXT_CLASSIFICATION:
                    prediction_objects=[Prediction(label=item["label"],score=item["score"]) for item in predictions[text] if item["score"]> settings.min_prediction_score] if text in predictions   else None
                elif settings.task_type == TASK_TYPES.TEXT_CLASSIFICATION:
                    top_prediction=max(predictions[text], key=lambda x: x["score"]) 
                    prediction_objects = [Prediction(label=top_prediction["label"],score=top_prediction["score"])]
            
            predictionItem = PredictedItem( predicted=prediction_objects, handling=handling)

            add_to_backlog=False
            if  handling==RouteHandlingTypes.MANUAL or handling==RouteHandlingTypes.MODEL_REVIEW:
                add_to_backlog=True

            reviewProjectId=settings.project_id
            if isinstance(rec, PredictionRequestRecord):
                predictionItem.key=rec.key
                if add_to_backlog:
                    backlog_payload = rec.dict()
                    if "reviewProjectId" in backlog_payload:
                        reviewProjectId = backlog_payload.pop("reviewProjectId") or settings.project_id # becasue reviewProjectId can be there but still null
            else:
                if add_to_backlog:
                    backlog_payload = {"text":rec}

            if add_to_backlog and query_vectors is not None:
                i = texts_to_handle.index(text)
                backlog_payload["vector"]=query_vectors[i].tolist()

            if add_to_backlog:
                if prediction_objects:
                    backlog_payload["predicted_labels"]=[pred.label for pred in prediction_objects ]
                if not reviewProjectId in to_backlog:
                    to_backlog[reviewProjectId]=[backlog_payload]
                else:
                    to_backlog[reviewProjectId].append(backlog_payload)


            if explain:
                predictionItem.explanations = list(explanations[text].values()) if text in explanations else None

            result.append(predictionItem)

        if not (test or TEST_MODE) and to_backlog:
            has_backlog_items=False
            for project_id, backlog_items in to_backlog.items():

                if backlog_items:
                    #print("to backlog: "+str([ item.get("key")  or f'text:{item.get("text")}'  for item in backlog_items]))
                    has_backlog_items=True
                    self.backlog_queue.put((project_id, backlog_items))
            if has_backlog_items:
                background_tasks.add_task(PredictionModule.flush_send_to_backlog, self)
                


        return result
                

    def predict_answer(self, background_tasks:BackgroundTasks, questions:Union[List[str], List[PredictionRequestRecord]], top_k:int=1, model_name:Optional[str]=None, explain:Optional[bool]=None, test:Optional[bool]=False):
        if self.configuration_error:
            raise HTTPException(520,{"error":f"Configuration error: {self.configuration_error}"})
        # if not self.configurationClient.settings.models:
        #     raise HTTPException(520,{"error":"No models are defined for this node"})
        # if not model_name:
        #     model_name = self.configurationClient.settings.default_model or self.configurationClient.settings.models[0].model_name
        
        settings =  self.configurationClient.settings.get_model_settings(model_name or "") 
        
        uniqueTexts =  {text for text in questions} if isinstance(questions[0],str) else {rec.text for rec in questions}
        
     
        texts_to_handle = list(uniqueTexts)

        correctly_predicted_min_range = min((route.similarity_range.min for route in settings.routing if route.rule_type==RouteRuleType.TRUE_POSITIVES), default=None)

        query_vectors = self.models_cahce.get_similarity_model(settings.project_id, settings.similarity_model).encode(texts_to_handle, normalize_embeddings=True)
        
        closest = self.closest.get_closest(
                    settings.project_id, 
                    query_vectors=query_vectors,
                     #ignore min score if default handling is not manual because othervise we dont get any answer
                     min_score=correctly_predicted_min_range/100 if not explain and settings.default_handling== RouteHandlingTypes.MANUAL  else 0,  
                     select_fields=["id", "answer"] ,
                     take=top_k,
                     answered=True )
        
        unique_results={}
        for i, closest_correctly_predicted_for_doc in enumerate(closest):
            handling=settings.default_handling
            answer=None
            for route in settings.routing:
                matched_answer=[]
                handling=route.handling
                for closest_correctly_predicted in closest_correctly_predicted_for_doc:
                    if closest_correctly_predicted and closest_correctly_predicted["answer"]:                
                        if closest_correctly_predicted and closest_correctly_predicted["score"]*100>=route.similarity_range.min and closest_correctly_predicted["score"]*100<=route.similarity_range.max:
                            answer = closest_correctly_predicted["answer"]
                            matched_answer.append(Answer(answer=answer, score=closest_correctly_predicted["score"] )) 
                
                
                if matched_answer:
                    if handling!=RouteHandlingTypes.MANUAL :
                        unique_results[texts_to_handle[i]]=(
                            matched_answer,
                            handling 
                        )
                    else:
                        unique_results[texts_to_handle[i]]=(
                            [],
                            handling
                        )
                
            if texts_to_handle[i] not in unique_results:
                 unique_results[texts_to_handle[i]]=(
                            [] if handling==RouteHandlingTypes.MANUAL else [Answer(answer=closest_correctly_predicted["answer"], score=closest_correctly_predicted["score"]) for closest_correctly_predicted in closest_correctly_predicted_for_doc],
                            settings.default_handling
                        )
                
                
        predictions=[] 
        to_backlog={}
        has_backlog_items=False
        for rec in questions:
            if isinstance(rec,str):
                text = rec
            else:
                text = rec.text
            
            (answers, handling) = unique_results[text]

            predictionItem= PredictedItem(predicted=answers, handling=handling)
            predictions.append(predictionItem)
              

            if not test and handling==RouteHandlingTypes.MANUAL or handling==RouteHandlingTypes.MODEL_REVIEW:
                add_to_backlog=True
            else:
                add_to_backlog=False

            reviewProjectId=settings.project_id
            if isinstance(rec, PredictionRequestRecord):
                predictionItem.key=rec.key
                if add_to_backlog:
                    backlog_payload = rec.dict()
                    if "reviewProjectId" in backlog_payload:
                        reviewProjectId = backlog_payload.pop("reviewProjectId") or settings.project_id # becasue reviewProjectId can be there but still null
            else:
                if add_to_backlog:
                    backlog_payload = {"text":rec}

            if add_to_backlog and query_vectors is not None:
                i = texts_to_handle.index(text)
                backlog_payload["vector"]=query_vectors[i].tolist()

            if add_to_backlog:
                if not reviewProjectId in to_backlog:
                    to_backlog[reviewProjectId]=[backlog_payload]
                else:
                    to_backlog[reviewProjectId].append(backlog_payload)

        if not test and to_backlog:
            has_backlog_items=False
            for project_id, backlog_items in to_backlog.items():
                    has_backlog_items=True
                    self.backlog_queue.put((project_id, backlog_items))
        if has_backlog_items:
            background_tasks.add_task(PredictionModule.flush_send_to_backlog, self)

        return  predictions


    # def get_pipeline(self,  model_name_or_id:str, task_type:str )->Pipeline:
    #     cache_key=f"get_pipeline.{model_name_or_id}"
    #     if cache_key in self.memory_cache:
    #         return self.memory_cache[cache_key]

    #     print(f"get_pipeline: {model_name_or_id}")
    #     huggingface_pipeline_task= TASK_TYPES.get_options(task_type,OptionTypes.pipeline_task )
    #     pipeline_kwrgars= TASK_TYPES.get_options(task_type,OptionTypes.pipeline_kwargs )

    #     result_model_path= get_model_path(model_name_or_id)
    #     if pipeline_kwrgars is None:
    #         pipeline_kwrgars={}
    #         self.memory_cache[cache_key]= pipeline(task=huggingface_pipeline_task, model=result_model_path, **pipeline_kwrgars)   
            
    #     else:
    #         self.memory_cache[cache_key]= pipeline(task=huggingface_pipeline_task, model=result_model_path, **pipeline_kwrgars)   
        
    #     return self.memory_cache[cache_key]

    
    
    # def get_similarity_model(self,  model_name_or_id:str ):
    #     cache_key=f"get_similarity_model.{model_name_or_id}"
    #     if cache_key in self.memory_cache:
    #         return self.memory_cache[cache_key] 
    #     print(f"get_similarity_model: {model_name_or_id}")
    #     result_model_path= self.models_paths[model_name_or_id]        
    #     self.memory_cache[cache_key] =  SentenceTransformer(result_model_path)
    #     return self.memory_cache[cache_key]

    async def flush_send_to_backlog(self):
        
        while True:
            try:
                (project_id, docs) = self.backlog_queue.get(block=False)
            except persistqueue.Empty:
                break

            try:
                res = self.labelatorio_client.documents.add_documents(project_id, docs, upsert=True)

                error_queue.retry_errors(
                    lambda data:  self.labelatorio_client.documents.add_documents(data["project_id"], data["docs"], upsert=True), 
                    max_retry_counter=1
                    )

            except Exception as ex:
                error_queue.put(docs=docs,project_id=project_id,error_msg=str(ex))
                logging.exception(ex)
        
        

class TemporaryPredictionModule(PredictionModule):

    def __init__(self, model_setttings:ModelSettings) -> None:
        self.settings=NodeSettings(default_model=model_setttings.model_name, models=[model_setttings], authorization=None)
        super().__init__(None)

    





class ClosestNeighbourEndpointGroup(EndpointGroup[dict]):
    """
    Special endpoint extension group dedicated to this usecase
    """

    def __init__(self, client: Client) -> None:
        super().__init__(client)    

    def get_closest(self,project_id, query_vectors, select_fields:List, min_score, 
            correctly_predicted=None,
            answered=None,
            take=5,
            ) -> List[dict]:
        """Get project by it's id

        Args:
            project_id (str): uuid of the project

        Returns:
            data_model.Project
        """

        if correctly_predicted:
            if "labels" not in select_fields:
                select_fields.append("labels")
            if "predicted_labels" not in select_fields:
                select_fields.append("predicted_labels")
            filter={"min_score":min_score, "labels":"!null"}
        else:
            if "id" not in select_fields:
                select_fields.append("id")
            filter={"min_score":min_score}
        
        if answered:
            filter["answer"] ="!null"

        payload = {
            "query_vectors":[list( float(s) for s in vec) for vec in query_vectors],
            "select_fields":select_fields,
            "filter":filter,
            "take":take
        }
        # if correctly_predicted:
        #     payload["filter"]={
        #         "false_negatives":"null", 
        #         "false_positives":"null",
        #     }
        result= self._call_endpoint("POST", f"/projects/{project_id}/doc/closest", body=payload, entityClass=dict)
        final_result=[]
        for subresult in result:
            if correctly_predicted and result:
                if subresult:
                    for rec in subresult: 
                        #todo - handle cases when predictions wasnt apllied yet
                        rec["correctly_predicted"] = set(rec["labels"])==set(rec["predicted_labels"]) if "predicted_labels" in rec and rec["predicted_labels" ] else None
                    final_result.append(subresult)
                else:
                    final_result.append(None)
                # correct_res = [rec for rec in subresult if rec["labels"]==rec["predicted_labels"]]
                # if len(correct_res)==len(subresult) and len(subresult)>0:
                #     final_result.append( correct_res[0] )
                # else:
                #     final_result.append(None)
            else:
                final_result.append(subresult)
        return final_result
        
