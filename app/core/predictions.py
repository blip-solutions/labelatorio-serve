from datetime import datetime
from email.policy import default
import functools
import os
from pickle import FALSE
from datetime import timezone
from typing import List, Optional, Union
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download as hf_download
import torch
import numpy as np
from labelatorio import Client
import shutil
from functools import lru_cache
from fastapi import HTTPException, BackgroundTasks
from labelatorio.client import EndpointGroup
import asyncio
import gc
from transformers.pipelines import Pipeline
from app.models.configuration import RoutingSetting
from ..models.requests import PredictionRequestRecord
from ..models.responses import PredictedItem, Prediction, PredictionMatchExplanation, RouteExplanation, SimilarExample
#from ..models.responses import Answer
from  .configuration import NodeConfigurationClient 
from  .contants import NodeStatusTypes, RouteRuleType, RouteHandlingTypes
import persistqueue
import logging
class OptionTypes:
    pipeline_task = "pipeline_task"
    pipeline_kwargs =  "pipeline_kwargs"
    is_multilabel="is_multilabel"
    labels_preddefined="labels_preddefined"
    labels_per_token="labels_per_token"

class PipelineTasks:
    audio_classification = "audio-classification"
    automatic_speech_recognition = "automatic-speech-recognition"
    conversational = "conversational"
    feature_extraction = "feature-extraction"
    fill_mask = "fill-mask"
    image_classification = "image-classification"
    question_answering = "question-answering"
    table_question_answering = "table-question-answering"
    text2text_generation = "text2text-generation"
    text_classification = "text-classification" 
    text_generation = "text-generation"
    token_classification = "token-classification"
    ner  = "ner" # this alias to token-classification
    translation = "translation"
    translation_xx_to_yy = "translation_xx_to_yy"
    summarization = "summarization"
    zero_shot_classification = "zero-shot-classification"

def get_model_path(model_name_or_id:str):
        return os.path.join(MODELS_CACHE_PATH, model_name_or_id.replace("/","_"))

class TASK_TYPES:
    NER="NER"
    MULTILABEL_TEXT_CLASSIFICATION="MultiLabelTextClassification"
    TEXT_CLASSIFICATION="TextClassification"
    TEXT_SIMILARITY="TextSimilarity"
    QUESTION_ANWERING="QuestionAnswering"
    TEXT_SCORING="TextScoring"

    def get_options(task_type:str, option_type:str=None):
        if task_type==TASK_TYPES.MULTILABEL_TEXT_CLASSIFICATION:
            options={
                OptionTypes.pipeline_task:PipelineTasks.text_classification,
                OptionTypes.pipeline_kwargs:{"return_all_scores":True},
                OptionTypes.is_multilabel:True,
                OptionTypes.labels_preddefined:True,
                OptionTypes.labels_per_token:False
            }
        elif task_type==TASK_TYPES.TEXT_CLASSIFICATION:
            options={
                OptionTypes.pipeline_task:PipelineTasks.text_classification,
                OptionTypes.pipeline_kwargs:{"return_all_scores":True},
                OptionTypes.is_multilabel:False,
                OptionTypes.labels_preddefined:True,
                OptionTypes.labels_per_token:False
            }
        elif task_type==TASK_TYPES.TEXT_SIMILARITY:
            options={
                OptionTypes.pipeline_task:None,
                OptionTypes.pipeline_kwargs:None,
                OptionTypes.is_multilabel:False,
                OptionTypes.labels_preddefined:False,
                OptionTypes.labels_per_token:False
            }
        elif task_type==TASK_TYPES.QUESTION_ANWERING:
            options={
                OptionTypes.pipeline_task:PipelineTasks.question_answering,
                OptionTypes.pipeline_kwargs:None,
                OptionTypes.is_multilabel:False,
                OptionTypes.labels_preddefined:False,
                OptionTypes.labels_per_token:False
            }
        elif task_type==TASK_TYPES.NER:
            options={
                OptionTypes.pipeline_task:PipelineTasks.ner,
                OptionTypes.pipeline_kwargs:None,
                OptionTypes.is_multilabel:True,
                OptionTypes.labels_preddefined:True,
                OptionTypes.labels_per_token:True
            }
        #  elif task_type==TASK_TYPES.TEXT_SCORING:
        #     options={
        #         OptionTypes.pipeline_task:None,
        #         OptionTypes.pipeline_kwargs:None,
        #         OptionTypes.is_multilabel:True,
        #         OptionTypes.labels_preddefined:True,
        #         OptionTypes.labels_per_token:True
        #     }
        else:
            raise Exception(f"Option {task_type} not supported")
        
        if option_type:
            return options[option_type]
        else:
            return options



MODEL_CACHE_SIZE=1

MODELS_CACHE_PATH = "models_cache"



class PredictionModule:
    def __init__(self, configurationClient:NodeConfigurationClient ) -> None:
        if (torch.cuda.is_available()):
            print("CUDA availible... using GPU")
        
        self.configurationClient=configurationClient

        self.backlog_queue= persistqueue.SQLiteQueue('queues/backlog_queue', auto_commit=True, multithreading=True)
        self.errors_queue= persistqueue.SQLiteQueue('queues/error_queue.db', auto_commit=True, multithreading=True)

        self.labelatorio_client = self.configurationClient.labelatorio_client
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
        self.configurationClient.ping(status=NodeStatusTypes.UPDATING)
        self.model_anchor_vectors={}
        old_paths = self.models_paths
        self.models_paths={}

        del self.memory_cache
        gc.collect()
        self.memory_cache={}
        try:
            if self.configurationClient.settings and self.configurationClient.settings.models:
                self.default_model=self.configurationClient.settings.default_model or self.configurationClient.settings.models[0].model_name 
                for modelConfig in self.configurationClient.settings.models:
                    
                    self.predownload_model(modelConfig.project_id,modelConfig.model_name)
                    
                    
                    if modelConfig.similarity_model:
                        self.predownload_model(modelConfig.project_id,modelConfig.similarity_model)
                    elif (modelConfig.routing or modelConfig.default_handling!=RouteHandlingTypes.MODEL_AUTO) :
                        raise Exception(f"{modelConfig.model_name} :Invalid configuration. No similarity model defined. This is allow only for full auto mode without routing")
                    

                    print("preparing anchors")
                    for route_id, route in enumerate(modelConfig.routing):
                        if route.rule_type==RouteRuleType.ANCHORS:
                            similarity_model=self.get_similarity_model(modelConfig.similarity_model)
                            if self.model_anchor_vectors.get(modelConfig.model_name) is None:
                                self.model_anchor_vectors[modelConfig.model_name]={}
                            self.model_anchor_vectors[modelConfig.model_name][route_id] =similarity_model.encode(route.anchors,  normalize_embeddings=True)
                
                if self.default_model: #preload default models into memory
                    self.get_pipeline(self.default_model, self.configurationClient.settings.get_model(self.default_model).task_type)  #preload into memory
                    default_similarity_model = next((model.similarity_model for model in self.configurationClient.settings.models if model.model_name==self.default_model),None)
                    if default_similarity_model:
                        self.get_similarity_model(default_similarity_model)
            self.configurationClient.ping(status=NodeStatusTypes.READY)
            print("Reconfiguraiton ready")
            #TODO: Delete old paths to save disk space?>>  old_paths
        except Exception as ex:
            self.configuration_error=repr(ex)
            self.configurationClient.ping(status=NodeStatusTypes.ERROR)
                        


    def predownload_model(self, project_id, model_name):
         if not os.path.exists(get_model_path(model_name)):
            print(f"predownload_model {model_name}")
            if self.labelatorio_client.models.get_info(model_name):
                self.models_paths[model_name] = get_model_path(model_name)
                self.labelatorio_client.models.download(project_id=project_id, model_name_or_id=model_name, target_path=get_model_path(model_name))
            else:
                
                self.models_paths[model_name] = hf_download(repo_id=model_name, cache_dir=MODELS_CACHE_PATH)
            print(f"download {model_name} finished")


    def predict_labels(self, background_tasks:BackgroundTasks, data_to_send:Union[List[str], List[PredictionRequestRecord]], model_name:Optional[str], explain:Optional[bool]=None, test:Optional[bool]=False)->PredictedItem:

        if self.configuration_error:
            raise HTTPException(520,{"error":f"Configuration error: {self.configuration_error}"})
        if not self.configurationClient.settings.models:
            raise HTTPException(520,{"error":"No models are defined for this node"})
        if not model_name:
            model_name = self.configurationClient.settings.default_model or self.configurationClient.settings.models[0].model_name
        
        pipe=None
        settings =  self.configurationClient.settings.get_model(model_name)
        if not settings:
            raise HTTPException(520,{"error":f"Model {model_name} is not deployed on this node"})

        if not data_to_send:
            return None

        uniqueTexts =  {text for text in data_to_send} if isinstance(data_to_send[0],str) else {rec.text for rec in data_to_send}
        


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
            
            query_vectors = self.get_similarity_model(settings.similarity_model).encode(texts_to_handle, normalize_embeddings=True)
            
            closest=None
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

                matched_routes:List[RoutingSetting]=[]
                
                for route_id, route in enumerate(settings.routing) :

                    if route.rule_type==RouteRuleType.TRUE_POSITIVES:
                        
                        if closest_correctly_predicted and closest_correctly_predicted["correctly_predicted"] is None:
                            # if no prediction was made yet we need to run it our self
                            pipe = self.get_pipeline(model_name, settings.task_type) 
                            predictions= set(p["label"] for p in pipe(closest_correctly_predicted["text"], padding=True, truncation=True)[0] if p["score"]>0.5)
                            if set(closest_correctly_predicted["labels"])==predictions:
                                closest_correctly_predicted["correctly_predicted"]=True

                        if closest_correctly_predicted and closest_correctly_predicted["correctly_predicted"] and closest_correctly_predicted["score"]*100>=route.similarity_range.min and closest_correctly_predicted["score"]*100<=route.similarity_range.max:
                            matched_routes.append( route)

                        if explain:
                            explanations[text2handle][route_id]=RouteExplanation(
                                    route_id=route_id, 
                                    route_type=route.rule_type, 
                                    route_handling=route.handling,
                                    matched=False,
                                    used=False,
                                    matched_prediction=None,
                                    matched_similar=True if route in matched_routes else False, 
                                    matched_similar_example=SimilarExample(
                                        text=closest_correctly_predicted["text"] , 
                                        score=closest_correctly_predicted["score"],
                                        labels=closest_correctly_predicted["labels"], 
                                        correctly_predicted=closest_correctly_predicted["correctly_predicted"]  
                                     ) if closest_correctly_predicted else None
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
                                matched_routes.append(route)
                                break

                        if explain:
                            explanations[text2handle][route_id]=RouteExplanation(
                                    route_id=route_id, 
                                    route_type=route.rule_type, 
                                    route_handling=route.handling,
                                    matched=False,
                                    used=False,
                                    matched_prediction=None,
                                    matched_similar=True if route in matched_routes else False, 
                                    matched_similar_example=SimilarExample(text=route.anchors[max_sim_anchor_index], score=max_similarity) if max_sim_anchor_index is not None else None
                                 )
                                
                            
                    else:
                        raise HTTPException(500,{"error":f"Invalid  routing rule type '{route.rule_type}'"})

                texts_routes_matches[i] = matched_routes
                  
        if RouteHandlingTypes.should_predict(settings.default_handling) or explain:
            texts_to_predict= texts_to_handle
        else:
            texts_to_predict = [text for  text, matched_routes in zip(texts_to_handle,texts_routes_matches) if  any(1 for route in matched_routes if RouteHandlingTypes.should_predict(route.handling) ) ]
        if texts_to_predict:
            if pipe is None:
                pipe = self.get_pipeline(model_name, settings.task_type) 
            predictions= {text:doc_predictions  for text, doc_predictions in zip(texts_to_predict,pipe(texts_to_predict, padding=True, truncation=True))}
        else:
            predictions={}

        text_handled_routes = { }
        
        for i, (text, matched_routes) in  enumerate(zip(texts_to_handle,texts_routes_matches)):
            if matched_routes:
                for route in matched_routes:
                    if route.prediction_score_range :
                        if route.predicted_labels and text in predictions:
                            if closest:
                                predictions_to_test= [prediction for prediction in predictions[text] if  prediction["label"] in route.predicted_labels ]
                            else:
                                predictions_to_test= [prediction for prediction in predictions[text] if prediction["label"] in route.predicted_labels]
                        else:
                            predictions_to_test=predictions.get(text) or []

                        if any(1 for prediction in predictions_to_test if prediction["score"]>=(route.prediction_score_range.min or -1000) and  prediction["score"]<=(route.prediction_score_range.max or 1000)  and  ((closest and prediction["label"] in closest[i]["labels"]) or not closest) ):
                            text_handled_routes[text] = route
                            break
                    else:
                        text_handled_routes[text] = route
                        break

            if explain:
                for route_id,explanation in explanations[text].items():
                
                    route = settings.routing[route_id]
                    
                    if route.prediction_score_range:
                        
                        matched_prediction_score=False
                        prediciton_matches=[]
                        for prediction in predictions[text]:
                            matched=(  (prediction["label"] in route.predicted_labels or not  route.predicted_labels ) and  prediction["score"]>=(route.prediction_score_range.min or -1000) and  prediction["score"]<=(route.prediction_score_range.max or 1000)  ) if route.predicted_labels else False
                            if matched and closest:
                                #if matched and closses to correctly predicted is also part of the matchin, we test also if label is the same as the clossest doc
                                matched= prediction["label"] in closest[i]["labels"]
                            prediciton_matches.append(PredictionMatchExplanation(
                                prediction=prediction,
                                matched= matched
                            ))
                            
                            matched_prediction_score=matched_prediction_score or matched
                        
                        explanation.matched_prediction=prediciton_matches
                        explanation.matched = matched_prediction_score and explanation.matched_similar
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
                prediction_objects=[Prediction(label=item["label"],score=item["score"]) for item in predictions[text] if item["score"]> settings.min_prediction_score] if text in predictions   else None
            
            predictionItem = PredictedItem( predicted=prediction_objects, handling=handling)

            add_to_backlog=False
            if not test and handling==RouteHandlingTypes.MANUAL or handling==RouteHandlingTypes.MODEL_REVIEW:
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

        if not test and add_to_backlog:
            has_backlog_items=False
            for project_id, backlog_items in to_backlog.items():
                if data_to_send:
                    has_backlog_items=True
                    self.backlog_queue.put((project_id, backlog_items))
            if has_backlog_items:
                background_tasks.add_task(PredictionModule.flush_send_to_backlog, self)
                


        return result
                

    def get_answer(self, background_tasks:BackgroundTasks, questions:Union[List[str], List[PredictionRequestRecord]], model_name:Optional[str], explain:Optional[bool]=None, test:Optional[bool]=False):
        if self.configuration_error:
            raise HTTPException(520,{"error":f"Configuration error: {self.configuration_error}"})
        # if not self.configurationClient.settings.models:
        #     raise HTTPException(520,{"error":"No models are defined for this node"})
        # if not model_name:
        #     model_name = self.configurationClient.settings.default_model or self.configurationClient.settings.models[0].model_name
        
        settings =  self.configurationClient.settings.get_model(model_name or "") 
        
        uniqueTexts =  {text for text in questions} if isinstance(questions[0],str) else {rec.text for rec in questions}
        
     
        texts_to_handle = list(uniqueTexts)

        correctly_predicted_min_range = min((route.similarity_range.min for route in settings.routing if route.rule_type==RouteRuleType.TRUE_POSITIVES), default=None)

        query_vectors = self.get_similarity_model(settings.similarity_model).encode(texts_to_handle, normalize_embeddings=True)
        
        closest = self.closest.get_closest(
                    settings.project_id, 
                    query_vectors=query_vectors,
                     min_score=correctly_predicted_min_range/100 if not explain else 0, 
                     select_fields=["id", "answer"] ,
                     answered=True )
        
        unique_results={}
        for i, closest_correctly_predicted in enumerate(closest):
            handling=settings.default_handling
            answer=None
            if closest_correctly_predicted and closest_correctly_predicted["answer"]:
                answer=closest_correctly_predicted["answer"]
                for route in settings.routing:
                    if closest_correctly_predicted and closest_correctly_predicted["score"]*100>=route.similarity_range.min and closest_correctly_predicted["score"]*100<=route.similarity_range.max:
                        handling=route.handling
                        break
            
            
            unique_results[texts_to_handle[i]]=(
                    Answer(answer=answer, score=closest_correctly_predicted["score"] ) if answer and handling!=RouteHandlingTypes.MANUAL else None,
                    handling if answer else RouteHandlingTypes.MANUAL 
            )
                
        predictions=[] 
        to_backlog={}
        has_backlog_items=False
        for rec in questions:
            if isinstance(rec,str):
                text = rec
            else:
                text = rec.text
            
            (answer, handling) = unique_results[text]

            predictionItem= PredictedItem(predicted=answer, handling=handling)
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

        if not test and add_to_backlog:
            has_backlog_items=False
            for project_id, backlog_items in to_backlog.items():
                    has_backlog_items=True
                    self.backlog_queue.put((project_id, backlog_items))
        if has_backlog_items:
            background_tasks.add_task(PredictionModule.flush_send_to_backlog, self)

        return  predictions


    def get_pipeline(self,  model_name_or_id:str, task_type:str )->Pipeline:
        cache_key=f"get_pipeline.{model_name_or_id}"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        print(f"get_pipeline: {model_name_or_id}")
        huggingface_pipeline_task= TASK_TYPES.get_options(task_type,OptionTypes.pipeline_task )
        pipeline_kwrgars= TASK_TYPES.get_options(task_type,OptionTypes.pipeline_kwargs )

        result_model_path= get_model_path(model_name_or_id)
        if pipeline_kwrgars is None:
            pipeline_kwrgars={}
            self.memory_cache[cache_key]= pipeline(task=huggingface_pipeline_task, model=result_model_path, **pipeline_kwrgars)   
            
        else:
            self.memory_cache[cache_key]= pipeline(task=huggingface_pipeline_task, model=result_model_path, **pipeline_kwrgars)   
        
        return self.memory_cache[cache_key]

    
    
    def get_similarity_model(self,  model_name_or_id:str ):
        cache_key=f"get_similarity_model.{model_name_or_id}"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key] 
        print(f"get_similarity_model: {model_name_or_id}")
        result_model_path= self.models_paths[model_name_or_id]        
        self.memory_cache[cache_key] =  SentenceTransformer(result_model_path)
        return self.memory_cache[cache_key]

    async def flush_send_to_backlog(self):
        
        while True:
            try:
                (project_id, docs) = self.backlog_queue.get(block=False)
            except persistqueue.Empty:
                break

            try:
                #print(f"{project_id}>> {docs}")
                res = self.labelatorio_client.documents.add_documents(project_id, docs, upsert=True)
                #print(f"{project_id} << {res}")

            except Exception as ex:
                self.errors_queue.put({"docs":docs, "project_id":project_id, "error":str(ex), "counter":1, "timestamp":datetime.now(timezone.utc)})
                print(ex)
                logging.exception(ex)
        
        






class ClosestNeighbourEndpointGroup(EndpointGroup[dict]):
    """
    Special endpoint extension group dedicated to this usecase
    """

    def __init__(self, client: Client) -> None:
        super().__init__(client)    

    def get_closest(self,project_id, query_vectors, select_fields:List, min_score, 
            correctly_predicted=None,
            answered=None
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
            "take":1
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
                        final_result.append(rec)
                else:
                    final_result.append(None)
                # correct_res = [rec for rec in subresult if rec["labels"]==rec["predicted_labels"]]
                # if len(correct_res)==len(subresult) and len(subresult)>0:
                #     final_result.append( correct_res[0] )
                # else:
                #     final_result.append(None)
            else:
                final_result.append(next(iter(subresult), None))
        return final_result
        
