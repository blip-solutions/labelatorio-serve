import functools
import time
from app.config import MAX_MODELS_IN_CACHE
from transformers.pipelines import Pipeline
from sentence_transformers import SentenceTransformer
import os
from transformers import pipeline
from huggingface_hub import snapshot_download as hf_download
from labelatorio import Client
import gc


MODEL_CACHE_SIZE=1

MODELS_CACHE_PATH = "models_cache"

def get_model_path(model_name_or_id:str):
        return os.path.join(MODELS_CACHE_PATH, "models--" + model_name_or_id.replace("/","--"))

WAIT_HANDLE={}
temp_mem_chache={} #this is just to fool lru_cache


def concurent_lru_cache(maxsize: int = MAX_MODELS_IN_CACHE or 2):
    def wrapper_cache(func):
        func = functools.lru_cache(maxsize=maxsize)(func)
        

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            cache_key=",".join(str(arg) for arg in args) + "," + ",".join(f"{k}:{v}" for k,v in kwargs.items())
            if temp_mem_chache.get(cache_key)==WAIT_HANDLE:
                for i in range(60):
                    time.sleep(1)
                    if temp_mem_chache.get(cache_key)!=WAIT_HANDLE:
                        break
            temp_mem_chache[cache_key]=WAIT_HANDLE
            res= func(*args, **kwargs)
            temp_mem_chache.pop(cache_key)
            return res


        return wrapped_func

    return wrapper_cache


class ModelsCache():

    def __init__(self, labelatorio_client:Client) -> None:
        self.models_paths={}
        self.labelatorio_client=labelatorio_client
    
    def get_pipeline(self, project_id,  model_name_or_id:str, task_type:str )->Pipeline:
        return self._get_pipeline(project_id, model_name_or_id, task_type)

    @concurent_lru_cache()
    def _get_pipeline(self, project_id,  model_name_or_id:str, task_type:str )->Pipeline:

        print(f"get_pipeline: {model_name_or_id}")
        huggingface_pipeline_task= TASK_TYPES.get_options(task_type,OptionTypes.pipeline_task )
        pipeline_kwrgars= TASK_TYPES.get_options(task_type,OptionTypes.pipeline_kwargs )

        result_model_path= get_model_path(model_name_or_id)
        
        self.predownload_model(project_id, model_name_or_id)

        if pipeline_kwrgars is None:
            pipeline_kwrgars={}
            return pipeline(task=huggingface_pipeline_task, model=result_model_path, **pipeline_kwrgars)   
            
        else:
            return pipeline(task=huggingface_pipeline_task, model=result_model_path, **pipeline_kwrgars)   
        
       

    @concurent_lru_cache()
    def get_similarity_model(self, project_id, model_name_or_id:str ):

        print(f"get_similarity_model: {model_name_or_id}")
        self.predownload_model(project_id, model_name_or_id)

        result_model_path= self.models_paths[model_name_or_id]        
        return SentenceTransformer(result_model_path)
     

    def predownload_model(self, project_id, model_name):
        model_path = get_model_path(model_name)
        if not os.path.exists(model_path):
            print(f"predownload_model {model_name}")
            if self.labelatorio_client.models.get_info(model_name):
                self.models_paths[model_name] =model_path
                self.labelatorio_client.models.download(project_id=project_id, model_name_or_id=model_name, target_path=model_path)
            else:
                
                model_path = hf_download(repo_id=model_name, cache_dir=MODELS_CACHE_PATH)
                
                print(f"download {model_name} finished")
        

        snapshot_path = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshot_path):
            snapshosts= os.listdir(snapshot_path)
            if snapshosts:
                self.models_paths[model_name] =os.path.join( snapshot_path, snapshosts[0])
            else:
                #backup plan...
                print(f"no snapshot found... use HF native methogs")
                from transformers import AutoConfig, AutoModel
                model = AutoModel.from_pretrained(model_name,  cache_dir=MODELS_CACHE_PATH)
                del model
                gc.collect()
                
        else:
            self.models_paths[model_name] =model_path












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
