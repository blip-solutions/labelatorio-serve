from typing import List,Optional, Dict, Any
from pydantic import BaseModel


class FloatRange(BaseModel):
    max:float
    min:float
    

class RoutingSetting(BaseModel):
    rule_type:str
    handling:str #manual | model-review | model-auto
    anchors:Optional[List[str]]=None
    similarity_range:Optional[FloatRange] = None
    predicted_labels:Optional[List[str]] = None
    prediction_score_range:Optional[FloatRange] = None
    name:Optional[str]=None
    regex:Optional[str]=None



class ModelSettings(BaseModel):
    project_id:str
    model_name:str
    similarity_model:Optional[str]=None
    task_type:str
    routing:Optional[List[RoutingSetting]]=None
    default_handling:str="model-auto"
    min_prediction_score:float = 0.5


class OidcSettings(BaseModel):
    issuer:str
    client_id:Optional[str]
    audience:Optional[str]
    base_authorization_server_uri:Optional[str]
    signature_cache_ttl:Optional[int]
    

class NodeAuthorization(BaseModel):
    enable_public_access:bool =False
    auth_method:Optional[str] ="API_KEY" #API_KEY|OIDC
    api_key:Optional[str] = None
    oidc:Optional[OidcSettings] = None


class NodeSettings(BaseModel):
    default_model:Optional[str]=None
    models:List[ModelSettings]
    authorization:Optional[NodeAuthorization]
    extra:Optional[Dict[str,Any]] = None
    version:Optional[str] = None

    def get_model_settings(self, model_name) -> ModelSettings:
        return next((model for model in self.models if model.model_name==model_name), None)


    