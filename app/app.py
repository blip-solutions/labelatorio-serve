from typing import List, Tuple
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,)
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
import os
import time
from fastapi_auth_middleware import AuthMiddleware

from .models.configuration import NodeAuthorization
from .core.predictions import PredictionModule
from .models.responses import Info
from  .config import ALLOWED_HOSTS, ROOT_PATH, SERVICE_ACCESS_TOKEN
from .core import router as core_router
from .utils.error_handlers import http_error_handler, http_422_error_handler
from .core.configuration import configuration_client
from fastapi import Request
from fastapi_utils.tasks import repeat_every
from starlette.authentication import  BaseUser, requires

from starlette.authentication import BaseUser
from typing import List, Optional, Tuple
from starlette.datastructures import Headers
from fastapi_oidc import get_auth


if  ROOT_PATH:
    docs_servers=[{"url": ROOT_PATH, "description": "hosted environment"}]
else:
    docs_servers=None

version="0.1." +  os.environ["BUILD_VERSION"] if "BUILD_VERSION" in os.environ else  "_dev_"

app = FastAPI(
    servers=docs_servers,
    root_path=ROOT_PATH,
    root_path_in_servers=False,
    title="Labelator.io Serving",
    description="Model hosting for Labelator.io models",
    version=version
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




app.state.prediction_module = PredictionModule(configuration_client)
configuration_client.on_config_change(lambda : app.state.prediction_module.reinitialize())
app.state.configuration_client = configuration_client

def build_auth():

    if (app.state.configuration_client.settings and 
        app.state.configuration_client.settings.authorization):
        auth_settings:NodeAuthorization = app.state.configuration_client.settings.authorization
        if auth_settings.enable_public_access:
            app.state.authenticate_user=lambda : ["authenticated", "control_access"] 
        if auth_settings.auth_method=="OIDC":
                

            try:
                auth = get_auth(**app.state.configuration_client.settings.authorization.oidc)
                def oidc_auth(headers):
                    if headers.get("Authorization"):
                        return  ["authenticated", "control_access"] if auth(headers.get("Authorization")) else None
                    elif SERVICE_ACCESS_TOKEN and headers.get("access_token"):
                        return  ["control_access"] if SERVICE_ACCESS_TOKEN==headers.get("access_token") else None
                        
                app.state.authenticate_user=oidc_auth
            except Exception as ex:
                def raiseEx(*_,**__):
                    raise HTTPException(500, f"Unable to build OIDC authentification from config: {ex}")
                app.state.authenticate_user= raiseEx
        elif auth_settings.auth_method=="API_KEY":
            def api_key_auth(headers):
                if auth_settings.api_key and headers.get("access_token")== auth_settings.api_key:
                    return ["authenticated", "control_access"] 
                elif SERVICE_ACCESS_TOKEN and headers.get("access_token"):
                   return  ["control_access"] if SERVICE_ACCESS_TOKEN==headers.get("access_token") else None

            app.state.authenticate_user= api_key_auth
    else:
        def no_auth(headers):
            if SERVICE_ACCESS_TOKEN and headers.get("access_token"):
                return  ["control_access"] if SERVICE_ACCESS_TOKEN==headers.get("access_token") else None

        app.state.authenticate_user= no_auth
                
build_auth()
configuration_client.on_config_change(build_auth)


def verify_authorization_header(headers: Headers) -> Tuple[List[str], BaseUser]: 
    
    if app.state.authenticate_user:
        scopes = app.state.authenticate_user(headers)
        if scopes:
            return scopes, BaseUser() 
        else:
            return [],None
    else:
        return [],None

        
app.add_middleware(AuthMiddleware, verify_header=verify_authorization_header)



@app.get("/")
async def root(request:Request):
    
    return Info(
        info="Labelator.io serving", 
        node_name=configuration_client.node_name,
        settings=configuration_client.settings if request.user else None, 
        version=version,
        root_path=request.scope.get("root_path")
        )
    


app.add_exception_handler(HTTPException, http_error_handler)
app.add_exception_handler(
    HTTP_422_UNPROCESSABLE_ENTITY, http_422_error_handler)

# add routers
app.include_router(core_router)



@app.post("/refresh")
@requires(['control_access'])
async def refresh(request:Request):
    print("refresh refresh")
    configuration_client.ping()

@app.on_event("startup")
@repeat_every(seconds=1*60)  # 1 minutes
def schedule():
    print("Scheduled ping")
    configuration_client.ping()

