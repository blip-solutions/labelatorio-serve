import functools
import logging
import os
from typing import Callable, Optional, Union
from labelatorio.client import EndpointGroup
from ..models.configuration import NodeSettings
from labelatorio import Client
from ..config import NODE_NAME, LABELATORIO_API_TOKEN, LABELATORIO_URL, SERVICE_ACCESS_TOKEN
from .contants import NodeStatusTypes


class NodeSettingsEndpointGroup(EndpointGroup[NodeSettings]):
    def __init__(self, client: Client) -> None:
        super().__init__(client)    

    def get(self,node_name:str)  -> NodeSettings:
        """Get node configuration
        """
        dictData = self._call_endpoint("GET", f"/serving/nodes-settings/{node_name}", entityClass=dict)
        if dictData:
            return NodeSettings(**dictData)

    def ping(self,node_name:str, status:Optional[str]=None, message:str=None)  -> Union[NodeSettings, None]:
        """Pings Labelator.io and retrieve new config if availible
        """
        body={
            "status":status,
            "message":message
            } 
        dictData = self._call_endpoint("POST", f"/serving/nodes-settings/{node_name}/ping", body=body , entityClass=dict)
        if dictData:
            return NodeSettings(**dictData)


class NodeConfigurationClient:
    def __init__(self, node_name:str) -> None:

        self.labelatorio_client = Client(LABELATORIO_API_TOKEN, url=LABELATORIO_URL)
        self.nodeSettingsEndpoint = NodeSettingsEndpointGroup(self.labelatorio_client)
        self.node_name = node_name
        self.change_handlers=[]
        try:
            self.refresh()
            self.ping()  # we need this so heartbeat would be updated on server side
        except Exception as ex:
            #this happanse in startup... we sont want this to fail,beacsue its probably connection issue... letry try again on next ping
            print("Error during client init !!!")
            print(ex)

    def refresh(self):
        """Refresh the config even when no change was made on server side
        """
        self.settings=self.nodeSettingsEndpoint.get(self.node_name)
        for handler in self.change_handlers:
            handler()
        return self.settings

    def ping(self, status=None, message=None):
        """Refresh the config even when no change was made on server side
        """
        try:
            
            reponse = self.nodeSettingsEndpoint.ping(self.node_name, status, message)
            if reponse and not status==NodeStatusTypes.UPDATING:
                if (self.settings and reponse.json()!=self.settings.json()) or self.settings is None:
                    self.settings = reponse
                    for handler in self.change_handlers:
                        handler()
                    
                
        except Exception as ex:
            logging.error(f"Error during ping:{ex}")
            pass
        
        
    def on_config_change(self, handler: Callable):
        self.change_handlers.append(handler)



configuration_client = NodeConfigurationClient(NODE_NAME)


