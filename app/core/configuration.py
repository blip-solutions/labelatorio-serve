import functools
import logging
import os
from typing import Callable, Optional, Union
from labelatorio.client import EndpointGroup
from ..models.configuration import NodeSettings
from labelatorio import Client
from ..config import NODE_NAME, LABELATORIO_API_TOKEN, LABELATORIO_URL
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

    def ping(self,node_name:str, status:Optional[str]=None, message:str=None, current_version:str=None)  -> Union[dict, None]:
        """Pings Labelator.io and retrieve new config if available
        """
        body={
            "status":status,
            "message":message
            } 
        if current_version:
            body["version"]=current_version
            
        try:
            return self._call_endpoint("POST", f"/serving/nodes-settings/{node_name}/ping", body=body , entityClass=dict)
        except Exception as ex:
            logging.exception(f"Unable to connect: Error during ping")
            raise ex


class NodeConfigurationClient:
    def __init__(self, node_name:str) -> None:

        self.labelatorio_client = Client(LABELATORIO_API_TOKEN, url=LABELATORIO_URL)
        self.nodeSettingsEndpoint = NodeSettingsEndpointGroup(self.labelatorio_client)
        self.node_name = node_name
        self.settings=None
        self.change_handlers=[]
        self.configuration_error=None
        try:
            self.refresh()
            self.ping()  # we need this so heartbeat would be updated on server side
        except Exception as ex:
            #this happens in startup... we dont want this to fail,because its probably connection issue... retry try again on next ping
            self.configuration_error=str(ex)
            logging.exception("Error during client initialization")

    def refresh(self):
        """Refresh the config even when no change was made on server side
        """
        self.settings=self.nodeSettingsEndpoint.get(self.node_name)
        for handler in self.change_handlers:
            handler()
        self.configuration_error=None
        return self.settings

    def ping(self, status=None, message=None)->bool:
        """Refresh the config even when no change was made on server side
        """
        if not status and not message and self.configuration_error:
            # if the ping doen't contain any particular message and status but we have configuration error, lets send that
            status=NodeStatusTypes.ERROR
            message=self.configuration_error

        if status == NodeStatusTypes.ERROR:
            logging.error("PING:" + message or self.configuration_error or "Unknown error")
        elif message:
            logging.info("PING:" + message)

        current_version =  self.settings.version if self.settings else None
        try:
            reponseData = self.nodeSettingsEndpoint.ping(self.node_name, status, message, current_version)
        except Exception as ex:
            return False

        try:
            if reponseData:
                newSettings = NodeSettings(**reponseData)
            else:
                newSettings=None
            if newSettings and not status==NodeStatusTypes.UPDATING: 
                #if we have new settings but not already in update pending state
                
                self.settings = newSettings
                for handler in self.change_handlers:
                    handler()
                
                    
                    
            return True # sucessful ping
        except Exception as ex:
            logging.exception("Error when processing the node settings")
            self.configuration_error=str(ex)
            return False
        
        
    def on_config_change(self, handler: Callable):
        self.change_handlers.append(handler)



configuration_client = NodeConfigurationClient(NODE_NAME)


