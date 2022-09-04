class RouteHandlingTypes:
    MANUAL="manual"
    MODEL_REVIEW="model-review"
    MODEL_AUTO="model-auto"

    def should_predict(handling_type:str):
        return handling_type == RouteHandlingTypes.MODEL_AUTO or handling_type == RouteHandlingTypes.MODEL_REVIEW 



class RouteRuleType:
    ANCHORS="anchors"
    TRUE_POSITIVES="true-predictions"

class NodeStatusTypes:
    PENDING="PENDING"
    UPDATING="UPDATING"
    READY="READY"
    CONFIG_OUT_OF_DATE="CONFIG_OUT_OF_DATE"
    OFFLINE="OFFLINE"
    ERROR="ERROR"