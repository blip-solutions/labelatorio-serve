# Labelator.io Node serving
Container for serving (inference) Labelator.io models as self-hosted nodes


## Dev envrioment setup

Create a venv
then install

```sh
$ pip install -r requirements.txt
```

or using pipenv

```sh
$ pipenv install
```

## Build from source

Build a docker container:
```
docker build -t labelatorio-serve  --platform linux/amd64 .
```



## Run with docker:
```sh

$ docker run --rm   \
                  -p 8080:80 \
                  -e NODE_NAME=[node_name] \
                  -e LABELATORIO_API_TOKEN={your_api_token} \
                  --platform linux/amd64 \
                  blipsolutions/labelatorio-serving-node:latest


```

## Configuration options

- NODE_NAME [required] - Name of the node (used to access configuration)
- LABELATORIO_API_TOKEN [required] - API token of the user that is used to access Labelator.io
  

- MAX_CONNECTIONS_COUNT [optional] - uvicorn max connections settings
- MIN_CONNECTIONS_COUNT [optional] - uvicorn max connections settings
- ROOT_PATH [optional] - Root path for serving behind reverse proxy, when a subpath is added to URL
- ALLOWED_HOSTS [optional] - CORS settings
- SERVICE_ACCESS_TOKEN [optional] - for internal use, a token that allows you to force refresh configuration
- LABELATORIO_URL [optional]

### Volumes
Mounting volumes is not necessary since no data are being permanently saved in the container, however mounting `app/models_cache` folder might be useful, since you optimize startup time on recreation (models that have been downloaded in the past won't be downloaded again)

Also mounting `/queues` would preserve data yet to be sent to Labelator.io for review (might be useful if unexpected shutdown occurred in the middle of the operation)
