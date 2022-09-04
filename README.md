# Labelator.io serving 

Container for serving Labelator.io models at self-hosted nodes


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

## Build

Build a docker container:
docker build -t labelatorio-serve  --platform linux/amd64 .


## Configuration options

- NODE_NAME [required] - Name of the node (used to access configuration)
- LABELATORIO_API_TOKEN [required] - API token of user that is used to access to labelator.io
  

- MAX_CONNECTIONS_COUNT [optional] - uvicorn max connections settings
- MIN_CONNECTIONS_COUNT [optional] - uvicorn max connections settings
- ROOT_PATH [optional] - Root path for serving behing reverse proxy, when a subpath is added to URL
- ALLOWED_HOSTS [optional] - CORS settings
- SERVICE_ACCESS_TOKEN [optional] - for internal use, token that allows you force refresh configuration
- LABELATORIO_URL [optional]

### Volumes
Mounting volumes is not nescesary since no data are beeing permanently saved in the container, however mounting `app/models_cache` folder might be usefull, since you optimize startup time on recreation

## Run with docker:


```sh

$ docker run --rm   \
                  -p 8080:80 \
                  -e NODE_NAME=[node_name] \
                  -e LABELATORIO_API_TOKEN={your_api_token} \
                  --platform linux/amd64 \
                  labelatorio-serve:latest


```





