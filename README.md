```
export HUGGINGFACE_TOKEN=<my huggingface token>
```
```
docker build -t <image-name> --build-arg HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN .
```
