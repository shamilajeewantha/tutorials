
https://fastapi.tiangolo.com/deployment/docker/


docker build -t radshamila/fastapi-sample-app .

docker run -d --name fastapi-sample-app-container -p 8000:8000 radshamila/fastapi-sample-app
docker run -it --name fastapi-sample-app-container -p 8000:8000 -p 27017:27017 radshamila/fastapi-sample-app


http://127.0.0.1/

http://127.0.0.1/items/5?q=somequery

http://127.0.0.1/docs


docker push radshamila/fastapi-sample-app



docker network create fastapi-network
docker run -d --name mongodb-container --network fastapi-network -p 27017:27017 mongodb/mongodb-community-server
docker run -d --name fastapi-sample-app-container --network fastapi-network -p 8000:8000 radshamila/fastapi-sample-app
