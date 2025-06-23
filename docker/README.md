https://docs.docker.com/build/concepts/dockerfile/

https://docs.docker.com/docker-hub/quickstart/

https://docs.docker.com/build/building/best-practices/

docker build -t radshamila/nginx-custom .

then use docker run


uses ubuntu dockerfile instead of the default Dockerfile.

docker build -t flask:latest --file ubuntu.Dockerfile .