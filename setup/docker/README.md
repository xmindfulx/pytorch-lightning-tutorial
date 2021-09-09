# Container Setup

The goal of this writeup is to provide instructions for building a development environment via docker container

## Using Docker Containers (Basic Workflow)

[General Overview](https://docs.docker.com/engine/docker-overview/)

Build docker **file** --> Run docker **image** --> Enter docker **container**

## Build Docker File

Docker containers begin with building a dockerfile. Building a dockerfile results in a docker **image**

``docker build --rm -t mindful/tutorial <path_file>``

- `--rm` : removes any dangling or intermediate image builds
- `-t` : name for your image. This is useful to set to a unique and familiar name. (e.g., `mindful/tutorial`)
- `<path_file>` : path to dockerfile (e.g., `.` would look inside the current directory)

## Run Docker Image

Building a docker file results in a docker image. Running this image spawns a docker **container**

``docker run --name tutorial -d -it -v <path_data>:/develop/data -v <path_results>:/develop/results -v <path_code>:/develop/code mindful/tutorial``

- `--name` : gives container a easy rememerable name
- `-d` : runs container in detached mode (in background)
- `-it`:  allows the container to be enterable with terminal shell
- `-v`: allows the container to mount a path from container host. 
   - `<path_data>:/develop/data` : dataset folder on host maps to `/develop/data` inside container
   - `<path_results>:/develop/results` : results folder on host maps to `/develop/results` inside container
   - `<path_code>:/develop/code` : results folder on host maps to `/develop/code` inside container

Addtional useful flags for larger experiments:

- `--gpus all` : gives container access to all GPUs
- `-p <port_number>:<port_number>`: exposes port from container and binds it to host. This is useful for things like jupyter notebook.
- `--ipc=host` allows the container to access more system share memory. This is useful for larger scale experiments.

## Enter Docker Container

Running a docker image results in an instane of that image or a docker container. 

``docker attach tutorial``

## Addtional Docker Commands 

The following commands are useful docker commands for different situations. 

### Show Docker Images 

```docker images```

- Useful to see all currently created docker images
- You should see the minful/tutorial image

```docker images | grep mindful```

- You can also specify a subset of docker images

### Show Docker Containers

```docker ps```

- Useful to see all currently running containers
- You should see minful/test running

```docker ps -a```

- Shows stopped all containers, even stopped ones

### Exit Docker Container (Without Closing)

```Ctrl+p --> Ctrl+q```

- First press Ctrl+P and afterwards press Ctrl+Q (sequentially, not simultaneously)

### Exit Docker Container (With Closing || Terminating)

```docker attach $(docker ps | grep mindful/tutorial | awk '{print $1}')```

```exit```

- Enter the container and exit from the inside

### Kill Docker Container From Docker Image Name

```docker kill $(docker ps | grep mindful/tutorial | awk '{print $1}')```

- Allows you to kill a running docker container. 

### Kill Docker Image

```docker rmi -f mindful/tutorial ```

- Allows you to kill a docker image. 

### Remove Stopped Containers

```docker ps -a``` 

- To show all stopped containers (remember each container has unique name)
- Important b/c If you want to reuse a used --name for another container from the above image, you must remove the previous stopped container

```docker rm tutorial```

- Remove the stopped container to free up the name again
