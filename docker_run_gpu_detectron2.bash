# Para dar permisos al fichero: "sudo chmod +x Nombre_archivo.bash"
# Para lanzarlo ./Nombre_archivo.bash

docker build -t detectron2 .

docker run -it \
    --name detectron2 \
    --gpus all \
    --volume /mnt/docker/Alimente21/matadero/detectron2:/home/enzo \
    --volume /mnt/data/Alimente21/Datasets:/home/enzo/Datasets \
    --ipc=host \
    --user $(id -u):$(id -g) \
    detectron2 \
    bash

