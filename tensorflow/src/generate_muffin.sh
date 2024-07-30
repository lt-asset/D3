#!/bin/bash

docker_name=muffin
if [ $# -gt 0 ]; then
    docker_name=$1
fi

# echo $docker_name

docker exec -it $docker_name bash -c "cd data && source activate lemon && sqlite3 ./data/cifar10.db < ./data/create_db.sql"

python3 ./setup_muffin.py --mode seq
docker exec -it $docker_name bash -c "cd data && source activate lemon && CUDA_VISIBLE_DEVICES=-1 python run.py"
python3 ./setup_muffin.py --mode dag
docker exec -it $docker_name bash -c "cd data && source activate lemon && CUDA_VISIBLE_DEVICES=-1 python run.py"
