#docker run -it --rm --gpus all --runtime=nvidia --ipc=host -v /home/maneesh/Desktop/Research2.0:/Research -v /home/maneesh/Desktop/Research2.0/offline_saved:/offline_saved maneesh/docker:1.0

#docker run -it --rm --gpus all --runtime=nvidia --ipc=host -v /home/maneesh/Lab2.0/TRPE_Multi_Object:/TRPE_Multi_Object -v /data/maneesh/MergeDataset/Output_Dataset/offline_saved:/offline_data/ maneesh/docker:1.0

docker run -it --rm --gpus all --runtime=nvidia --ipc=host -v /home/maneesh/Lab2.0/TRPE_Multi_Object:/TRPE_Multi_Object -v /data/maneesh/MergeDataset/OK/offline_saved:/offline_saved maneesh/docker:1.0


