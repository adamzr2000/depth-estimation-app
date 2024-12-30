# ROS Master


adamzr@Alienware-m16-R1-AMD:~/keras-example-monocular-depth$ nvidia-smi
Sat Dec 28 09:13:25 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    Off | 00000000:01:00.0 Off |                  N/A |
| N/A   37C    P4              15W /  45W |    898MiB /  8188MiB |      2%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2362      G   /usr/lib/xorg/Xorg                          304MiB |
|    0   N/A  N/A      2571      G   /usr/bin/gnome-shell                        112MiB |
|    0   N/A  N/A      3344      G   ...ures=SpareRendererForSitePerProcess      227MiB |
|    0   N/A  N/A      5490      G   ...seed-version=20241225-174432.450000      174MiB |
|    0   N/A  N/A     10974      G   ...erProcess --variations-seed-version       69MiB |
+---------------------------------------------------------------------------------------+

 
```bash
docker run --gpus all -it --rm -v $(pwd)/test_tensorflow_gpu.py:/app/test_tensorflow_gpu.py tensorflow/tensorflow:2.14.0-gpu python /app/test_tensorflow_gpu.py --log-level=2

docker run --gpus all -it --rm -v $(pwd)/test_tensorflow_gpu.py:/app/test_tensorflow_gpu.py tensorflow/tensorflow:2.14.0-gpu bash
```



    libgl1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    libgtk-3-dev \