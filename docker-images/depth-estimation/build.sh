#!/bin/bash

#Assemble docker image. 
echo 'Building docker image.'
sudo docker build . -t depth-estimation-app