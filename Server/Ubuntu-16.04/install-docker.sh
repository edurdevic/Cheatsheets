#!/bin/bash

sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates
sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D


echo "deb https://apt.dockerproject.org/repo ubuntu-xenial main" >> /etc/apt/sources.list.d/docker.list

sudo apt-get update

sudo apt-get purge lxc-docker

apt-cache policy docker-engine

# Install  aufs storage driver
sudo apt-get install linux-image-extra-$(uname -r)

sudo apt-get install docker-engine

sudo groupadd docker

sudo usermod -aG docker root

echo "Now logout and login again... Docker installation complete."



