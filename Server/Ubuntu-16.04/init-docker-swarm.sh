#!/bin/bash
# Do not execute this as a script
# This is just a snippets reference
exit 1

# Create new manager node
docker swarm init --advertise-addr <MANAGER-IP>

# Get Swarm info
docker info

# Get node list
docker node ls

# Generate snippet to add a new worker to the swarm
docker swarm join-token worker

# Generate snippet to add a new manager to the swarm
docker swarm join-token manager


