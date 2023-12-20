# Defining base image
FROM ubuntu:18.04

# Setting environment variable DEBIAN_FRONTEND to noninteractive to avoid interactive prompts during package installation when using 
# APT (Advanced Package Tool, The package manager  for debian based system)
ENV DEBIAN_FRONTEND noninteractive

# For ensuring that the apt package list is up to date and for installing required packages
# ca-certificates contains a set of trusted certificate autorities that is used by the operating system and various applocations to establish the
# authenticity of SSL/TLS connections
# SSL (Secure Sockets Layer) and its successor TLS (Transport Layer Security) are cryptographic protocols used to secure communication over a network.
# They establish an encrypted connection between a client and a server, ensuring that the data transmitted between them remains confidential and integral
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3.8-distutils ca-certificates python3-dev python3-opencv git wget unzip

# update-alternative is a system in linux for managing multiple versions of the same software package
# update-alternatives --install ------- to add a new alternative for a particular command/tool
# /usr/bin/python3              ------- symbolic link of the command for which alternative is being added
# python3                       ------- name of the command for which alternative is being added
# /usr/bin/python3.8            ------- symbolic link of the alternative
# 1                             ------- priority of the alternative
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# To add a new user
# Can be USER_ID value can be provided explicitly when running the container
ARG USER_ID=1000
# useradd           ------- to add a user 
# -m                ------- to create user's home directory if it doesn't exist
# --no-log-init     ------- to prevent the creation of log file containing the initial log session for the user
# --system          ------- means that the user is a system user add it won't have a home dir by default
# --uid ${USER_ID}  ------- to assgin the id to the user
# appuser           ------- user name
# -g sudo           ------- to assign user a group "sudo", which means it will have administrative privileges
RUN useradd -m --no-log-init --system --uid ${USER_ID} -g sudo

# To allow sudoers (sudo group users) to run any command as any user without providing any password
# echo                            ------- appends line to the end of the file and also used to display command to a terminal
# '%sudo ALL=(ALL) NOPASSWD:ALL'  ------- %sudo refers to the sudo group and "ALL=(ALL) NOPASSWD:ALL" configuration of the sudo group. Here
#                                         to allow sudoers to run any command as any user without providing any password
# >>                              ------- to point to the file to which the line is to appended
# /etc/sudoers                    ------- filepath to which the line is to appended
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# To specify the user context under which subsequent commands will run
USER appuser

# To set the working directory
WORKDIR /home/appuser

# To set the PATH environment variable
# extends the current PATH variable by adding "/home/appuser/.local/bin" to the begining of the current PATH value
# It allows executables located in "/home/appuser/.local/bin" to be found and executed directly from the command line 
# without specifying the full path 
# This modification to the PATH environment variable ensures that any executables installed locally for the appuser within the .local/bin directory 
# are easily accessible without specifying their full path every time. This can be useful when installing user-specific tools or applications within 
# the container.
ENV PATH="/home/appuser/.local/bin:${PATH}"

# To install pip package installer for python
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

ARG CLONE_REPO=1






