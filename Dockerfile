# Reference image with Python 3.7.
FROM debian:stable

# Set up working directory with all mxmc files.
COPY ./ /mxmc/
WORKDIR /mxmc/

# Install Pip.
RUN apt update && apt install -y python3-pip

# Install dependencies.
RUN pip3 install -r requirements.txt
	
# Install mxmc.
RUN pip3 install .

