# Reference image with Python 3.7 and Pip preinstalled.
FROM python:3.7.8-slim

# Set up working directory with all mxmc files.
COPY ./ /mxmc/
WORKDIR /mxmc/

# Install MXMC dependencies.
RUN pip3 install -r requirements.txt
	
# Install MXMC.
RUN pip3 install .