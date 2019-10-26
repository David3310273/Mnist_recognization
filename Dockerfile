FROM python:3.5.2   

WORKDIR /opt/project/lenet
COPY requirements.txt .
RUN pip3 install    
COPY . .

CMD [ "python3", "service.py" ]