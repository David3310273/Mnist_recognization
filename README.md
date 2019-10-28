# Mnist_recognition
A MNIST recognition service using pytorch and python3 

Firstly, run this command under the project root directory to train the model:

```shell script
python3 main.py
```

then the model should be stored under the `model` directory

Finally, run this command to open the service

```shell script
python3 service.py
``` 

with an image uploaded, you can make a form request to this service and it will return the prediction result of your input image