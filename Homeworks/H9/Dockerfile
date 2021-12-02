#FROM public.ecr.aws/lambda/python:3.8
#COPY cats-dogs-v2.tflite .
FROM agrigorev/zoomcamp-cats-dogs-lambda:v2
RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

#The model is already in image, so no need to copy it
#COPY clothing-model.tflite .
COPY lambda_function.py .
CMD [ "lambda_function.lambda_handler" ]


#RUN pip install pipenv
#WORKDIR /app
#COPY ["Pipfile", "Pipfile.lock", "./"]

