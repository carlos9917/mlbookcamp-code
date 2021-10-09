#FROM python:3.8.12-slim
FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

#These two are already in the Pipfile
#RUN pipenv install scikit-learn~=1.0
#RUN pipenv install gunicorn
#this one will install it in the system using the pipfile above
RUN pipenv install --system --deploy

COPY ["question5.py", "model1.bin","dv.bin", "./"]
#expose the port 9696
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "question5:app"]
