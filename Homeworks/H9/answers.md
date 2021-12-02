# Q1
43 MB

# Q2
Output index: 13

# Q3
no comprende...
See notebook.

# Q4
See notebook

# Q5
Sending build context to Docker daemon  330.8kB
Step 1/3 : FROM agrigorev/zoomcamp-cats-dogs-lambda:v2
 ---> 322fc756f258
Step 2/3 : RUN pip install keras-image-helper

----------------
# Q6
Build the docker container using build.sh
Run container with:
docker run -it --rm -p 8080:8080 homework9:latest
Should see:
```
INFO[0000] exec '/var/runtime/bootstrap' (cwd=/var/task, handler=) 
```

While running container, run in another window the script test.py to check result
from prediction on image provided
Result:
[0.7210462093353271]

