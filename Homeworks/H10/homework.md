# Q1 
0.11.1

# Q2 
10.96.0.1

# Q3

kind load docker-image churn-model:v001

# Q4

The correct yaml file is 
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn
spec:
  selector:
    matchLabels:
      app: churn
  template:
    metadata:
      labels:
        app: churn
    spec:
      containers:
      - name: churn
        image: churn-model:v001
        resources:
          limits:
            memory: "128Mi"
            cpu: "500m"
        ports:
        - containerPort: 9696
```

# Q5
churn-8449c67c88-fvwvt

# Q6
churn
# Q7
