# Notes from the lectures week5

## Saving the model
Use pickle to save the model

```
import pickle
output_file=f'model_C={C}.bin'
with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

```

## Load the model
Before loading the model I need to have
the libraries I used to create the model
or the command below will not work

```
model_file='model_C=1.0.bin'
with open(model_file,'rb') as f_in:
    (dv,model) = pickle.load(f_in)

```
customer={} # this a dict will all the type of customer
cv.transferm([customer])

model.predict_proba(X)[0,1]
