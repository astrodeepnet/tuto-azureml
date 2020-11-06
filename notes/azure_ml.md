# Azure ML

[back to README](../README.md)

- [First step](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#local)
- [Main page for documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Machine learning CLI](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)


## Using the Python SDK to interact with the Azure ML toolbox

Prepare a script with the ML model
In any Jupyter notebook, setup the Workspace, Compute target, Dataset, Experiment and execute the script.



## Jupyter notebook (used to create experiments and submit jobs)

### Initialize workspace and connect to account

```python
from azureml.core import Workspace

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

# Performing interactive authentication. Please follow the instructions on the terminal.
# To sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code XXXXXXXX to authenticate.
```

### Create compute target...

Here is [a list of the VM types and their prices](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)

```python
from azureml.core.compute import ComputeTarget, AmlCompute

# choose a name for the cluster
cluster_name = "gpu-cluster"
# choose a VM type
azure_vm = 'STANDARD_NC6' # <= 6 cores – 1 K80 GPU – 56Go RAM

compute_config = AmlCompute.provisioning_configuration(vm_size=azure_vm, 
                                                       max_nodes=4)

# create the cluster
compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

# can poll for a minimum number of nodes and for a specific timeout. 
# if no min node count is provided it uses the scale settings for the cluster
compute_target.wait_for_completion(show_output=True, 
                                   min_node_count=None,
                                   timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())
```

### or connect to a running compute target

from azureml.core.compute import ComputeTarget

```python
compute_target = ComputeTarget(workspace=ws, name=cluster_name)
```

### Create and register a dataset

```python
from azureml.core.dataset import Dataset

# Create Dataset from blob storage files
web_paths = [
    'https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
    'https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
    'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
    'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz'
]
dataset = Dataset.File.from_files(path = web_paths)

# Register the dataset
dataset = dataset.register(workspace=ws,
                           name='mnist-dataset',
                           description='training and test dataset',
                           create_new_version=True)

# or fetch an existing dataset
dataset = Dataset.get_by_name(ws, 'mnist-dataset')
```

## ML script

This is an example script running the MNIST classification.

All *model parameters that need specific tuning* should be put *as arguments* in the parser.

The specific Azure part is the Run context which is tied to the Azure Experiment and logs specific info to be retrieved later. In the following keras_mnist.py script, a keras callback is defined to log the accuracy and loss after each epoch through the run context. At the end of training, a plot of both curves is saved that way too.

```python
from azureml.core import Run
from keras.callbacks import Callback

# start an Azure ML run
run = Run.get_context()

class LogRunMetrics(Callback):
    # callback at the end of every epoch
    def on_epoch_end(self, epoch, log):
        # log a value repeated which creates a list
        run.log('Loss', log['val_loss'])
        run.log('Accuracy', log['val_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(X_valid, y_valid),
                    callbacks=[LogRunMetrics()])
```

```python
score = model.evaluate(X_test, y_test, verbose=0)
# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])
```

```python
# log a figure
run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('MNIST with Keras MLP ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['val_accuracy'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['val_loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss', plot=plt)
```


### Create a TensorFlow environment, specify the training script and the model parameters

```python
from azureml.train.dnn import TensorFlow

script_params = {
    '--data-folder': dataset.as_named_input('mnist').as_mount(),
    '--batch-size': 50,
    '--first-layer-neurons': 300,
    '--second-layer-neurons': 100,
    '--learning-rate': 0.001
}

est = TensorFlow(source_directory=script_folder,
                 script_params=script_params,
                 compute_target=compute_target, 
                 entry_script='keras_mnist.py',
                 framework_version='2.0', 
                 pip_packages=[
                     'keras<=2.3.1',
                     'azureml-dataset-runtime[pandas,fuse]',
                     'matplotlib'
                 ])
```

### Register an experiment and run the job

Experiments are useful to store and version the files and configuration used with a specific run.

```python
from azureml.core import Experiment

experiment_name = 'keras-mnist'

# create the experiment and submit the job
exp = Experiment(workspace=ws, name=experiment_name)

run = exp.submit(est)
```

```python
# then display the stdout
run.wait_for_completion(show_output=True)
```

```python
# or use a widget to do the same
from azureml.widgets import RunDetails
RunDetails(run).show()
```

After the job has completed, other information is accessible from the run context

```python
run.get_details()
run.get_metrics()
run.get_file_names()
```


### Saving trained models

```python
model = run.register_model(model_name='keras-dnn-mnist', model_path='outputs/model')
```

The created files or models can also be downloaded using the run context, e.g.

```python
# create a model folder in the current directory
os.makedirs('./model', exist_ok=True)

for f in run.get_file_names():
    if f.startswith('outputs/model'):
        output_file_path = os.path.join('./model', f.split('/')[-1])
        print('Downloading from {} to {} ...'.format(f, output_file_path))
        run.download_file(name=f, output_file_path=output_file_path)
```


## To go further

In *all* of the following pages there a tab above the code to select the Python SDK or the Azure CLI

- [install and use the Azure ML command-line interface](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)
- [deploy models with Azure ML](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where)
- [use distributed training](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-tensorflow#distributed-training)
- [standardise and optimise models with ONNX](https://docs.microsoft.com/en-us/azure/machine-learning/concept-onnx) (Open Neural Network Exchange)
