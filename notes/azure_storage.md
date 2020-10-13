Fonctionnement de Azure storage
===

[back to README](../README.md)


### Au sein d'un script TF

https://www.tensorflow.org/io/tutorials/azure

```bash
export TF_AZURE_STORAGE_KEY="XXX"
```

```python
import os
import tensorflow as tf
import tensorflow_io as tfio

account_name = 'blobsandbox1'
container_name = 'aztest'
blob_name = 'hello.txt'
    
pathname = f'az://{account_name}/{container_name}'
filename = f'{pathname}/{blob_name}'

# Création de container avec TF utils
tf.io.gfile.mkdir(pathname)

# Ecriture d'un blob
with tf.io.gfile.GFile(filename, mode='w') as w:
	w.write("Hello, world!")

# Lecture d'un blob
with tf.io.gfile.GFile(filename, mode='r') as r:
	print(r.read())
```



### Interaction avec un container / blob

**Prerequis :** `pip install azure-mgmt-storage`

**Connection string :** a trouver dans Storage Account > Access Keys

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=blobsandbox1;AccountKey=XXX;EndpointSuffix=core.windows.net"
```

```bash
az login
az storage blob list --container-name aztest
```

```python
import os
from azure.storage.blob import BlobServiceClient

conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

bsc = BlobServiceClient.from_connection_string(conn_str)

bsc.create_container(...)
bsc.delete_container(...)

container = bsc.get_container_client(container_name)
blob = bsc.get_blob_client(container_name, blob_name)


```



```Python
from azure.storage.blob import ContainerClient, BlobClient

storage_account_name = 'blobsandbox1'
container_name = 'aztest'
blob_name = 'hello.txt'
storage_key = 'XXX'

container_url = f'https://{storage_account_name}.blob.core.windows.net/{container_name}'
blob_url = f'{container_url}/{blob_name}'

container = ContainerClient.from_container_url(container_url, credential=storage_key)

# Listing blobs in the container
for blob in container.list_blobs():
    print(blob.name)

# Select a particular blob based on its name
blob = container.get_blob_client(blob_name)
# ... or directly the blob url
blob = BlobClient.from_blob_url(blob_url, credential=storage_keys)
```



### Récupération/upload d'un fichier depuis une url

```python
from azure.storage.blob import download_blob_from_url, upload_blob_to_url

storage_account_name = 'blobsandbox1'
container_name = 'aztest'
blob_name = 'hello.txt'
storage_key = 'XXX'

blob_url = f'https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_name}'
local_filepath = 'azure_hello.txt'

# Download file from storage
download_blob_from_url(
	blob_url, 
	local_filepath, 
	credential=storage_key)

# Upload file to storage
upload_blob_to_url(
	blob_url, 
	local_filepath, 
	credential=storage_key)
```



Upload contents of a folder to Blob storage
---

Found in [this page](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-blobs)

Install `azcopy` and login with 

```
azcopy login
```

then

```bash
# To upload new files
azcopy copy "<local-folder-path>" "https://<storage-account-name>.<blob or dfs>.core.windows.net/<container-name>" --recursive=true

# To synchronize existing files
azcopy sync "<local-folder-path>" "https://<storage-account-name>.<blob or dfs>.core.windows.net/<container-name>" --recursive=true
```

Marche dans les deux sens : permet de telecharger rapidement le contenu 

```bash
azcopy copy 'https://<storage-account-name>.<blob or dfs>.core.windows.net/<container-name>/<blob-path>' '<local-file-path>'

azcopy copy 'https://<storage-account-name>.<blob or dfs>.core.windows.net/<container-name>/<directory-path>' '<local-directory-path>' --recursive

azcopy copy 'https://<storage-account-name>.blob.core.windows.net/<container-name>/*' '<local-directory-path>/'
```
