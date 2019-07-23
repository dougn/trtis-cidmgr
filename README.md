# trtis-cidmgr
 Custom backend and client libraries for the NVIDIA TensorRT-Inference-Server to manage unique Correlation ID's

The [NNIDIA TensorRT Inference Server](https://github.com/NVIDIA/tensorrt-inference-server) has an issue with [stateful models](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/models_and_schedulers.html#stateful-models).
The client API requires specifying a Correlaiton ID (a.k.a. Instance ID), which must be unique to the state. So if you are performing multiple inferences in parallel which have unique state, you need a unique id for each. It is up to the multiple clients (which may be on different machines) to coordinate the Correlation ID management. This is something the inference server should provide, as it is already a coordination point.

This custom backend and helper clients use the inference server framework to manage the distrobution and management of unique correlation id's for clients. The backend is it's self a stateful model holding a registry of in use correlation id's. Clients send a simple tensor to this backend and get back a dimension [1] tensor containing a new unique correlaiton id for use with another stateful model.

This currently only works with the 1.5.0-dev mainline of the tensorrt-inference-server project.

## Building
You must supply the tensorrt-inference-server builddir with the targets ```trtis-custom-backends``` and ```trtis-clients``` built, following that projects [instructions for building](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/build.html#configure-inference-server).
```
$ mkdir build
$ cmake .. -DTRTIS_BUILDDIR=../../tensorrt-inference-server/builddir
$ make install
```

This will generate the following directory tree:

* install/
    * 3.5.env/ *- virtualenv with the trtis_cidmgr package and all dependencies*
    * model/ *- trtserver [model repository](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_repository.html)*
        * cidmgr/
            * [config.pbtxt](src/config.pbtxt.in)
            * 1/
                * libcidmgr.so
    * lib/
        * libcidmgr.so *- custom backend*
        * libcidmgr_client.a *- client helper library*
    * include/
        * cidmgr_client.h
        * cidmgr_codes.h
    * wheelhouse/
        * trtis_cidmgr-0.0.1-py2.py3-none-any.whl
        * tensorrtserver-1.5.0.dev0-py2.py3-none-manylinux1_x86_64.whl
