# trtis-cidmgr
 Custom backend and client libraries for the NVIDIA TensorRT-Inference-Server to manage unique Correlation ID's

The [NVIDIA TensorRT Inference Server](https://github.com/NVIDIA/tensorrt-inference-server) has an issue with [stateful models](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/models_and_schedulers.html#stateful-models).
The client API requires specifying a Correlation ID (a.k.a. Instance ID), which must be unique to the state. So if you are performing multiple inferences in parallel which have unique state, you need a unique id for each. It is up to the multiple clients (which may be on different machines) to coordinate the Correlation ID management. This is something the inference server should provide, as it is already a coordination point.


This custom backend and helper clients use the inference server framework to manage the distribution and management of unique correlation id's for clients. The backend is it's self a stateful model holding a registry of in use correlation id's. Clients send a simple tensor to this backend and get back a dimension [1] tensor containing a new unique correlation id for use with another stateful model.


This currently only works with the 1.5.0-dev mainline of the tensorrt-inference-server project.

Tested with Windows, Ubuntu, and OSX (private port) and with Python 2.7 and 3.7

## C++ Interface

There is a [CIDMgr class](src/clients/c++/cidmgr_client.h) which replaces the normal ```InferGrpcContext::Create(...)``` and ```InferGrpcStreamContext::Create(...)``` factories, but without the ``CorrelationID correlation_id`` argument. When a CIDMgr instance is deallocated, all CorrelationIDs by that instance will be deleted on the server.

```c++
#include <cidmgr_client.h>

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;
namespace dicc = dnapoleone::inferenceserver::correlation_id_mgr::client;

int main(int argc, char** argv)
{
  std::string url("localhost:8001");
  std::string model_name("simple_sequence");
  std::unique_ptr<dicc::CIDMgr> cidmgr;
  std::unique_ptr<nic::InferContext> ctx0, ctx1;
  
  // Create two different contexts, one is using streaming while the other
  // isn't. Then we can compare their difference in sync/async runs
  nic::Error err = dicc::CIDMgr::Create(&cidmgr, url);

  if (err.IsOk()) {
    err = cidmgr->Create(
        &ctx0, url, model_name, -1 /* model_version */, verbose, true /* streaming */);
  }
  if (err.IsOk()) {
    err = cidmgr->Create(
        &ctx1, url, model_name, -1 /* model_version */, verbose, false /* streaming */);
  }
}
```

The inference contexts can be used from there. 

## Python Interface

Example of using the simple_sequence stateful custom backend.
Will have the same results as the tensorrt-inference-server stateful client 
[simple_sequence_client.py](https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/simple_sequence_client.py)

```python
from trtis_cidmgr import StatefulContext, InferRequestHeader
from numpy import full, int32

values = [0, 11, 7, 5, 3, 2, 0, 1]
tensors = [[full(shape=[1], fill_value=v, dtype=int32),
            InferRequestHeader.FLAG_NONE] for v in values]
tensors[0][1] |= InferRequestHeader.FLAG_SEQUENCE_START
tensors[-1][1] |= InferRequestHeader.FLAG_SEQUENCE_END

with StatefulContext('localhost:8001', 'simple_sequence') as ctx:
    for tensor, flags in tensors:
        result = ctx.run(
            {'INPUT' : (tensor,)}, 
            {'OUTPUT': StatefulContext.ResultFormat.RAW },
            flags=flags)
        print(result['OUTPUT'][0][0])
```

Output:
```
0
11
18
23
26
28
28
29
```

See the [python code](src/clients/python/trtis_cidmgr/context.py) for more API details (Doc TBD)

## Building
You must supply the tensorrt-inference-server builddir with the targets ```trtis-custom-backends``` and ```trtis-clients``` built, following that projects [instructions for building](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/build.html#configure-inference-server).
```bash
$ mkdir build
$ cmake .. -DTRTIS_BUILDDIR=../../tensorrt-inference-server/builddir
$ make install
```

This will generate the following directory tree:

* install/
    * 3.5.env/ *- virtualenv with the trtis_cidmgr package and all dependencies*
    * model_repository/ *- trtserver [model repository](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_repository.html)*
        * cidmgr/
            * [config.pbtxt](src/config.pbtxt.in)
            * 1/
                * libcidmgr.so
        * simple_sequence/ *- test simple sequence model*
            * [config.pbtxt](test/simple_sequence_config.pbtxt.in)
            * 1/
                * libsequence.so *- tensorrt-inference-server custom sequence backend*
    * bin/
        * cidmgr_sequence_client *- tensorrt-inference-server simple_sequence_client modified to use cidmgr*
    * lib/
        * libcidmgr.so *- custom backend*
        * librequest.so *- tensorrt-inference-server client library*
        * libcidmgr_client.a *- cidmgr client helper library*
        * libsequence.so *- tensorrt-inference-server custom sequence backend*
    * include/
        * cidmgr_client.h
    * wheelhouse/
        * trtis_cidmgr-0.0.1-py2.py3-none-any.whl
        * tensorrtserver-1.5.0.dev0-py2.py3-none-manylinux1_x86_64.whl

## Testing

Running the trtserver

```bash
$ cd tensorrt-inference-server/builddir/trtis/bin
$ export LD_LIBRARY_PATH=../lib
$ ./trtserver --model-repository ../../../../../trtis-cidmgr/build/install/model_repository/
```

Running the simple_sequence custom backend and test with correlation id's from cidmgr

```bash
$ cd trtis-cidmgr/test
$ source ../build/install/3.7.env/bin/activate
$ cd test
$ python ./doug_sequence_client.py
Have Correlation Id: 1
Have Correlation Id: 2
streaming : non-streaming
[0] 0 : 100
[1] 11 : 89
[2] 18 : 82
[3] 23 : 77
[4] 26 : 74
[5] 28 : 72
[6] 28 : 72
[7] 29 : 71
Removing managed Correlation ID: 2
Removing managed Correlation ID: 1
```

Running 50 simple_sequence model clients in parallel with correlation id's from cidmgr
```bash
$ cd trtis-cidmgr/test
$ source ../build/install/3.7.env/bin/activate
$ python ./runmany.py
```

Running the [cidmgr_sequence_client](src/clients/c++/cidmgr_sequence_client.cc) c++ client

```bash
$ cd trtis-cidmgr/build/install/bin
$ export LD_LIBRARY_PATH=../lib
$ ./cidmgr_sequence_client
streaming : non-streaming
[0] 0 : 100
[1] 11 : 89
[2] 18 : 82
[3] 23 : 77
[4] 26 : 74
[5] 28 : 72
[6] 28 : 72
[7] 29 : 71
```
