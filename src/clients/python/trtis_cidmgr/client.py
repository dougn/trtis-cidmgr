# Copyright (c) 2019, Doug Napoleone. All rights reserved.

from tensorrtserver.api import ProtocolType, InferContext, InferRequestHeader
import numpy as np
import contextlib
from .codes import *

@contextlib.contextmanager
def ManagedContext(url, model_name, 
                   model_version=-1, 
                   verbose=False, streaming=True, 
                   cidmgr_model_name='cidmgr',
                   cidmgr_model_version=-1,
                   cidmgr_correlation_id=1):
    """
    Create a context for sequence models with a guarenteed unique context ID.
    Requires having the custom cidmgr backend with model on the server.

    .. code::
    
        with ManagedContext("localhost:8001", "tensor_sequence_model", streaming=True) as ctx:
            start_tensor = tensors.pop(0)
            end_tensor = tensors.pop()
            result = ctx.run({'INPUT' : start_tensor}, {'OUTPUT': InferContext.ResultFormat.RAW },
                            flags=InferRequestHeader.FLAG_SEQUENCE_START)
            results.append(result)
            for tensor in tensors:
                result = ctx.run({'INPUT' : tensor}, {'OUTPUT': InferContext.ResultFormat.RAW },
                                flags=InferRequestHeader.FLAG_NONE)
                results.append(result)
            result = ctx.run({'INPUT' : end_tensor}, {'OUTPUT': InferContext.ResultFormat.RAW },
                            flags=InferRequestHeader.FLAG_SEQUENCE_END)
            results.append(result)

    """
    protocol = ProtocolType.from_str("grpc")

    # Create the context for the cidmgr, non-streaming (don't keep the TCP live.
    cidmgr_ctx = InferContext(
        url, protocol, cidmgr_model_name, cidmgr_model_version,
        correlation_id=cidmgr_correlation_id, verbose=verbose, streaming=False)
    
    # Create the tensor for CODE/CORELATION_ID.
    cnew = np.full(shape=[1], fill_value=CIDMGR_NEW, dtype=np.int8)
    cid0 = np.full(shape=[1], fill_value=0, dtype=np.uint64)

    result = cidmgr_ctx.run({ 'CODE' : (cnew,) , 'CORRELATION_ID': (cid0,) },
                            { 'OUTPUT' : InferContext.ResultFormat.RAW },
                            batch_size=1, flags=InferRequestHeader.FLAG_SEQUENCE_START)
    
    # get the correlaiton_id
    correlation_id = result['OUTPUT'][0][0]
    print("Have Correlation Id: " + str(correlation_id))

    # yield the requeted context
    yield InferContext(
        url, protocol, model_name, model_version,
        correlation_id=correlation_id, verbose=verbose, streaming=streaming)
    
    # free the corelation id
    print("Removing managed Correlation ID: " + str(correlation_id))
    cdel = np.full(shape=[1], fill_value=CIDMGR_DELETE, dtype=np.int8)
    cid = np.full(shape=[1], fill_value=correlation_id, dtype=np.uint64)
    cidmgr_ctx.run({ 'CODE' : (cdel,) , 'CORRELATION_ID': (cid,) },
                   { 'OUTPUT' : InferContext.ResultFormat.RAW },
                   batch_size=1, flags=InferRequestHeader.FLAG_NONE)
