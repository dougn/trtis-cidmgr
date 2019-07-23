from tensorrtserver.api import ProtocolType, InferContext, InferRequestHeader
import numpy as np
import contextlib
from .codes import *

class CIDMgrContext(InferContext):
    """Smart InferContext for the cidmgr custom backend.

    This context is used for generating unique correlation_ids for use
    with stateful sequence model InferContexts. The stateful() method
    should be used to generate new stateful contexts.

    .. code::
        with CIDMgrContext('localhost:8001') as cidmgr:
            with cidmgr.stateful('localhost:8001', 'simple_sequence') as ctx:
                start_tensor = tensors.pop(0)
                end_tensor = tensors.pop()
                result = ctx.run(
                    {'INPUT' : start_tensor},
                    {'OUTPUT': InferContext.ResultFormat.RAW },
                    flags=InferRequestHeader.FLAG_SEQUENCE_START)
                results.append(result)
                for tensor in tensors:
                    result = ctx.run(
                        {'INPUT' : tensor}, 
                        {'OUTPUT': InferContext.ResultFormat.RAW },
                        flags=InferRequestHeader.FLAG_NONE)
                    results.append(result)
                result = ctx.run(
                    {'INPUT' : end_tensor}, 
                    {'OUTPUT': InferContext.ResultFormat.RAW },
                    flags=InferRequestHeader.FLAG_SEQUENCE_END)
                results.append(result)
    """
    def __init__(self, url, model_name='cidmgr', model_version=-1,
                 verbose=False, correlation_id=1, streaming=False):
        protocol = ProtocolType.from_str("grpc")
        self._id_registry = set()
        # make it work with both 2 and 3 as InferContext does not
        # inherit from object, so super in broken in 2.
        InferContext.__init__(self,
            url, protocol, model_name, model_version, 
            verbose, correlation_id, streaming)

    def _cidmgr_run(self, code, cid=0, start=False):
        tcode = np.full(shape=[1], fill_value=code, dtype=np.int8)
        tcid  = np.full(shape=[1], fill_value=cid,  dtype=np.uint64)
        flags = InferRequestHeader.FLAG_NONE
        if start:
            flags |= InferRequestHeader.FLAG_SEQUENCE_START
        result = self.run(
            { 'CODE' : (tcode,) , 'CORRELATION_ID': (tcid,) },
            { 'OUTPUT' : InferContext.ResultFormat.RAW },
            batch_size=1, flags=flags)
    
        # get the correlaiton_id
        return result['OUTPUT'][0][0]
    
    def close(self):
        """Delete any held correlation_ids, and then close the context. 
        Any future calls to object will result in an Error.
        """
        for id in self.correlation_ids():
            self.delete(id)
        # make it work with both 2 and 3 as InferContext does not
        # inherit from object, so super in broken in 2.
        InferContext.close(self)

    def new(self):
        """Get a new unique correlation_id from the server.
        """
        correlation_id = self._cidmgr_run(CIDMGR_NEW, start=True)
        self._id_registry.add(correlation_id)
        return correlation_id
    
    def delete(self, correlation_id):
        """Remove the correlation_id from the active reserved list on the server.
        """
        if self._ctx is not None and correlation_id in self._id_registry:
            self._id_registry.remove(correlation_id)
            self._cidmgr_run(CIDMGR_DELETE, correlation_id)
    
    def active(self):
        """Return the number of active reserved correlation id's on the server.

        If this number is always increasing, then there is a bug somewhere in client
        code where they are not properly deleting reserved id's.
        """
        return self._cidmgr_run(CIDMGR_ACTIVE)

    def inactive(self):
        """Return the number of inactive id's in the reserved space on the server.
        """
        return self._cidmgr_run(CIDMGR_INACTIVE)

    def peak(self):
        """Return the peak number of parallel correlation id's reserved by the server.
        """
        return self._cidmgr_run(CIDMGR_PEAK)
    
    def correlation_ids(self):
        """Return the list of correlation_id's registered with this context.
        """
        return list(self._id_registry)

    def stateful(self, 
        url, protocol, model_name, model_version=None,
        verbose=False, streaming=False):
        return StatefulContext(
            url, protocol, model_name, model_version,
            verbose, self, streaming)

class StatefulContext(InferContext):
    """Smart Wrapper around the InferContext which gets a unique correlation_id using the CIDMgrContext.

    If you will be creating multiple model contexts, it is better to use the context() method
    on an instance of CIDMgrContext.

    .. code::
        with StatefulContext('localhost:8001', 'simple_sequence') as ctx:
            start_tensor = tensors.pop(0)
            end_tensor = tensors.pop()
            result = ctx.run(
                {'INPUT' : start_tensor},
                {'OUTPUT': InferContext.ResultFormat.RAW },
                flags=InferRequestHeader.FLAG_SEQUENCE_START)
            results.append(result)
            for tensor in tensors:
                result = ctx.run(
                    {'INPUT' : tensor}, 
                    {'OUTPUT': InferContext.ResultFormat.RAW },
                    flags=InferRequestHeader.FLAG_NONE)
                results.append(result)
            result = ctx.run(
                {'INPUT' : end_tensor}, 
                {'OUTPUT': InferContext.ResultFormat.RAW },
                flags=InferRequestHeader.FLAG_SEQUENCE_END)
            results.append(result)
    """
    def __init__(self, url, model_name, model_version=None,
                 verbose=False, cidmgr_context=None, streaming=False):
        if not cidmgr_context:
            cidmgr_context = CIDMgrContext(url, verbose=verbose)
        self.cidmgr=cidmgr_context
        protocol = ProtocolType.from_str("grpc")
        correlation_id = self.cidmgr.new()
        # make it work with both 2 and 3 as InferContext does not
        # inherit from object, so super in broken in 2.
        InferContext.__init__(self,
            url, protocol, model_name, model_version, 
            verbose, correlation_id, streaming)
    
    def close(self):
        self.cidmgr.delete(self.correlation_id())
        # make it work with both 2 and 3 as InferContext does not
        # inherit from object, so super in broken in 2.
        InferContext.close(self)
    
