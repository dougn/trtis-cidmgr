
#pragma once

// Why trtis? WHY?
#include <src/clients/c++/request.h>

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

namespace dnapoleone { namespace inferenceserver { namespace correlation_id_mgr { namespace client {

using CorrelationIDSet = std::set<ni::CorrelationID>;

class CIDMgr {
 public:
  virtual ~CIDMgr() = default;

  // Get a new unique CorrelationId from the server
  virtual nic::Error NewCorrelationID(ni::CorrelationID* correlation_id) = 0;

  // Remove the CorrelationId from use
  virtual nic::Error DeleteCorrelationID(ni::CorrelationID correlation_id) = 0;

  // Get the number of active in use CorrelationIDs
  virtual nic::Error Active(uint64_t *active) = 0;

  // Get the number of inactive CorrelationIDs
  virtual nic::Error InActive(uint64_t *inactive) = 0;

  // Get the peak number of CorrelationIDs reserved
  virtual nic::Error Peak(uint64_t *peak) = 0;
  
  // Get all the CorrelationIDs currently in use by this context
  virtual nic::Error CorrelationIDs(std::unique_ptr<CorrelationIDSet> correlation_ids) = 0;

  // Remove all the CorrelationIDs in use by this context
  virtual nic::Error DeleteAllCorrelationIDs() = 0;

  // Create the InferGrpcContext or InferGrpcStreamContext with
  // a unique ni::CorrelationID
  virtual nic::Error Create(
    std::unique_ptr<nic::InferContext>* ctx, 
    const std::string& server_url, 
    const std::string& model_name,
    int64_t model_version = -1, 
    bool verbose = false,
    bool streaming = true) = 0;

  static nic::Error Create(
    std::unique_ptr<CIDMgr>* cidmgr,
    const std::string& server_url, 
    const std::string& model_name="cidmgr",
    int64_t model_version = -1, 
    bool verbose = false,
    bool streaming = false);

};

}}}} // namespace dnapoleone::inferenceserver::correlation_id_mgr::client
