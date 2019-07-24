// Copyright (c) 2019 Doug Napoleone, All rights reserved.

#include "cidmgr_client.h"
#include <cidmgr_codes.h>
#include <src/clients/c++/request_grpc.h>

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

namespace dnapoleone { namespace inferenceserver { namespace correlation_id_mgr { namespace client {

class CIDMgrImpl : public CIDMgr
{
 public:
  CIDMgrImpl(): ctx_(nullptr), correlation_ids_()
  {

  }
  virtual ~CIDMgrImpl()
  {
    DeleteAllCorrelationIDs();
  }

  nic::Error Init(
    const std::string& server_url, 
    const std::string& model_name,
    int64_t model_version, 
    bool verbose,
    bool streaming);
  
  virtual nic::Error Create(
    std::unique_ptr<nic::InferContext>* ctx, 
    const std::string& server_url, 
    const std::string& model_name,
    int64_t model_version = -1, 
    bool verbose = false,
    bool streaming = true);

  virtual nic::Error NewCorrelationID(ni::CorrelationID* correlation_id)
  {
    nic::Error err = Run(static_cast<uint64_t*>(correlation_id), CIDMGR_NEW, 0);
    if (err.IsOk())
    {
      correlation_ids_.insert(*correlation_id);
    }
    return err;
  }

  virtual nic::Error DeleteCorrelationID(ni::CorrelationID correlation_id)
  {
    nic::Error err = Run(nullptr, CIDMGR_DELETE, correlation_id);
    auto it = correlation_ids_.find(correlation_id);
    if (it != correlation_ids_.end()) {
      correlation_ids_.erase(it);
    }
    return err;
  }

  virtual nic::Error Active(uint64_t *active)
  {
    return Run(active, CIDMGR_ACTIVE, 0);
  }

  virtual nic::Error InActive(uint64_t *inactive)
  {
    return Run(inactive, CIDMGR_INACTIVE, 0);
  }

  virtual nic::Error Peak(uint64_t *peak)
  {
    return Run(peak, CIDMGR_PEAK, 0);
  }

  virtual nic::Error CorrelationIDs(std::unique_ptr<CorrelationIDSet> correlation_ids)
  {
    correlation_ids->clear();
    *correlation_ids = correlation_ids_;
    return nic::Error::Success;
  }

  virtual nic::Error DeleteAllCorrelationIDs()
  {
    nic::Error err = nic::Error::Success;
    for (auto it = correlation_ids_.begin(); it != correlation_ids_.end(); )
    {
      err = Run(nullptr, CIDMGR_DELETE, *it);
      it = correlation_ids_.erase(it);
      if (!err.IsOk())
      {
        return err;
      }
    }
    return err;
  }

 protected:
  nic::Error GetInput(
    std::shared_ptr<nic::InferContext::Input>* input,
    const std::string& name,
    uint8_t* value,
    size_t size);

  nic::Error Run(
    uint64_t *result, 
    CIDMGR_Code code, 
    ni::CorrelationID correlation_id);

  std::unique_ptr<nic::InferContext> ctx_;
  CorrelationIDSet correlation_ids_;

};

nic::Error CIDMgrImpl::GetInput(
  std::shared_ptr<nic::InferContext::Input>* input,
  const std::string& name,
  uint8_t* value,
  size_t size)
{
  nic::Error err = ctx_->GetInput(name, input);
  if (!err.IsOk())
  {
    return err;
  }
  err = (*input)->Reset();
  if (!err.IsOk())
  {
    return err;
  }
  err = (*input)->SetRaw(reinterpret_cast<uint8_t*>(value), size);
  return err;
}

nic::Error 
CIDMgrImpl::Run(
  uint64_t *result, 
  CIDMGR_Code code, 
  ni::CorrelationID correlation_id)
{
  nic::Error err = nic::Error::Success;
  int8_t vcode = code;
  uint64_t vcorrelation_id = correlation_id;

  // Set options
  std::unique_ptr<nic::InferContext::Options> options;
  err = nic::InferContext::Options::Create(&options);
  if (!err.IsOk()) { return err; }
  options->SetFlags(0);
  if (code == CIDMGR_NEW) {
    options->SetFlag(ni::InferRequestHeader::FLAG_SEQUENCE_START, true);
  }
  options->SetBatchSize(1);
  for (const auto& output : ctx_->Outputs()) {
    options->AddRawResult(output);
  }

  err = ctx_->SetRunOptions(*options);
  if (!err.IsOk()) { return err; }


  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> icode;
  std::shared_ptr<nic::InferContext::Input> icorrelation_id;
  err = GetInput(&icode, "CODE", 
                 reinterpret_cast<uint8_t*>(&vcode), sizeof(int8_t));
  if (!err.IsOk()) { return err; }
  err = GetInput(&icorrelation_id, "CORRELATION_ID", 
                 reinterpret_cast<uint8_t*>(&vcorrelation_id), sizeof(uint64_t));
  if (!err.IsOk()) { return err; }

  // Send inference request to the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  err = ctx_->Run(&results);
  if (!err.IsOk()) { return err; }

  uint64_t r = 0;
  err = results["OUTPUT"]->GetRawAtCursor(0 /* batch idx */, &r);

  if (result != nullptr) {
    *result=r;
  }

  return err;
}

nic::Error 
CIDMgrImpl::Init(
  const std::string& server_url, 
  const std::string& model_name,
  int64_t model_version, 
  bool verbose,
  bool streaming)
{
  nic::Error err = nic::Error::Success;
  if (streaming) {
    err = nic::InferGrpcStreamContext::Create(
      &ctx_, 1, server_url, model_name, model_version, verbose);
  } else {
    err = nic::InferGrpcContext::Create(
      &ctx_, 1, server_url, model_name, model_version, verbose);
  }
  return err;
}

nic::Error 
CIDMgrImpl::Create(
  std::unique_ptr<nic::InferContext>* ctx, 
  const std::string& server_url, 
  const std::string& model_name,
  int64_t model_version, 
  bool verbose,
  bool streaming)
{
  ni::CorrelationID correlation_id = 0;
  nic::Error err = NewCorrelationID(&correlation_id);
  if(!err.IsOk()){
    return err;
  }
  if (streaming) {
    err = nic::InferGrpcStreamContext::Create(
      ctx, correlation_id, server_url, model_name, model_version, verbose);
  } else {
    err = nic::InferGrpcContext::Create(
      ctx, correlation_id, server_url, model_name, model_version, verbose);
  }
  return err;
}

nic::Error 
CIDMgr::Create(
  std::unique_ptr<CIDMgr>* cidmgr,
  const std::string& server_url, 
  const std::string& model_name,
  int64_t model_version, 
  bool verbose,
  bool streaming)
{
  CIDMgrImpl* cidmgr_ptr = new CIDMgrImpl();
  cidmgr->reset(static_cast<CIDMgr*>(cidmgr_ptr));

  nic::Error err = cidmgr_ptr->Init(
    server_url, model_name, model_version, verbose, streaming);

  if (!err.IsOk()) {
    cidmgr->reset();
  }

  return err;
}

}}}} // namespace dnapoleone::inferenceserver::correlation_id_mgr::client
