// Copyright (c) 2019 Doug Napoleone, All rights reserved.

#include <chrono>
#include <string>
#include <thread>
// I wish tensorrt-inference-server did this properly.
// This is in it's install directory under include:
#include "src/backends/custom/custom.h"
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"

#include "cidmgr.h"

namespace ni = nvidia::inferenceserver;
//namespace nic = nvidia::inferenceserver::custom;

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout
#define QUOTE(seq) "\""#seq"\""

// Just to be sane, we will set the max to be well below the uint64 max.
// We will only hit this if contexts are not being cleaned up. 
// trtis will clean up stale contexts, but this is really a bug in client
// code, or clients are dying. We want to know about this happening.
// However we don't want undue errors, or runaway allocation. 
// This gives us space to see the problem via Peak() stat always increasing 
// before we get close to an overflow or insane amounts of memory allocated. 
// This will reach ~4GB of memory allocated before we run out of ID's.
#define MAX_CORRELATION_ID (1<<30)

// 1 hour minimum idle recommended to prevent premature context deletion for
// the it manager. We want there to only ever be 1 manager, and we only want
// it deleted if there are no outstanding id's. 
#define MIN_SEQUENCE_IDLE 3600000000


// This custom backend takes two one-element input tensors, and one
// two-element tensor. Two INT32 control values and one an [uns8, uint64] input; 
// and produces a one-element output tensor. The input tensors must be named "START",
// "READY" and "INPUT". The output tensor must be named "OUTPUT".
//
// The backend maintains an INT32 accumulator which is updated based
// on the control values in "START" and "READY":
//
//   READY=0, START=*, CONTROL=*:               CORRELATION_ID=*: Ignore value input, do nothing.
//   READY=1, START=1: CONTROL=CIDMGR_NEW:      CORRELATION_ID=*: Create new correlation ID and return it.
//   READY=1, START=*: CONTROL=CIDMGR_DELETE:   CORRELATION_ID=N: Clear correlation ID N for reuse.
//   READY=1, START=*: CONTROL=CIDMGR_ACTIVE:   CORRELATION_ID=*: Num context id's in use.
//   READY=1, START=*: CONTROL=CIDMGR_INACTIVE: CORRELATION_ID=*: Num context id's no longer in use.
//   READY=1, START=*: CONTROL=CIDMGR_PEAK:     CORRELATION_ID=*: Peak num contexts used at one time.
//
// We abuse the START=1 control value and never reset the registry.
// By always passing START=1 there are no race conditions on being the first client to
// initialize the registry.

namespace dnapoleone { namespace inferenceserver { namespace correlation_id_mgr {
namespace backend {

// Integer error codes. TRTIS requires that success must be 0. All
// other codes are interpreted by TRTIS as failures.
enum ErrorCodes {
  kSuccess,
  kUnknown,
  kInvalidModelConfig,
  kGpuNotSupported,
  kSequenceBatcher,
  kModelControl,
  kSequenceIdle,
  kInputOutput,
  kInputName,
  kOutputName,
  kInputOutputDataType,
  kInputContents,
  kInputSize,
  kOutputBuffer,
  kBatchNotOne,
  kTimesteps,
  kInvalidId,
  kInvalidCode,
  kOutOfIDS,
  kDeleteWhileActive
};

// Context object. All state must be kept in this object.
class Context {
 public:
  Context(
      const std::string& instance_name, const ni::ModelConfig& config,
      const int gpu_device);
  ~Context();

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

  // Stats
  // In use reserved context id's
  uint64_t Active() const { return (uint64_t)reserved_.size();}
  // No longer in use, created id's
  uint64_t Inactive() const { return (uint64_t)available_.size(); }
  // Peak number of contexts in use at one time
  uint64_t Peak() const { return next_correlation_id_; }

 private:
  int GetInputTensor(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const size_t expected_byte_size, std::vector<uint8_t>* input);

  // generate a new correlation id, 0 is an error.
  uint64_t NewCorrelationID();

  // clear an already registered correlation id.
  int ClearCorrelationID(uint64_t id);


  // The name of this instance of the backend.
  const std::string instance_name_;

  // The model configuration.
  const ni::ModelConfig model_config_;

  // The GPU device ID to execute on or CUSTOM_NO_GPU_DEVICE if should
  // execute on CPU.
  const int gpu_device_;

  // registry of active ID's.
  std::unordered_set<uint64_t> reserved_;
  std::set<uint64_t> available_;
  uint64_t next_correlation_id_;

};

Context::Context(
    const std::string& instance_name, const ni::ModelConfig& model_config,
    const int gpu_device)
    : instance_name_(instance_name), model_config_(model_config),
      gpu_device_(gpu_device), reserved_(), available_(), 
      next_correlation_id_(1)
{
}

Context::~Context() 
{

}

int
Context::Init()
{
  // Execution on GPUs not supported since only a trivial amount of
  // computation is required.
  if (gpu_device_ != CUSTOM_NO_GPU_DEVICE) {
    return kGpuNotSupported;
  }

  // The model configuration must specify the sequence batcher and
  // must use the START and READY input to indicate control values.
  if (!model_config_.has_sequence_batching()) {
    return kSequenceBatcher;
  }

  auto& batcher = model_config_.sequence_batching();
  if (batcher.control_input_size() != 2) {
    return kModelControl;
  }
  if (!(((batcher.control_input(0).name() == "START") &&
         (batcher.control_input(1).name() == "READY")) ||
        ((batcher.control_input(0).name() == "READY") &&
         (batcher.control_input(1).name() == "START")))) {
    return kModelControl;
  }

  if (batcher.max_sequence_idle_microseconds() < MIN_SEQUENCE_IDLE)
  {
    return kSequenceIdle;
  }

  if (model_config_.max_batch_size() != 1) {
    return kBatchNotOne;
  }

  // There must be one INT8 input called CODE defined in the model
  // configuration with shape [1].
  if (model_config_.input_size() != 2) {
    return kInputOutput;
  }

  
  if ((model_config_.input(0).dims().size() != 1) ||
      (model_config_.input(0).dims(0) != 1)) {
    return kInputOutput;
  }
  if (model_config_.input(0).data_type() != ni::DataType::TYPE_INT8) {
    return kInputOutputDataType;
  }
  if (model_config_.input(0).name() != "CODE") {
    return kInputName;
  }

  // There must be one uint64 input called CORRELATION_ID 
  // defined in the model configuration with shape [1].
  if ((model_config_.input(1).dims().size() != 1) ||
      (model_config_.input(1).dims(0) != 1)) {
    return kInputOutput;
  }
  if (model_config_.input(1).data_type() != ni::DataType::TYPE_UINT64) {
    return kInputOutputDataType;
  }
  if (model_config_.input(1).name() != "CORRELATION_ID") {
    return kInputName;
  }

  // There must be one uint64 output with shape [1]. The output must be
  // named OUTPUT.
  if (model_config_.output_size() != 1) {
    return kInputOutput;
  }
  if ((model_config_.output(0).dims().size() != 1) ||
      (model_config_.output(0).dims(0) != 1)) {
    return kInputOutput;
  }
  if (model_config_.output(0).data_type() != ni::DataType::TYPE_UINT64) {
    return kInputOutputDataType;
  }
  if (model_config_.output(0).name() != "OUTPUT") {
    return kOutputName;
  }

  return kSuccess;
}

int
Context::GetInputTensor(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const size_t expected_byte_size, std::vector<uint8_t>* input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input' vector. A
  // more performant solution would attempt to use the input tensors
  // in-place instead of having this copy.
  uint64_t total_content_byte_size = 0;

  while (true) {
    const void* content;
    uint64_t content_byte_size = expected_byte_size - total_content_byte_size;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputContents;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    LOG_INFO << std::string(name) << ": size " << content_byte_size << ", ";
    if (std::string(name) == "CODE") {
      LOG_INFO << (reinterpret_cast<const int8_t*>(content)[0]) << std::endl;
    } else if (std::string(name) == "CORRELATION_ID") {
      LOG_INFO << (reinterpret_cast<const uint64_t*>(content)[0]) << std::endl;
    } else {
      LOG_INFO << (reinterpret_cast<const int32_t*>(content)[0]) << std::endl;
    }

    // If the total amount of content received exceeds what we expect
    // then something is wrong.
    total_content_byte_size += content_byte_size;
    if (total_content_byte_size > expected_byte_size) {
      return kInputSize;
    }

    input->insert(
        input->end(), static_cast<const uint8_t*>(content),
        static_cast<const uint8_t*>(content) + content_byte_size);
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (total_content_byte_size != expected_byte_size) {
    return kInputSize;
  }

  return kSuccess;
}

// generate a new correlation id, 0 is an error.
uint64_t 
Context::NewCorrelationID()
{
  auto it = available_.upper_bound(0);
  uint64_t new_id = next_correlation_id_;
  if (available_.size() == 0 || it == available_.end()) {
    if (next_correlation_id_ >= MAX_CORRELATION_ID) {
      return 0;
    }
    next_correlation_id_++;
  } else {
    new_id = *it;
    available_.erase(it);
  }
  reserved_.insert(new_id);
  return new_id;
}

// clear an already registered correlation id.
int 
Context::ClearCorrelationID(uint64_t id)
{
  auto it = reserved_.find(id);
  if (it == reserved_.end()) {
    return kInvalidId;
  }
  reserved_.erase(it);
  available_.insert(id);
  return kSuccess;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  LOG_INFO << "Correlation ID Mgr executing " << payload_cnt << " payloads" << std::endl;

  // Each payload represents different sequence. Each payload must have
  // batch-size 1 inputs which is the next timestep for that
  // sequence. The total number of payloads will not exceed the
  // max-batch-size specified in the model configuration.
  // The max-batch-size must be set to 1 in the model configuration.
  // Each payload must have a batch side of 1.

  // We return kSuccess on most errors, and attempt to send back the message.
  // We want the error on the payload. When that is not possible we return
  // a real error.

  if (payload_cnt != 1)
  {
    for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
      CustomPayload& payload = payloads[pidx];
      payload.error_code = kTimesteps;
    }
    return kTimesteps;
  }

  int err;

  CustomPayload& payload = payloads[0];
  if (payload.batch_size != 1) {
    payload.error_code = kTimesteps;
    return kTimesteps;
  }

  const size_t batch1_int32_size = GetDataTypeByteSize(ni::TYPE_INT32);
  const size_t batch1_int8_size  = GetDataTypeByteSize(ni::TYPE_INT8);
  const size_t batch1_uint64_size = GetDataTypeByteSize(ni::TYPE_UINT64);


  // Get the input tensors.
  std::vector<uint8_t> start_buffer, ready_buffer, code_buffer, correlation_id_buffer;
  err = GetInputTensor(
      input_fn, payload.input_context, "START", batch1_int32_size,
      &start_buffer);
  if (err != kSuccess) {
    payload.error_code = err;
    return kSuccess;
  }

  err = GetInputTensor(
      input_fn, payload.input_context, "READY", batch1_int32_size,
      &ready_buffer);
  if (err != kSuccess) {
    payload.error_code = err;
    return kSuccess;
  }

  err = GetInputTensor(
      input_fn, payload.input_context, "CODE", batch1_int8_size,
      &code_buffer);
  if (err != kSuccess) {
    payload.error_code = err;
    return kSuccess;
  }

  err = GetInputTensor(
      input_fn, payload.input_context, "CORRELATION_ID", batch1_uint64_size,
      &correlation_id_buffer);
  if (err != kSuccess) {
    payload.error_code = err;
    return kSuccess;
  }

  //int32_t*  start = reinterpret_cast<int32_t*>(&start_buffer[0]);
  int32_t*  ready = reinterpret_cast<int32_t*>(&ready_buffer[0]);
  int8_t*   code  = reinterpret_cast<int8_t*>(&code_buffer[0]);
  uint64_t* correlation_id = reinterpret_cast<uint64_t*>(
                              &correlation_id_buffer[0]);

  // if not ready, why? batch size? scheduler?
  // the READY behavior has changed between 1.3 and 1.5?
  if (ready[0] == 0) {
    return kSuccess;
  }

  uint64_t output_correlation_id = correlation_id[0];

  switch (code[0]) {
    case CIDMGR_NEW:
      output_correlation_id = NewCorrelationID();
      if (output_correlation_id == 0)
      {
        payload.error_code = kOutOfIDS;
      }
      break;
    case CIDMGR_DELETE:
      payload.error_code = ClearCorrelationID(output_correlation_id);
      break;
    case CIDMGR_ACTIVE:
      output_correlation_id = Active();
      break;
    case CIDMGR_INACTIVE:
      output_correlation_id = Inactive();
      break;
    case CIDMGR_PEAK:
      output_correlation_id = Peak();
      break;
    default:
      payload.error_code = kInvalidCode;
  }

  // If the output is requested, copy the calculated output value
  // into the output buffer.
  if ((payload.error_code == 0) && (payload.output_cnt > 0)) {
    const char* output_name = payload.required_output_names[0];
    
    // The output shape is [1]
    std::vector<int64_t> shape;
    shape.push_back(payload.batch_size);
    shape.push_back(1);
    
    void* obuffer;
    if (!output_fn(
            payload.output_context, output_name, 
            shape.size(), &shape[0],
            batch1_uint64_size, &obuffer)) {
      payload.error_code = kOutputBuffer;
      return kSuccess;
    }
    //std::cout << "writing out: " << output_correlation_id << std::endl;
    // If no error but the 'obuffer' is returned as nullptr, then
    // skip writing this output.
    if (obuffer != nullptr) {
      memcpy(obuffer, &output_correlation_id, batch1_uint64_size);
    }
  }

  return kSuccess;
}

/////////////

extern "C" {

int
CustomInitialize(const CustomInitializeData* data, void** custom_context)
{
  // Convert the serialized model config to a ModelConfig object.
  ni::ModelConfig model_config;
  if (!model_config.ParseFromString(std::string(
          data->serialized_model_config, data->serialized_model_config_size))) {
    return kInvalidModelConfig;
  }

  // Create the context and validate that the model configuration is
  // something that we can handle.
  Context* context = new Context(
      std::string(data->instance_name), model_config, data->gpu_device_id);
  int err = context->Init();
  if (err != kSuccess) {
    return err;
  }

  *custom_context = static_cast<void*>(context);

  return kSuccess;
}

int
CustomFinalize(void* custom_context)
{
  int err = kSuccess;
  if (custom_context != nullptr) {
    Context* context = static_cast<Context*>(custom_context);
    std::cout << "Deleting Correlation ID Mgr Context with " << 
      context->Active() << " active id's" << std::endl;
    if (context->Active()) {
      err = kDeleteWhileActive;
    }
    delete context;
  }

  return kSuccess;
}

const char*
CustomErrorString(void* custom_context, int errcode)
{
  switch (errcode) {
    case kSuccess:
      return "success";
    case kInvalidModelConfig:
      return "invalid model configuration";
    case kGpuNotSupported:
      return "execution on GPU not supported";
    case kSequenceBatcher:
      return "model configuration must configure sequence batcher";
    case kModelControl:
      return "'START' and 'READY' must be configured as the control inputs";
    case kSequenceIdle:
      return "model max_sequence_idle_microseconds is set below the minimum " QUOTE(MIN_SEQUENCE_IDLE);
    case kInputOutput:
      return "model must have two inputs and one output with shape [1]";
    case kInputName:
      return "model inputs must be named 'CODE' and 'CORRELATION_ID'";
    case kOutputName:
      return "model output must be named 'OUTPUT'";
    case kInputOutputDataType:
      return "model inputs and output must have TYPE_INT32 data-type";
    case kInputContents:
      return "unable to get input tensor values";
    case kInputSize:
      return "unexpected size for input tensor";
    case kOutputBuffer:
      return "unable to get buffer for output tensor values";
    case kBatchNotOne:
      return "max-batch-size must be set to 1";
    case kTimesteps:
      return "unable to execute more than 1 timestep at a time";
    case kInvalidId:
      return "invalid CORRELATION_ID given for deletion";
    case kInvalidCode:
      return "unknwon input CODE";
    case kOutOfIDS:
      return "out of calid correlation id space; clients leaking";
    case kDeleteWhileActive:
      return "deleting corelation id mgr context while there are active contexts";
    default:
      break;
  }

  return "unknown error";
}

int
CustomExecute(
    void* custom_context, const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  if (custom_context == nullptr) {
    return kUnknown;
  }

  Context* context = static_cast<Context*>(custom_context);
  return context->Execute(payload_cnt, payloads, input_fn, output_fn);
}

}  // extern "C"

}}}}  // namespace dnapoleone::inferenceserver::correlation_id_mgr::backend
