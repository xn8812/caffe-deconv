// Copyright 2014 BVLC and contributors.
//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.
//
// Adding caffe::solver class

#include <string>
#include <vector>

#include "mex.h"
#include "caffe/caffe.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

// The interim macro that simply strips the excess and ends up with the required macro
#define CHECK_EQ_X(A,B,C,FUNC, ...) FUNC

// The macro that the programmer uses
#define CHECK_EQ(...)  CHECK_EQ_X(__VA_ARGS__,  \
                       CHECK_EQ_3(__VA_ARGS__),    \
                       CHECK_EQ_2(__VA_ARGS__))

#define CHECK_EQ_2(a, b)  do {                                  \
  if ( (a) != (b) ) {                                           \
    fprintf(stderr, "%s:%d: Check failed because %s != %s\n",   \
            __FILE__, __LINE__, #a, #b);                        \
    mexErrMsgTxt("Error in CHECK_EQ");                          \
  }                                                             \
} while (0);

#define CHECK_EQ_3(a, b, m)  do {                               \
  if ( (a) != (b) ) {                                           \
    fprintf(stderr, "%s:%d: Check failed because %s != %s\n",   \
            __FILE__, __LINE__, #a, #b);                        \
    fprintf(stderr, "%s:%d: %s\n",                              \
            __FILE__, __LINE__, #m);                            \
    mexErrMsgTxt(#m);                                           \
  }                                                             \
} while (0);

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess, "CUDA_CHECK")  \
  } while (0)

using namespace caffe;  // NOLINT(build/namespaces)

// The pointer to the internal caffe::Net instance
//static shared_ptr<Net<float> > net_;
// The pointer to the internal caffe::Solver instance
static shared_ptr<Solver<float> > solver_;
static int init_key = -2;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.

static mxArray* do_forward(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = solver_->net()->input_blobs();
  //mexPrintf("blob size: %d\n",input_blobs.size());
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
      input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(input_blobs[i]->mutable_cpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(input_blobs[i]->mutable_gpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  //LOG(INFO) << "transfer data to GPU";
  const vector<Blob<float>*>& output_blobs = solver_->net()->ForwardPrefilled();
  //LOG(INFO) << "forward";
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(data_ptr, output_blobs[i]->cpu_data(),
          sizeof(float) * output_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
          sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  //LOG(INFO) << "transfer data to CPU";
  return mx_out;
}

/*
static mxArray* do_forward_test(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = solver_->test_nets_->input_blobs();
  //mexPrintf("blob size: %d\n",input_blobs.size());
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]),
      input_blobs.size());
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(input_blobs[i]->mutable_cpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(input_blobs[i]->mutable_gpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = solver_->test_nets_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {output_blobs[i]->width(), output_blobs[i]->height(),
      output_blobs[i]->channels(), output_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(data_ptr, output_blobs[i]->cpu_data(),
          sizeof(float) * output_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
          sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}
*/

static mxArray* do_backward(const mxArray* const top_diff) {
  vector<Blob<float>*>& output_blobs = solver_->net()->output_blobs();
  vector<Blob<float>*>& input_blobs = solver_->net()->input_blobs();
  
  //LOG(INFO) << "input blobs number: " << solver_->net()->num_inputs();

  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(top_diff)[0]),
      output_blobs.size());

  // First, copy the output diff
  // LOG(INFO) << "input blobs number: " << solver_->net()->num_inputs();
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(output_blobs[i]->mutable_cpu_diff(), data_ptr,
        sizeof(float) * output_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(output_blobs[i]->mutable_gpu_diff(), data_ptr,
        sizeof(float) * output_blobs[i]->count(), cudaMemcpyHostToDevice);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  solver_->net()->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(data_ptr, input_blobs[i]->cpu_diff(),
          sizeof(float) * input_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(data_ptr, input_blobs[i]->gpu_diff(),
          sizeof(float) * input_blobs[i]->count(), cudaMemcpyDeviceToHost);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = solver_->net()->layers();
  const vector<string>& layer_names = solver_->net()->layer_names();

  // char* c_layer_name = mxArrayToString(layer_names);
  // LOG(INFO) << c_layer_name; 
  // Step 1: count the number of layers with weights
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"layer_names","weights"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {layer_blobs.size(), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};

        mxArray* mx_weights =
          mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        switch (Caffe::mode()) {
        case Caffe::CPU:
          memcpy(weights_ptr, layer_blobs[j]->cpu_data(),
              sizeof(float) * layer_blobs[j]->count());
          break;
        case Caffe::GPU:
          CUDA_CHECK(cudaMemcpy(weights_ptr, layer_blobs[j]->gpu_data(),
              sizeof(float) * layer_blobs[j]->count(), cudaMemcpyDeviceToHost));
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
      }
    }
  }

  return mx_layers;
}


static void do_set_layer_weights(const mxArray* const layer_name,
    const mxArray* const mx_layer_weights) {
  const vector<shared_ptr<Layer<float> > >& layers = solver_->net_->layers();
  //LOG(INFO) << "Done with get solver layers.";
  const vector<string>& layer_names = solver_->net_->layer_names();
  //LOG(INFO) << "Done with get solver layer names.";

  char* c_layer_names = mxArrayToString(layer_name);
  LOG(INFO) << "Looking for: " << c_layer_names;

  for (unsigned int i = 0; i < layers.size(); ++i) {
    DLOG(INFO) << layer_names[i];
    if (strcmp(layer_names[i].c_str(),c_layer_names) == 0) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      DLOG(INFO) << "Found layer " << layer_names[i] << "layer_blobs.size() = " << layer_blobs.size();
      CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(mx_layer_weights)[0]),
        layer_blobs.size(), "Num of cells don't match layer_blobs.size");
      LOG(INFO) << "layer_blobs.size() = " << layer_blobs.size();
      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        const mxArray* const elem = mxGetCell(mx_layer_weights, j);
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};
        DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
        CHECK_EQ(layer_blobs[j]->count(), mxGetNumberOfElements(elem),
          "Numel of weights don't match count of layer_blob");
        const mwSize* dims_elem = mxGetDimensions(elem);
        DLOG(INFO) << dims_elem[0] << " " << dims_elem[1];
        const float* const data_ptr =
            reinterpret_cast<const float* const>(mxGetPr(elem));
        DLOG(INFO) << "elem: " << data_ptr[0] << " " << data_ptr[1];
        DLOG(INFO) << "count: " << layer_blobs[j]->count();
        switch (Caffe::mode()) {
        case Caffe::CPU:
          memcpy(layer_blobs[j]->mutable_cpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count());
          break;
        case Caffe::GPU:
          cudaMemcpy(layer_blobs[j]->mutable_gpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count(), cudaMemcpyHostToDevice);
          break;
        default:
          LOG(FATAL) << "Unknown Caffe mode.";
        }
      }
    }
  }
}

static mxArray* do_get_all_data() {
  const vector<shared_ptr<Blob<float> > >& blobs = solver_->net_->blobs();
  const vector<string>& blob_names = solver_->net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_data;
  {
    const int num_blobs[1] = {blobs.size()};
    const char* fnames[2] = {"name", "data"};
    mx_all_data = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {
    DLOG(INFO) << blob_names[i];
    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};
    DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
    mxArray* mx_blob_data =
      mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);

    float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));

    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(blob_data_ptr, blobs[i]->cpu_data(),
          sizeof(float) * blobs[i]->count());
      break;
    case Caffe::GPU:
      CUDA_CHECK(cudaMemcpy(blob_data_ptr, blobs[i]->gpu_data(),
          sizeof(float) * blobs[i]->count(), cudaMemcpyDeviceToHost));
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
    mxSetField(mx_all_data, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_data, i, "data",mx_blob_data);
  }
  return mx_all_data;
}

static mxArray* do_get_all_diff() {
  const vector<shared_ptr<Blob<float> > >& blobs = solver_->net_->blobs();
  const vector<string>& blob_names = solver_->net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_diff;
  {
    const int num_blobs[1] = {blobs.size()};
    const char* fnames[2] = {"name", "diff"};
    mx_all_diff = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {
    DLOG(INFO) << blob_names[i];
    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};
    DLOG(INFO) << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3];
    mxArray* mx_blob_data =
      mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);

    float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));

    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(blob_data_ptr, blobs[i]->cpu_diff(),
          sizeof(float) * blobs[i]->count());
      break;
    case Caffe::GPU:
      CUDA_CHECK(cudaMemcpy(blob_data_ptr, blobs[i]->gpu_diff(),
          sizeof(float) * blobs[i]->count(), cudaMemcpyDeviceToHost));
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
    mxSetField(mx_all_diff, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_diff, i, "diff",mx_blob_data);
  }
  return mx_all_diff;
}

static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}


static void set_weights(MEX_ARGS) {
 if (nrhs != 1) {
     LOG(ERROR) << "Given " << nrhs << " arguments expecting 1";
     mexErrMsgTxt("Wrong number of arguments");
  }
  const mxArray* const mx_weights = prhs[0];
  if (!mxIsStruct(mx_weights)) {
     mexErrMsgTxt("Input needs to be struct");
  }
  int num_layers = mxGetNumberOfElements(mx_weights);
  // LOG(INFO) << "begin set layers with layer number: " << num_layers;	 
  for (int i = 0; i < num_layers; ++i) {
    const mxArray* layer_name= mxGetField(mx_weights,i,"layer_names");
    const mxArray* weights= mxGetField(mx_weights,i,"weights");
    do_set_layer_weights(layer_name,weights);
  }
}

static void get_all_data(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_all_data();
}

/*
static void set_all_data(MEX_ARGS) {
 if (nrhs != 1) {
     LOG(ERROR) << "Given " << nrhs << " arguments expecting 1";
     mexErrMsgTxt("Wrong number of arguments");
  }
  const mxArray* const mx_blobs = prhs[0];
  if (!mxIsStruct(mx_blobs)) {
     mexErrMsgTxt("Input needs to be struct");
  }
  int num_blobs = mxGetNumberOfElements(mx_blobs);
  // LOG(INFO) << "begin set blobs with blob number: " << num_blobs;	 
  for (int i = 0; i < num_blobs; ++i) {
    const mxArray* blob_name= mxGetField(mx_blobs,i,"blob_names");
    const mxArray* blobs= mxGetField(mx_blobs,i,"blobs");
    do_set_data(blob_name,blobs);
  }
}
*/

static void get_all_diff(MEX_ARGS) {
  if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  plhs[0] = do_get_all_diff();
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  Caffe::set_phase(Caffe::TRAIN);
}

static void set_phase_test(MEX_ARGS) {
  Caffe::set_phase(Caffe::TEST);
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_device(MEX_ARGS) {
    if (nrhs != 0) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }
  Caffe::DeviceQuery();
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  //char* param_file = mxArrayToString(prhs[0]);
//  char* model_file = mxArrayToString(prhs[1]);
  char* solver_file = mxArrayToString(prhs[0]);
  SolverParameter solver_param;
  ReadProtoFromTextFile(solver_file, &solver_param);
//  std::cout << "has read proto text file" << std::endl;
//  solver_->Init(solver_file);
  solver_.reset(GetSolver<float>(solver_param));
  solver_->iter_ = 0;
  // net_ = solver_->net_;
  // mexPrintf("%s\n",solver_->net_->name().c_str());
  // mexPrintf("input blob number: %d\n",solver_->net_->num_inputs());
  // solver_->net_->CopyTrainedLayersFrom(string(model_file));

  //vector<Blob<float>*>& input_blobs = solver_->net()->input_blobs();
  //mexPrintf("blob output size: %d\n",solver_->net()->num_outputs());
  
  //mxFree(param_file);
//  mxFree(model_file);
  mxFree(solver_file);

  init_key = random();  // NOLINT(caffe/random_fn)

  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

/*
static void reset(MEX_ARGS) {
  if (solver_) {
    solver_->ResetNet();
    init_key = -2;
    LOG(INFO) << "Network reset, call init before use it again";
  }
}
*/

static void snapshot(MEX_ARGS){
  if (solver_) {
	solver_->Snapshot();
	LOG(INFO) << "Snapshot done...";
  } 
}

static void presolve(MEX_ARGS){
  if (solver_) {
	solver_->PreSolve();
	LOG(INFO) << "Presolve done...";
  } 
}

static void update(MEX_ARGS) {
  if (solver_->net_) {
//	LOG(INFO) << "Begin update";
	solver_->ComputeUpdateValue();
    solver_->iter_++;
//	LOG(INFO) << "Compute updated values";
	solver_->net()->Update();
//    LOG(INFO) << "Network updated";    
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = do_forward(prhs[0]);
}

/*
static void forward_test(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = do_forward_test(prhs[0]);
}
*/

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  plhs[0] = do_backward(prhs[0]);
}


static void is_initialized(MEX_ARGS) {
 //mexPrintf("init...");
   	if (!solver_) {
    plhs[0] = mxCreateDoubleScalar(0);
	//mexPrintf("plhs0...");
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
	//mexPrintf("plhs1...");
  }
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "snapshot",		  snapshot    },	
  { "backward",           backward        },
  { "init",               init            },
  { "presolve",           presolve        },
  { "update",             update          },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  { "get_device",         get_device      },
  { "get_weights",        get_weights     },
  { "set_weights",        set_weights     },
  { "get_all_diff",       get_all_diff    },
  { "get_all_data",       get_all_data    },
  { "get_init_key",       get_init_key    },
//  { "reset",              reset           },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "No API command given";
    mexErrMsgTxt("An API command is requires");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("API command not recognized");
    }
    mxFree(cmd);
  }
}
