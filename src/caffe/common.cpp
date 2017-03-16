#include <boost/thread.hpp>
#include <glog/logging.h>
#include <svm.h>

#include <cmath>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

// Make sure each thread can have different values.
static boost::thread_specific_ptr<Caffe> thread_instance_;

Caffe& Caffe::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Caffe());
  }
  return *(thread_instance_.get());
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, "
              "using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  pid = getpid();
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
    : random_generator_(), mode_(Caffe::CPU),
      solver_count_(1), root_solver_(true) { }

Caffe::~Caffe() { }

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int Caffe::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe()
    : cublas_handle_(NULL), curand_generator_(NULL), random_generator_(),
    mode_(Caffe::CPU), solver_count_(1), root_solver_(true) {
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
  model_ = svm_load_model("/tmp/gemm.model");
  int nNumAttr = 12;
  node_ = (struct svm_node *) malloc(nNumAttr*sizeof(struct svm_node));
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, current_device));
  node_[0].index = 1;
  node_[1].index = 2;
  node_[2].index = 3;
  node_[3].index = 4; 
  node_[4].index = 5;
  node_[5].index = 6;
  node_[6].index = 7; 
  node_[7].index = 8; 
  node_[0].value = prop.major/6.0;              
  node_[1].value = prop.minor/7.0;              
  node_[2].value = prop.totalGlobalMem/double(11995578368.0);     
  node_[3].value = prop.multiProcessorCount/20.0;
  node_[4].value = prop.clockRate/double(1733500.0);          
  node_[5].value = prop.memoryClockRate/double(5005000.0);  
  node_[6].value = prop.memoryBusWidth/384.0;     
  node_[7].value = prop.l2CacheSize/double(2097152.0);        

  //init xgboost
  LOG(INFO) << "Start to init xgboost ......";
  int nCols=8, nRows=1;
  float train[nRows][nCols];
  for (int i=0;i<nRows;i++)
      for (int j=0;j<nCols;j++)
          train[i][j] = (i+1) * (j+1);
  XGDMatrixCreateFromMat((float *) train, nRows, nCols, 0, &(xgTrain_[0]));
  XGBoosterCreate(xgTrain_, 1, &xgBooster_);
  //XGBoosterLoadModel(xgBooster_, "/tmp/xgboostgemm2.model");
  XGBoosterLoadModel(xgBooster_, "/tmp/xgboostgemmpascal.model");
  XGBoosterSetParam(xgBooster_, "booster", "gbtree");
  XGBoosterSetParam(xgBooster_, "objective", "binary:logistic");
  XGBoosterSetParam(xgBooster_, "max_depth", "8");
  XGBoosterSetParam(xgBooster_, "eta", "1");
  XGBoosterSetParam(xgBooster_, "gamma", "0");
  XGBoosterSetParam(xgBooster_, "silent", "1");
  XGBoosterSetParam(xgBooster_, "nthread", "1");
  xgSampleFeatures_[0][0] = prop.totalGlobalMem;
  xgSampleFeatures_[0][1] = prop.multiProcessorCount;
  xgSampleFeatures_[0][2] = prop.clockRate;
  xgSampleFeatures_[0][3] = prop.memoryBusWidth;
  xgSampleFeatures_[0][4] = prop.l2CacheSize;
  //xgSampleFeatures_[0][7] = prop.sharedMemPerBlock;
  //xgSampleFeatures_[0][8] = prop.totalConstMem;
  //xgSampleFeatures_[0][9] = prop.regsPerBlock;

  bst_ulong out_len;
  const float *f;
  XGDMatrixCreateFromMat((float *) train, 1, nCols, -1, &xgTestSample_);
  XGBoosterPredict(xgBooster_, xgTestSample_, 0,8,&out_len,&f);
  LOG(INFO) << "Init xgboost finished!"; 
}

Caffe::~Caffe() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
  if (model_) {
      svm_free_and_destroy_model(&model_);
  }
  if (node_) {
      free(node_);
  }
  XGDMatrixFree(xgTrain_[0]);
  XGBoosterFree(xgBooster_);
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  static bool g_curand_availability_logged = false;
  if (Get().curand_generator_) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
    CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  } else {
    if (!g_curand_availability_logged) {
        LOG(ERROR) <<
            "Curand not available. Skipping setting the curand seed.";
        g_curand_availability_logged = true;
    }
  }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
  if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  }
  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
      cluster_seedgen()));
}

void Caffe::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  LOG(INFO) << "Device id:                     " << device;
  LOG(INFO) << "Major revision number:         " << prop.major;
  LOG(INFO) << "Minor revision number:         " << prop.minor;
  LOG(INFO) << "Name:                          " << prop.name;
  LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  LOG(INFO) << "Warp size:                     " << prop.warpSize;
  LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum dimension of block:    "
      << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
      << prop.maxThreadsDim[2];
  LOG(INFO) << "Maximum dimension of grid:     "
      << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
      << prop.maxGridSize[2];
  LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  LOG(INFO) << "Concurrent copy and execution: "
      << (prop.deviceOverlap ? "Yes" : "No");
  LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  LOG(INFO) << "Kernel execution timeout:      "
      << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  return;
}


double Caffe::predict(size_t nWA, size_t nHA, size_t nHB) {

  Get().node_[8].index = 9;
  Get().node_[8].value = log2(double(nHA));
  Get().node_[9].index = 10;
  Get().node_[9].value = log2(double(nHB));
  Get().node_[10].index = 11;
  Get().node_[10].value = log2(double(nWA));
  Get().node_[11].index = -1;
  double predict_label = svm_predict(Get().model_, Get().node_);
  ///LOG(INFO) << "Feature: " << Get().node_[0].value << Get().node_[1].value << Get().node_[10].value;
  ///LOG(INFO) << "Feature0: " << Get().node_[0].value; 
  ///LOG(INFO) << "Feature2: " << Get().node_[2].value; 
  ///LOG(INFO) << "Feature8: " << Get().node_[8].value; 
  ///LOG(INFO) << "Feature9: " << Get().node_[9].value; 
  ///LOG(INFO) << "Feature10: " << Get().node_[10].value; 
  ///LOG(INFO) << "Predict Label:      " << predict_label;
  return predict_label;
}

float Caffe::xgPredict(size_t nWA, size_t nHA, size_t nHB) {
   // return 1.0;
  //LOG(INFO) << "Start to predict ";
  xgSampleFeatures_[0][5] = nHA;
  xgSampleFeatures_[0][6] = nHB;
  xgSampleFeatures_[0][7] = nWA;
  XGDMatrixCreateFromMat((float *) xgSampleFeatures_, 1, 8, -1, &xgTestSample_);
  //LOG(INFO) << "Create matrix finished:      ";
  bst_ulong out_len;
  const float *f;
  XGBoosterPredict(xgBooster_, xgTestSample_, 0, 8, &out_len,&f);
  float label = f[0];
  //LOG(INFO) << "Predict Label:      " << label;
  //LOG(INFO) << "Out length:      " << out_len;
  //XGBoosterFree(xgTestSample);
  return label;
}

class Caffe::RNG::Generator {
 public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) { }

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}



#endif  // CPU_ONLY

}  // namespace caffe
