#include <iostream>
#include <sstream>
#include <fstream>
#include "cuda_runtime.h"
#include <memory>
#include "./params.h"
#include "./bevdet.h"
#include <vector>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}
void Getinfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}
void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    int count=0;
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.l << " ";
          ofs << box.w << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << "\n";
          count++;
          if(count==boxes.size()){
            break;
            }
        }
          // ofs << box.id << " ";
          // ofs << box.score << " ";
          
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[len];
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}


int main(int argc, const char **argv)
{
    Getinfo();
    std::string model_type=argv[1];
    Params params_;
    std::vector<Bndbox> nms_pred;
    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;
    cudaStream_t stream = NULL;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreate(&stream));
    
    int driver=0;
    int runtime=0;

    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);

    printf("Cuda Runtime Version = %d\n", runtime);
    printf("Cuda Driver  Version = %d\n", driver);
    std::vector<float> point_range = {0, -40, -3, 70.4, 40, 1};
    std::vector<int> size = {1600, 1408}; 
    std::string model_path="/home/nvidia/way/Fast_det/fastDet.onnx";
    fs::path path("/home/nvidia/way/Fast_det/data/kitti/training/velodyne");
    std::cout<<model_type<<std::endl;
    if(model_type=="waymo"){
      model_path="/home/nvidia/way/Fast_det/fastDet_waymo.onnx";
      size={1504,1504};
      point_range={-75.2,-75.2,-2,75.2,75.2,4};
      path= fs::path("/home/nvidia/way/Fast_det/data/waymo/waymo_processed_data_v0_5_0/segment-15832924468527961_1564_160_1584_160_with_camera_labels_bin");
      std::cout<<"waymo"<<std::endl;
    }
    BEVDet bevdet(model_path, stream);

    float x_scale_factor = size[1] / (point_range[3] - point_range[0]);
    float y_scale_factor = size[0] / (point_range[4] - point_range[1]);
    
    for(const auto &entry : fs::directory_iterator(path)){
      if(!fs::is_directory(entry.path())){
        auto start1 = std::chrono::system_clock::now();
        std::string dataFile=entry.path().string();
        std::stringstream ss;
        std::cout << "<<<<<<<<<<<" <<std::endl;
        std::cout << "load file: "<< dataFile <<std::endl;
        //load points cloud
        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
        loadData(dataFile.data(), &data, &length);
        buffer.reset((char *)data);
        float* points = (float*)buffer.get();
        size_t points_size = length/sizeof(float)/4;
        std::cout << "find points num: "<< points_size <<std::endl;
        while(points_size==0){
            loadData(dataFile.data(), &data, &length);
            buffer.reset((char *)data);
            points = (float*)buffer.get();
            points_size = length/sizeof(float)/4;
        }
        float *points_data = nullptr;
        unsigned int points_data_size = points_size * 4 * sizeof(float);
        checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
        checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
        checkCudaErrors(cudaDeviceSynchronize());
        cudaEventRecord(start, stream);
       
        // 总大小 = 1 * 2 * 1600 * 1408
        std::vector<float> bev(1 * 2 * size[0] * size[1], -10.0f);
        std::fill(bev.begin() + size[0] * size[1], bev.end(), 0.0f); // 将强度图部分设置为0.0f
        for (int i = 0; i < points_size; i++) {
            float4 point = ((float4*)points)[i];
            int x_index = static_cast<int>((point.x) * x_scale_factor);
            if(model_type=="waymo"){
              x_index = static_cast<int>((point.x-point_range[0]) * x_scale_factor);

            }
            int y_index = static_cast<int>((point.y - point_range[1]) * y_scale_factor);
            if (x_index >= 0 && x_index < size[1] && y_index >= 0 && y_index < size[0]) {
                int height_index = y_index * size[1] + x_index; // 高度图索引
                int intensity_index = size[0] * size[1] + y_index * size[1] + x_index; // 强度图索引
                float& height_val = bev[height_index];
                float& intensity_val = bev[intensity_index];
                if (point.z > height_val) height_val = point.z;
                if (point.w > intensity_val) intensity_val = point.w;
            }
        }
       
        // std::ofstream outfile("/data/xqm/click2box/bevdet/output.txt");
        // for (int i = 0; i < dim1; i++) {
        //     for (int j = 0; j < dim2; j++) {
        //         for (int k = 0; k < dim3; k++) {
        //             for (int l = 0; l < dim4; l++) {
        //                 int index = i * (dim2 * dim3 * dim4) + j * (dim3 * dim4) + k * dim4 + l;
        //                 outfile <<bev[index] << " ";
        //             }
        //             outfile << std::endl;
        //         }
        //         outfile << std::endl;
        //     }
        // }
        // outfile.close();
        
        nms_pred.reserve(100);
        
        float* hostArray =bev.data();
        // 在 GPU 上分配内存
        float* deviceArray;
        cudaMalloc(&deviceArray, bev.size() * sizeof(float));
        // 将数据从主机拷贝到 GPU
        cudaMemcpy(deviceArray, hostArray, bev.size() * sizeof(float), cudaMemcpyHostToDevice);
        bevdet.doinfer(deviceArray,nms_pred);
        std::cout<<"Bndbox objs: "<< nms_pred.size()<<std::endl;
        //SaveBoxPred(nms_pred, save_file_name);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start1);
        std::cout <<"all infer time:"<< elapsed.count() <<"ms" << '\n';
        checkCudaErrors(cudaFree(points_data));
        checkCudaErrors(cudaFree(deviceArray));
        nms_pred.clear();
        }
      
    }
 



    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
  
    return 0;
}
