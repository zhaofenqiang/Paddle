#include "ACLOperator.hpp"

#ifdef USE_PROFILING

#include "arm_neon.h"

unsigned int acl_log_flags = (0 | \
                              MASK_LOG_APP_TIME | \
                            /*MASK_LOG_ALLOCATE | */\
                            /*MASK_LOG_ALLOCATE | */\
                            /*MASK_LOG_RUN      | */\
                            /*MASK_LOG_CONFIG   | */\
                            /*MASK_LOG_COPY     | */\
                              MASK_LOG_ABSVAL   | \
                              MASK_LOG_BNLL     | \
                              MASK_LOG_CONV     | \
                              MASK_LOG_FC       | \
                              MASK_LOG_LRN      | \
                              MASK_LOG_POOLING  | \
                              MASK_LOG_RELU     | \
                              MASK_LOG_SIGMOID  | \
                              MASK_LOG_SOFTMAX  | \
                              MASK_LOG_TANH     | \
                              MASK_LOG_LC       | \
                              MASK_LOG_BN       | \
                              MASK_LOG_CONCAT   | \
                              0);                                          
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* getenv */
#endif //USE_PROFILING

namespace paddle {
//bool AclEnableSchedule(int enable){
//    enable_schedule=enable;
//    if (enable) {
//        Caffe::set_mode(Caffe::GPU);
//    }
//    return true;
//}
//int isScheduleEnable()
//{
//    return enable_schedule;
//}

bool ACLOperator::init_cl_env = true;
bool ACLOperator::support_opencl_ = false;
bool opencl_is_available()
{
    return arm_compute::opencl_is_available();
}

bool ACLOperator::new_tensor(std::unique_ptr<ACLTensor> &tensor,arm_compute::TensorShape &shape,void *mem,bool commit)
{
    auto acl_tensor = new ACLTensor(arm_compute::TensorInfo(shape, arm_compute::Format::F32));
    acl_tensor->set_target(getTargetHint());
    acl_tensor->bindmem(mem);
    if (commit) acl_tensor->commit();
    tensor=(std::unique_ptr<ACLTensor>) std::move(acl_tensor);
    return true;
}
bool ACLOperator::new_tensor(std::unique_ptr<ACLSubTensor> &tensor,std::unique_ptr<ACLTensor> &parent,arm_compute::TensorShape &shape,arm_compute::Coordinates& coord)
{
    auto acl_tensor=new ACLSubTensor(parent,shape, coord);
    acl_tensor->set_target(getTargetHint());
    tensor=(std::unique_ptr<ACLSubTensor>) std::move(acl_tensor);
    return true;
}

void ACLTensor::commit(TensorType type)
{
    settensortype(type);
    if (mem_) {
        if (!allocate_){ 
#ifdef USE_PROFILING
            logtime_util log_time(ACL_ALLOCATE_INFO);
#endif //USE_PROFILING
            allocate(); 
            allocate_=true;
        }
        if (type_!= tensor_output) {
           tensor_copy(mem_);
        }
        mem_=nullptr;
    }
}

int BaseACLTensor::tensor_copy(arm_compute::ITensor* tensor,void * mem,bool toTensor)
{
#ifdef USE_PROFILING
    logtime_util log_time(ACL_COPY_INFO);
#endif //USE_PROFILING
    arm_compute::Window window;
    window.use_tensor_dimensions(tensor->info()->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY); // Iterate through the rows (not each element)
    int width = tensor->info()->tensor_shape()[0]; 
    int height = tensor->info()->tensor_shape()[1];
    int deepth = tensor->info()->tensor_shape()[2];
    map();
    // Create an iterator:
    arm_compute::Iterator it(tensor, window);
    // Except it works for an arbitrary number of dimensions
    if (toTensor) { //mem->tensor
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
        {
                memcpy(it.ptr(), ((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width + id.x()) * tensor->info()->element_size()), width * tensor->info()->element_size());
        },
        it);
    }else{ //tensor-->mem
        arm_compute::execute_window_loop(window, [&](const arm_compute::Coordinates & id)
        {
                memcpy(((char*)mem) + ((id[3] * (width * height * deepth) + id.z() * (width * height) + id.y() * width) * tensor->info()->element_size()), it.ptr(), width * tensor->info()->element_size());
        },
        it);
    }
    unmap();

    return 0;
}

}
