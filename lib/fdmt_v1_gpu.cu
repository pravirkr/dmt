#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <fdmt_v1_gpu.cuh>

#include "npy.hpp" //! delete_

extern cudaError_t cudaStatus0 ;

FDMT_v1_GPU::FDMT_v1_GPU(float f_min, float f_max, size_t nchans, size_t nsamps,
                 float tsamp, size_t dt_max, size_t dt_step, size_t dt_min)
    : FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min)
{
    // Allocate memory for the state buffers
    const auto& plan      = get_plan();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    m_state_in_d.resize(state_size, 0.0F);
    m_state_out_d.resize(state_size, 0.0F);
    transfer_plan_to_device(plan, m_fdmt_plan_d);
    m_nsamps_d.resize(1);
    m_nsamps_d[0] = get_m_nsamps();
}

void FDMT_v1_GPU::transfer_plan_to_device(const FDMTPlan& plan, FDMTPlan_D& plan_d)
{
    // Transfer the plan to the device
    const auto niter = plan.state_shape.size();   
    
    // 1. "coordinates_d" allocation on GPU
      // 1.1 "lenof_innerVects_coords_cumsum" creation and  allocation on GPU
    std::vector< SizeType>h_lenof_innerVects_coords_cumsum_s(niter +1);
    h_lenof_innerVects_coords_cumsum_s[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        h_lenof_innerVects_coords_cumsum_s[i] = h_lenof_innerVects_coords_cumsum_s[i - 1] + plan.coordinates[i - 1].size() *2;
    } 

    std::vector< int>lenof_innerVects_coords_cumsum_h(h_lenof_innerVects_coords_cumsum_s.size());
    //plan_d.lenof_innerVects_coords_cumsum_h.resize(h_lenof_innerVects_coords_cumsum_s.size());
    std::copy(h_lenof_innerVects_coords_cumsum_s.begin(), h_lenof_innerVects_coords_cumsum_s.end(), lenof_innerVects_coords_cumsum_h.begin());

    plan_d.lenof_innerVects_coords_cumsum_d.resize(lenof_innerVects_coords_cumsum_h.size());
    thrust::copy(lenof_innerVects_coords_cumsum_h.begin(), lenof_innerVects_coords_cumsum_h.end(), plan_d.lenof_innerVects_coords_cumsum_d.begin());
    // !1.1

    // 1.2  
    std::vector< SizeType> h_coordinates_flattened_s;      
    for (const auto& innerVec : plan.coordinates)
    {
        for (const auto& pair : innerVec)
        {
            h_coordinates_flattened_s.push_back(pair.first);
            h_coordinates_flattened_s.push_back(pair.second);
        }
    }
    std::vector<int>h_coordinates_flattened(h_coordinates_flattened_s.size());
    std::copy(h_coordinates_flattened_s.begin(), h_coordinates_flattened_s.end(), h_coordinates_flattened.begin());
    plan_d.coordinates_d.resize(h_coordinates_flattened.size());
    thrust::copy(h_coordinates_flattened.begin(), h_coordinates_flattened.end(), plan_d.coordinates_d.begin());   
    
        // !1.2
    //!1

    // 2. "coordinates_to_copy" allocation on GPU
          // 2.1 "lenof_innerVects_coords_to_copy_cumsum" creation and allocation on GPU
    std::vector< SizeType>h_lenof_innerVects_coords_to_copy_cumsum_s(niter + 1);
    h_lenof_innerVects_coords_to_copy_cumsum_s[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        h_lenof_innerVects_coords_to_copy_cumsum_s[i] = h_lenof_innerVects_coords_to_copy_cumsum_s[i - 1] + plan.coordinates_to_copy[i - 1].size() * 2;
    }    

    plan_d.lenof_innerVects_coords_to_copy_cumsum_d.resize(h_lenof_innerVects_coords_to_copy_cumsum_s.size());
    std::vector< int>lenof_innerVects_coords_to_copy_cumsum_h(h_lenof_innerVects_coords_to_copy_cumsum_s.size());
    std::copy(h_lenof_innerVects_coords_to_copy_cumsum_s.begin(), h_lenof_innerVects_coords_to_copy_cumsum_s.end(), lenof_innerVects_coords_to_copy_cumsum_h.begin());
    thrust::copy(lenof_innerVects_coords_to_copy_cumsum_h.begin(), lenof_innerVects_coords_to_copy_cumsum_h.end(), plan_d.lenof_innerVects_coords_to_copy_cumsum_d.begin());

    // !2.1

    // 2.2
    std::vector< SizeType> h_coordinates_to_copy_flattened_s;

    for (const auto& innerVec : plan.coordinates_to_copy)
    {
        for (const auto& pair : innerVec)
        {
            h_coordinates_to_copy_flattened_s.push_back(pair.first);
            h_coordinates_to_copy_flattened_s.push_back(pair.second);
        }
    }
    plan_d.coordinates_to_copy_d.resize(h_coordinates_to_copy_flattened_s.size());

    std::vector<int>h_coordinates_to_copy_flattened(h_coordinates_to_copy_flattened_s.size());

    std::copy(h_coordinates_to_copy_flattened_s.begin(), h_coordinates_to_copy_flattened_s.end(), h_coordinates_to_copy_flattened.begin());  

    thrust::copy(h_coordinates_to_copy_flattened.begin(), h_coordinates_to_copy_flattened.end(), plan_d.coordinates_to_copy_d.begin());
    //!2
    
    // 3. "mappings" allocation on GPU
        // 3.1 "len_mappings_cumsum" creation and allocation on GPU
    std::vector< int>len_mappings_cumsum_h(niter + 1);
    len_mappings_cumsum_h[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        len_mappings_cumsum_h[i] = len_mappings_cumsum_h[i - 1] + plan.mappings[i - 1].size() * 5;
    }
    plan_d.len_mappings_cumsum_d.resize(len_mappings_cumsum_h.size());

    thrust::copy(len_mappings_cumsum_h.begin(), len_mappings_cumsum_h.end(), plan_d.len_mappings_cumsum_d.begin());
    // !3.1

    // 3.2
    std::vector<SizeType> mappings_flattened_s = flatten_mappings_(plan.mappings);
    plan_d.mappings_d.resize(mappings_flattened_s.size());
    std::vector<int>mappings_flattened(mappings_flattened_s.size());
    std::copy(mappings_flattened_s.begin(), mappings_flattened_s.end(), mappings_flattened.begin());
    thrust::copy(mappings_flattened.begin(), mappings_flattened.end(), plan_d.mappings_d.begin()); 
    //!3.2, !3    

    // 4. "mappings_to_copy" allocation on GPU
        // 4.1 "len_mappings_to_copy_cumsum" creation and allocation on GPU
    std::vector< int>len_mappings_to_copy_cumsum_h(niter + 1);
    len_mappings_to_copy_cumsum_h[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        len_mappings_to_copy_cumsum_h[i] = len_mappings_to_copy_cumsum_h[i - 1] + plan.mappings_to_copy[i - 1].size() * 5;
    }
    plan_d.len_mappings_to_copy_cumsum_d.resize(len_mappings_to_copy_cumsum_h.size());

    std::copy(len_mappings_to_copy_cumsum_h.begin(), len_mappings_to_copy_cumsum_h.end(), plan_d.len_mappings_to_copy_cumsum_d.begin());
    // !4.1

    // 4.2
    std::vector<SizeType> mappings_to_copy_flattened_s = flatten_mappings_(plan.mappings_to_copy);
    std::vector<int> mappings_to_copy_flattened(mappings_to_copy_flattened_s.size());
    std::copy(mappings_to_copy_flattened_s.begin(), mappings_to_copy_flattened_s.end(), mappings_to_copy_flattened.begin());

    plan_d.mappings_to_copy_d.resize(mappings_to_copy_flattened_s.size());
    thrust::copy(mappings_to_copy_flattened.begin(), mappings_to_copy_flattened.end(), plan_d.mappings_to_copy_d.begin());
    //!4.2, !4

    // 5."state_sub_idx_d" allocation on GPU
        // 5.1 "len_state_sub_idx_cumsum" creation and allocation on GPU
    std::vector< int>len_state_sub_idx_cumsum_h(niter + 1);
    len_state_sub_idx_cumsum_h[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        len_state_sub_idx_cumsum_h[i] = len_state_sub_idx_cumsum_h[i - 1] + plan.state_sub_idx[i - 1].size() ;
    }
    plan_d.len_state_sub_idx_cumsum_d.resize(len_state_sub_idx_cumsum_h.size());
    std::copy(len_state_sub_idx_cumsum_h.begin(), len_state_sub_idx_cumsum_h.end(), plan_d.len_state_sub_idx_cumsum_d.begin());
    // !5.1

    //5.2
    std::vector<SizeType> flattened_s;
    for (const auto& innerVec : plan.state_sub_idx)
    {
        flattened_s.insert(flattened_s.end(), innerVec.begin(), innerVec.end());
    }
    std::vector <int>flattened(flattened_s.size());
    plan_d.state_sub_idx_d.resize(flattened_s.size());
    std::copy(flattened_s.begin(), flattened_s.end(), flattened.begin());
    thrust::copy(flattened.begin(), flattened.end(), plan_d.state_sub_idx_d.begin());
    
    // !5.2, !5

    // 6. "dt_grid_d" allocation on GPU
        // 6.1 CPU preparations
    std::vector<SizeType> dt_grid_flattened_s;    
    std::vector<SizeType> pos_gridInnerVects_h_s;    
    std::vector<SizeType> pos_gridSubVects_h_s;

    SizeType currentIndex_1star = 0;
    SizeType currentIndex_2star = 0;
    
    for (const auto& subVect : plan.dt_grid)
    {
        // Save the starting point of each subvector in dt_grid
        pos_gridSubVects_h_s.push_back(currentIndex_2star);

        for (const auto& dtGrid : subVect) 
        {
            // Save the starting point of each DtGridType
            pos_gridInnerVects_h_s.push_back(currentIndex_1star);

            // Append elements of dtGrid to the flattened vector
            dt_grid_flattened_s.insert(dt_grid_flattened_s.end(), dtGrid.begin(), dtGrid.end());
            currentIndex_1star += dtGrid.size();
            currentIndex_2star++;
        }
    }
    pos_gridSubVects_h_s.push_back(currentIndex_2star);
    pos_gridInnerVects_h_s.push_back(currentIndex_1star);    

    //
    std::vector<int> dt_grid_flattened(dt_grid_flattened_s.size());
    std::vector<int> pos_gridInnerVects_h(pos_gridInnerVects_h_s.size());
    std::vector<int> pos_gridSubVects_h(pos_gridSubVects_h_s.size());

    std::copy(dt_grid_flattened_s.begin(), dt_grid_flattened_s.end(), dt_grid_flattened.begin());
    std::copy(pos_gridInnerVects_h_s.begin(), pos_gridInnerVects_h_s.end(), pos_gridInnerVects_h.begin());

    plan_d.pos_gridSubVects_h.resize(pos_gridSubVects_h_s.size());
    std::copy(pos_gridSubVects_h_s.begin(), pos_gridSubVects_h_s.end(), plan_d.pos_gridSubVects_h.begin());  

        // 6.2 GPU allocation
    plan_d.dt_grid_d.resize(dt_grid_flattened.size());
    
    plan_d.pos_gridInnerVects_d.resize(pos_gridInnerVects_h_s.size());

    thrust::copy(dt_grid_flattened.begin(), dt_grid_flattened.end(), plan_d.dt_grid_d.begin());
    thrust::copy(pos_gridInnerVects_h.begin(), pos_gridInnerVects_h.end(), plan_d.pos_gridInnerVects_d.begin());
    std::copy(pos_gridSubVects_h_s.begin(), pos_gridSubVects_h_s.end(), plan_d.pos_gridSubVects_h.begin());
    // !7
}
//------------------------------------------------------------
//  Function to flatten the vector of vectors of FDMTCoordMapping
std::vector<SizeType>  flatten_mappings_(const std::vector<std::vector<FDMTCoordMapping>>& mappings)
{
    std::vector<SizeType> flattened;
    
    size_t totalSize = 0;
    for (const auto& vec : mappings) {
        totalSize += vec.size() * 5;  // Each FDMTCoordMapping has 5 SizeType elements
    }
    flattened.reserve(totalSize);
    
    for (const auto& innerVec : mappings) {
        for (const auto& mapping : innerVec) {
            flattened.push_back(mapping.tail.first);
            flattened.push_back(mapping.tail.second);
            flattened.push_back(mapping.head.first);
            flattened.push_back(mapping.head.second);            
            flattened.push_back(mapping.offset);
        }
    }

    return flattened;
}

//-----------------------------------------------------
void  FDMT_v1_GPU::execute(const float* waterfall, size_t waterfall_size, float* dmt,  size_t dmt_size)
{
    check_inputs(waterfall_size, dmt_size);
    float* state_in_ptr = thrust::raw_pointer_cast(m_state_in_d.data());
    float* state_out_ptr = thrust::raw_pointer_cast(m_state_out_d.data());
    //
    const auto& plan = get_plan();
    const auto& dt_grid_init = plan.dt_grid[0];
    const int nchan = dt_grid_init.size();   
    //const int nsamps = plan.state_shape[0][4];
    int *pnsamp = thrust::raw_pointer_cast(m_nsamps_d.data());
    int* pstate_sub_idx = thrust::raw_pointer_cast(m_fdmt_plan_d.state_sub_idx_d.data()); //+
    int* pstate_sub_idx_cumsum = thrust::raw_pointer_cast(m_fdmt_plan_d.len_state_sub_idx_cumsum_d.data());
   
    int* pdt_grid = thrust::raw_pointer_cast(m_fdmt_plan_d.dt_grid_d.data()); //+
   // int* ppos_gridSubVects = thrust::raw_pointer_cast(m_fdmt_plan_d.pos_gridSubVects.data());
    int* ppos_gridInnerVects = thrust::raw_pointer_cast(m_fdmt_plan_d.pos_gridInnerVects_d.data());  //+ 
    
   // int* pstate_sub_idx_cur = &pstate_sub_idx[m_fdmt_plan_d.len_state_sub_idx_cumsum_d[0]]; //+

    int* ppos_gridInnerVects_cur = &ppos_gridInnerVects[m_fdmt_plan_d.pos_gridSubVects_h[0]]; //+
    const dim3 blockSize = dim3(256, 1);
    //const dim3 gridSize = dim3(40, 1);
    const dim3 gridSize = dim3((get_m_nsamps() + blockSize.x - 1) / blockSize.x, nchan);

    auto start = std::chrono::high_resolution_clock::now();
   
        kernel_init_fdmt_v1_ << < gridSize, blockSize >> > (waterfall, pstate_sub_idx, pdt_grid
            , ppos_gridInnerVects_cur, state_in_ptr, pnsamp);
       // cudaDeviceSynchronize();
    

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
   // std::cout << "  Time taken by function initialise : " << duration.count() << " microseconds" << std::endl;

  /*  int lenarr4 = plan.state_shape[0][3] * plan.state_shape[0][4];
    std::vector<float> data4(lenarr4, 0);
    cudaMemcpy(data4.data(), state_in_ptr, lenarr4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::array<long unsigned, 1> leshape2{ lenarr4 };
    npy::SaveArrayAsNumpy("state_gpu.npy", false, leshape2.size(), leshape2.data(), data4);*/

    cudaStatus0 = cudaGetLastError();
    if (cudaStatus0 != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
        return;
    } 

    int* pcoords = thrust::raw_pointer_cast(m_fdmt_plan_d.coordinates_d.data());

    int* pcoords_cumsum = thrust::raw_pointer_cast(m_fdmt_plan_d.lenof_innerVects_coords_cumsum_d.data());
 
    int* pcoords_to_copy = thrust::raw_pointer_cast(m_fdmt_plan_d.coordinates_to_copy_d.data());

    int* pcoords_to_copy_cumsum = thrust::raw_pointer_cast(m_fdmt_plan_d.lenof_innerVects_coords_to_copy_cumsum_d.data());
   
    int* pmappings = thrust::raw_pointer_cast(m_fdmt_plan_d.mappings_d.data());
    int* pmappings_cumsum = thrust::raw_pointer_cast(m_fdmt_plan_d.len_mappings_cumsum_d.data());
    
  
    int* pmappings_to_copy = thrust::raw_pointer_cast(m_fdmt_plan_d.mappings_to_copy_d.data());
    int* pmappings_to_copy_cumsum = thrust::raw_pointer_cast(m_fdmt_plan_d.len_mappings_to_copy_cumsum_d.data());
    
    cudaStatus0 = cudaGetLastError();
    if (cudaStatus0 != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
        return;
    }
    int niters = static_cast<int>(get_niters());
    for (int i_iter = 1; i_iter < niters + 1; ++i_iter)
    {       
        int nsamps = get_m_nsamps();// static_cast<int>(plan.state_shape[i_iter][4]);
        const auto& coords_cur = plan.coordinates[i_iter];        
        const auto& coords_copy_cur = plan.coordinates_to_copy[i_iter];
        
        int  coords_cur_size = static_cast<int>(coords_cur.size());
       
        int  coords_copy_cur_size = coords_copy_cur.size();

     
        int coords_max = std::max(coords_cur_size, coords_copy_cur_size);

        const dim3 blockSize = dim3(256, 1);
        const dim3 gridSize = dim3((nsamps + blockSize.x - 1) / blockSize.x, coords_max);
         kernel_execute_iter_v2_ << < gridSize, blockSize >> > (state_in_ptr
            , state_out_ptr
            , i_iter
            , pcoords
            , pcoords_cumsum          
            , pcoords_to_copy
            , pcoords_to_copy_cumsum
            , pmappings
            , pmappings_cumsum
            , pmappings_to_copy
            , pmappings_to_copy_cumsum
            , pstate_sub_idx
            , pstate_sub_idx_cumsum          
            , pnsamp            
          );
      //  cudaDeviceSynchronize();
        cudaStatus0 = cudaGetLastError();
        if (cudaStatus0 != cudaSuccess)
        {
            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
            return;
        }
     
        
        std::swap(state_in_ptr, state_out_ptr);
        if (i_iter == (niters - 1))
        {
            state_out_ptr = dmt;
        }
    }
  
}
//----------------------------------------------------------------
__global__
void kernel_execute_iter_v2_(const float* __restrict state_in
    , float* __restrict state_out
    , const int i_iter
    , int* __restrict pcoords  
    , int* __restrict  pcoords_cumsum    
    , int* __restrict pcoords_to_copy
    , int* __restrict  pcoords_to_copy_cumsum
    , int* __restrict  pmappings
    , int* __restrict  pmappings_cumsum   
    , int* __restrict pmappings_copy
    , int* __restrict pmappings_to_copy_cumsum
    , int* __restrict pstate_sub_idx
    ,int* __restrict pstate_sub_idx_cumsum   
    , int* pnsamps   
)
{
    __shared__ int iarr_sh[8];
    int  i_coord = blockIdx.y;

    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= (*pnsamps))
    {
        return;
    }
    iarr_sh[6] = pcoords_cumsum[i_iter + 1] - pcoords_cumsum[i_iter];
    int* pcoords_cur = &pcoords[pcoords_cumsum[i_iter]];

    iarr_sh[7] = pcoords_to_copy_cumsum[i_iter + 1] - pcoords_to_copy_cumsum[i_iter];
    int* pcoords_copy_cur = &pcoords_to_copy[pcoords_to_copy_cumsum[i_iter]];
    int* pstate_sub_idx_cur = &pstate_sub_idx[pstate_sub_idx_cumsum[i_iter]];
    int* pstate_sub_idx_prev = &pstate_sub_idx[pstate_sub_idx_cumsum[i_iter - 1]];
    int* pmappings_cur = &pmappings[pmappings_cumsum[i_iter]];
    if (i_coord < iarr_sh[6])
    {    
        iarr_sh[0] = pstate_sub_idx_prev[pmappings_cur[i_coord * 5]] + pmappings_cur[i_coord * 5 + 1] * (*pnsamps);
        iarr_sh[1] = pstate_sub_idx_prev[pmappings_cur[i_coord * 5 + 2]] + pmappings_cur[i_coord * 5 + 3] * (*pnsamps);
        iarr_sh[2] = pstate_sub_idx_cur[pcoords_cur[i_coord * 2]] + pcoords_cur[i_coord * 2 + 1] * (*pnsamps);
        iarr_sh[3] = pmappings_cur[i_coord * 5 + 4]; // offset
        //---
        if (isamp < iarr_sh[3])
        {
           state_out[iarr_sh[2] + isamp] = state_in[iarr_sh[0] + isamp];
        }
        else
        {
         state_out[iarr_sh[2] + isamp] = state_in[iarr_sh[0] + isamp] + state_in[iarr_sh[1] + isamp - iarr_sh[3]];
        }
    }
    __syncthreads();

    if (i_coord < iarr_sh[7])
    {
        int* pmappings_copy_cur = &pmappings_copy[i_iter];
        int i_sub = pcoords_copy_cur[i_coord * 2];       
        int state_sub_idx = pstate_sub_idx_cur[i_sub];
        
        iarr_sh[4] = pstate_sub_idx_prev[pmappings_copy_cur[i_coord * 5]] + pmappings_copy_cur[i_coord * 5 + 1] * (*pnsamps);
        iarr_sh[5] = pstate_sub_idx_cur[pcoords_copy_cur[i_coord * 2]] + pcoords_copy_cur[i_coord * 2 + 1] * (*pnsamps);
        //--
        state_out[iarr_sh[5] + isamp] = state_in[iarr_sh[4] + isamp];  
    }
}
//--------------------------------------------------------------
void  FDMT_v1_GPU::initialise(const float* waterfall, float* state)
{

 }
//--------------------------------------------------------------
__global__
void  kernel_init_fdmt_(const float* __restrict waterfall, int* __restrict pstate_sub_idx_cur
    , int* __restrict p_dt_grid, int* __restrict ppos_gridInnerVects_cur, float* __restrict state, const int nsamps)
{
    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= nsamps)
    {
        return;
    }
    int i_sub = blockIdx.y;
    //// Initialise state for [:, dt_init_min, dt_init_min:]
    int  dt_grid_sub_min = p_dt_grid[ppos_gridInnerVects_cur[i_sub]];

    // int  state_sub_idx = pstate_sub_idx_cur[p_len_state_sub_idx_cumsum[0] + i_sub];
    int  state_sub_idx = pstate_sub_idx_cur[i_sub];
    if (isamp >= dt_grid_sub_min)
    {
        float sum = 0.0F;
        for (int i = isamp - dt_grid_sub_min; i <= isamp; ++i)
        {
            sum += waterfall[i_sub * nsamps + i];
        }
        state[state_sub_idx + isamp] = sum / static_cast<float>(dt_grid_sub_min + 1);
    }
    ////---
    int  dt_grid_sub_size = ppos_gridInnerVects_cur[i_sub + 1] - ppos_gridInnerVects_cur[i_sub];

    for (int i_dt = 1; i_dt < dt_grid_sub_size; ++i_dt)
    {
        int dt_cur = p_dt_grid[ppos_gridInnerVects_cur[i_sub] + i_dt];// dt_grid_sub[i_dt];
        int dt_prev = p_dt_grid[ppos_gridInnerVects_cur[i_sub] + i_dt - 1];
        float sum = 0.0F;
        if (isamp >= dt_cur)
        {
            for (int i = isamp - dt_cur; i < isamp - dt_prev; ++i)
            {
                sum += waterfall[i_sub * nsamps + i];
            }
            state[state_sub_idx + i_dt * nsamps + isamp] = (state[state_sub_idx + (i_dt - 1) * nsamps + isamp] *
                (static_cast<float>(dt_prev) + 1.0F) + sum) / (static_cast<float>(dt_cur) + 1.0F);
        }
        else
        {
            state[state_sub_idx + i_dt * nsamps + isamp] = 0.0F;  // ???? rid of it???
        }
    }
}

//--------------------------------------------------------------
__global__
void  kernel_init_fdmt_v1_(const float* __restrict waterfall, int* __restrict pstate_sub_idx_cur
    , int* __restrict p_dt_grid, int* __restrict ppos_gridInnerVects_cur, float* __restrict state,  int *pnsamps)
{
  // __shared__ int iarr_sh[4];
    int iarr_sh[4];
    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= (*pnsamps))
    {
        return;
    }
    int i_sub = blockIdx.y;
    ////// Initialise state for [:, dt_init_min, dt_init_min:]
   

    iarr_sh[0] =  p_dt_grid[ppos_gridInnerVects_cur[i_sub]]; // =dt_grid_sub_min    

    iarr_sh[1] = pstate_sub_idx_cur[i_sub]; // = state_sub_idx

    iarr_sh[2] = ppos_gridInnerVects_cur[i_sub + 1] - ppos_gridInnerVects_cur[i_sub]; //= dt_grid_sub_size

    iarr_sh[3] = ppos_gridInnerVects_cur[i_sub];
    
    if (isamp >= iarr_sh[0])
    {
        float sum = 0.0F;
        for (int i = isamp - iarr_sh[0]; i <= isamp; ++i)
        {
            sum += waterfall[i_sub * (*pnsamps) + i];
        }
        state[iarr_sh[1] + isamp] = fdividef(sum, static_cast<float>(iarr_sh[0] + 1))  ;
    }
    //////---   

    for (int i_dt = 1; i_dt < iarr_sh[2]; ++i_dt)
    {
        int dt_cur = p_dt_grid[iarr_sh[3] + i_dt];// dt_grid_sub[i_dt];
        int dt_prev = p_dt_grid[iarr_sh[3] + i_dt - 1];
        float sum = 0.0F;
        if (isamp >= dt_cur)
        {
            for (int i = isamp - dt_cur; i < isamp - dt_prev; ++i)
            {
                sum += waterfall[i_sub * (*pnsamps) + i];
            }
            state[iarr_sh[1] + i_dt * (*pnsamps) + isamp] = state[iarr_sh[1] + (i_dt - 1) * (*pnsamps) + isamp] *
                fdividef(static_cast<float>(dt_prev) + 1.0F + sum, static_cast<float>(dt_cur) + 1.0F) ;
        }
        
    }
}
//--------------------------------------------------------------------------------------
__global__
void kernel_execute_iter_(const float* __restrict state_in, float* __restrict state_out
    , int* __restrict pcoords_cur//+   
    , int* __restrict pmappings_cur //+
    , int* __restrict pcoords_copy_cur //+
    , int* __restrict pmappings_copy_cur //+
    , int* __restrict pstate_sub_idx_cur //+
    , int* __restrict pstate_sub_idx_prev//+
    , int nsamps //+
    , int  coords_cur_size//+
    , int  coords_copy_cur_size//+
)
{
    int  i_coord = blockIdx.y;

    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= nsamps)
    {
        return;
    }

    if (i_coord < coords_cur_size)
    {
        int i_sub = pcoords_cur[i_coord * 2];
        int i_dt = pcoords_cur[i_coord * 2 + 1];
        int i_sub_tail = pmappings_cur[i_coord * 5];
        int i_dt_tail = pmappings_cur[i_coord * 5 + 1];
        int i_sub_head = pmappings_cur[i_coord * 5 + 2];
        int i_dt_head = pmappings_cur[i_coord * 5 + 3];
        int offset = pmappings_cur[i_coord * 5 + 4];
        int state_sub_idx = pstate_sub_idx_cur[i_sub];
        int state_sub_idx_tail = pstate_sub_idx_prev[i_sub_tail];
        int state_sub_idx_head = pstate_sub_idx_prev[i_sub_head];

        const float* __restrict tail = &state_in[state_sub_idx_tail + i_dt_tail * nsamps];
        const float* __restrict head = &state_in[state_sub_idx_head + i_dt_head * nsamps];
        float* __restrict out = &state_out[state_sub_idx + i_dt * nsamps];
        if (isamp < offset)
        {
            out[isamp] = tail[isamp];
        }
        else
        {
            out[isamp] = tail[isamp] + head[isamp - offset];
        }
    }
    __syncthreads();

    if (i_coord < coords_copy_cur_size)
    {
        int i_sub = pcoords_copy_cur[i_coord * 2];
        int i_dt = pcoords_copy_cur[i_coord * 2 + 1];
        int i_sub_tail = pmappings_copy_cur[i_coord * 5];
        int i_dt_tail = pmappings_copy_cur[i_coord * 5 + 1];
        int state_sub_idx = pstate_sub_idx_cur[i_sub];
        int state_sub_idx_tail = pstate_sub_idx_prev[i_sub_tail];

        const float* __restrict tail = &state_in[state_sub_idx_tail + i_dt_tail * nsamps];
        float* __restrict out = &state_out[state_sub_idx + i_dt * nsamps];
        out[isamp] = tail[isamp];
    }
}
////--------------------------------------------------------------------------------------
//__global__
//void kernel_execute_iter_v1_(const float* __restrict state_in, float* __restrict state_out
//    , int* __restrict pcoords_cur//+   
//    , int* __restrict pmappings_cur //+
//    , int* __restrict pcoords_copy_cur //+
//    , int* __restrict pmappings_copy_cur //+
//    , int* __restrict pstate_sub_idx_cur //+
//    , int* __restrict pstate_sub_idx_prev//+
//    , int nsamps //+
//    , int  coords_cur_size//+
//    , int  coords_copy_cur_size//+
//)
//{
//    __shared__ int iarr_sh[6];
//    int  i_coord = blockIdx.y;
//
//    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
//    if (isamp >= nsamps)
//    {
//        return;
//    }
//
//    if (i_coord < coords_cur_size)
//    {
//        int i_sub = pcoords_cur[i_coord * 2];
//        int i_dt = pcoords_cur[i_coord * 2 + 1];
//        int i_sub_tail = pmappings_cur[i_coord * 5];
//        int i_dt_tail = pmappings_cur[i_coord * 5 + 1];
//        int i_sub_head = pmappings_cur[i_coord * 5 + 2];
//        int i_dt_head = pmappings_cur[i_coord * 5 + 3];
//       
//        int state_sub_idx = pstate_sub_idx_cur[i_sub];
//        int state_sub_idx_tail = pstate_sub_idx_prev[i_sub_tail];
//        int state_sub_idx_head = pstate_sub_idx_prev[i_sub_head];
//
//        iarr_sh[0] = state_sub_idx_tail + i_dt_tail * nsamps;
//        iarr_sh[1] = state_sub_idx_head + i_dt_head * nsamps;
//        iarr_sh[2] = state_sub_idx + i_dt * nsamps;
//        iarr_sh[3] = pmappings_cur[i_coord * 5 + 4]; // offset
//        //---
//       // const float* tail = &state_in[iarr_sh[0]];
//       // const float* head = &state_in[iarr_sh[1]];
//       // float* out = &state_out[iarr_sh[2]];
//        if (isamp < iarr_sh[3])
//        {
//            state_out[iarr_sh[2] + isamp] = state_in[iarr_sh[0] + isamp];
//           
//        }
//        else
//        {
//            state_out[iarr_sh[2] + isamp] = state_in[iarr_sh[0] + isamp] + state_in[iarr_sh[1] + isamp - iarr_sh[3]];
//        }
//    }
//    __syncthreads();
//
//    if (i_coord < coords_copy_cur_size)
//    {
//        int i_sub = pcoords_copy_cur[i_coord * 2];
//        int i_dt = pcoords_copy_cur[i_coord * 2 + 1];
//        int i_sub_tail = pmappings_copy_cur[i_coord * 5];
//        int i_dt_tail = pmappings_copy_cur[i_coord * 5 + 1];
//        int state_sub_idx = pstate_sub_idx_cur[i_sub];
//        int state_sub_idx_tail = pstate_sub_idx_prev[i_sub_tail];
//
//        iarr_sh[4] = state_sub_idx_tail + i_dt_tail * nsamps;
//        iarr_sh[5] = state_sub_idx + i_dt * nsamps;
//        //--
//
//       // const float* tail = &state_in[iarr_sh[4]];
//       // float* out = &state_out[iarr_sh[5]];
//        state_out[iarr_sh[5] + isamp] =  state_in[iarr_sh[4] + isamp] ;
//        //out[isamp] = tail[isamp];
//    }
//}

//--------------------------------------------------------------------------------------
__global__
void kernel_execute_iter_v1_(const float* __restrict state_in, float* __restrict state_out
    , int* __restrict pcoords_cur//+   
    , int* __restrict pmappings_cur //+
    , int* __restrict pcoords_copy_cur //+
    , int* __restrict pmappings_copy_cur //+
    , int* __restrict pstate_sub_idx_cur //+
    , int* __restrict pstate_sub_idx_prev//+
    , int* pnsamps //+
    , int  coords_cur_size//+
    , int  coords_copy_cur_size//+
)
{
    __shared__ int iarr_sh[6];
    int  i_coord = blockIdx.y;

    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= (*pnsamps))
    {
        return;
    }

    if (i_coord < coords_cur_size)
    {
        int i_sub = pcoords_cur[i_coord * 2];
        //  int i_dt = pcoords_cur[i_coord * 2 + 1];
         // int i_sub_tail = pmappings_cur[i_coord * 5];
         // int i_dt_tail = pmappings_cur[i_coord * 5 + 1];
          //int i_sub_head = pmappings_cur[i_coord * 5 + 2];
         // int i_dt_head = pmappings_cur[i_coord * 5 + 3];

        int state_sub_idx = pstate_sub_idx_cur[i_sub];
        // int state_sub_idx_tail = pstate_sub_idx_prev[i_sub_tail];
        // int state_sub_idx_head = pstate_sub_idx_prev[i_sub_head];

        iarr_sh[0] = pstate_sub_idx_prev[pmappings_cur[i_coord * 5]] + pmappings_cur[i_coord * 5 + 1] * (*pnsamps);
        iarr_sh[1] = pstate_sub_idx_prev[pmappings_cur[i_coord * 5 + 2]] + pmappings_cur[i_coord * 5 + 3] * (*pnsamps);
        iarr_sh[2] = pstate_sub_idx_cur[i_sub] + pcoords_cur[i_coord * 2 + 1] * (*pnsamps);
        iarr_sh[3] = pmappings_cur[i_coord * 5 + 4]; // offset
        //---
       // const float* tail = &state_in[iarr_sh[0]];
       // const float* head = &state_in[iarr_sh[1]];
       // float* out = &state_out[iarr_sh[2]];
        if (isamp < iarr_sh[3])
        {
            state_out[iarr_sh[2] + isamp] = state_in[iarr_sh[0] + isamp];

        }
        else
        {
            state_out[iarr_sh[2] + isamp] = state_in[iarr_sh[0] + isamp] + state_in[iarr_sh[1] + isamp - iarr_sh[3]];
        }
    }
    __syncthreads();

    if (i_coord < coords_copy_cur_size)
    {
        int i_sub = pcoords_copy_cur[i_coord * 2];
        //int i_dt = pcoords_copy_cur[i_coord * 2 + 1];
       // int i_sub_tail = pmappings_copy_cur[i_coord * 5];
       // int i_dt_tail = pmappings_copy_cur[i_coord * 5 + 1];
        int state_sub_idx = pstate_sub_idx_cur[i_sub];
        //int state_sub_idx_tail = pstate_sub_idx_prev[i_sub_tail];

        iarr_sh[4] = pstate_sub_idx_prev[pmappings_copy_cur[i_coord * 5]] + pmappings_copy_cur[i_coord * 5 + 1] * (*pnsamps);
        iarr_sh[5] = pstate_sub_idx_cur[pcoords_copy_cur[i_coord * 2]] + pcoords_copy_cur[i_coord * 2 + 1] * (*pnsamps);
        //--

       // const float* tail = &state_in[iarr_sh[4]];
       // float* out = &state_out[iarr_sh[5]];
        state_out[iarr_sh[5] + isamp] = state_in[iarr_sh[4] + isamp];
        //out[isamp] = tail[isamp];
    }
}
