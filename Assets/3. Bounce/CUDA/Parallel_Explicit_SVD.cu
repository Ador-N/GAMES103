#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "float3x3.hu"
#include "float3operators.hu"
#include "math.h"

#include "cstring_device.hu"
#include <cstdio>
#include "printf/printf.hu"

enum HyperelasticModelType
{
    StVK,
    neoHookean
} modelType;

float3 *Force, *d_X, *V;

int *d_Tet;
float3x3 *d_inv_Dm;
float *d_det_Dm;

float3 *V_sum;
int *V_num;

int number = 0, tet_number = 0;
float dt, s0, s1, damp, mass, floorY, omega = 0.67;

bool useGravity = true, laplacianSmoothing = true;

#define MAX_DEBUG_BUFFER_SIZE 2048
__managed__ int debug_tet_id = -1, debug_info_size = 0;
__managed__ char debug_info[MAX_DEBUG_BUFFER_SIZE];

__device__ __forceinline__
    float3x3
    Build_Edge_Matrix(float3 *X, int *Tet, int tet)
{
    float3x3 ret;
    // TODO: Need to build edge matrix here.
    float3 X0 = X[Tet[tet * 4 + 0]];
    float3 X1 = X[Tet[tet * 4 + 1]];
    float3 X2 = X[Tet[tet * 4 + 2]];
    float3 X3 = X[Tet[tet * 4 + 3]];

    ret.setColumn(0, X0 - X1);
    ret.setColumn(1, X0 - X2);
    ret.setColumn(2, X0 - X3);

    return ret;
}

__global__ void _preUpdate(float3 *Force, int number, bool useGravity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number)
        return;

    Force[i] = make_float3(0, -9.81, 0) * useGravity;
}

template <typename T>
__global__ void _calcStress(
    float3 *X, float3 *Force,
    int *Tet, float3x3 *inv_Dm, float *det_Dm,
    int tet_number, float s0, float s1,
    T hyperElasticModel)
{
    int tet = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet >= tet_number)
        return;

    float3x3 F = Build_Edge_Matrix(X, Tet, tet) * inv_Dm[tet];

    float3x3 U, V;
    float3 l;
    F.svd(U, l, V);
    float3 dWdL = hyperElasticModel(l, s0, s1);

    float3x3 P = U * float3x3(dWdL) * V.transpose();
    float3x3 force = (-det_Dm[tet] / 6) * P * inv_Dm[tet].transpose();

    // Atomic add
    atomicAdd(&Force[Tet[tet * 4 + 0]], -force * make_float3(1, 1, 1));
    atomicAdd(&Force[Tet[tet * 4 + 1]], force.getColumn(0));
    atomicAdd(&Force[Tet[tet * 4 + 2]], force.getColumn(1));
    atomicAdd(&Force[Tet[tet * 4 + 3]], force.getColumn(2));

    if (debug_tet_id != -1 && tet == debug_tet_id)
    {
        char *tail = debug_info + strlen_d(debug_info);
        tail = strcat_d(tail, "inv_Dm: ");
        tail = to_string(inv_Dm[tet], tail);
        tail = strcat_d(tail, "\nF: ");
        tail = to_string(F, tail);
        tail = strcat_d(tail, "\nF - UlVt: ");
        tail = to_string(F - U * float3x3(l) * V.transpose(), tail);
        tail = strcat_d(tail, "\ndWdL: ");
        tail = to_string(dWdL, tail);
        tail = strcat_d(tail, "\nP: ");
        tail = to_string(P, tail);
        tail = strcat_d(tail, "\nforce: ");
        tail = to_string(force, tail);
        tail = strcat_d(tail, "\n");
        debug_tet_id = -1;
    }
}

__global__ void _particleUpdate(
    float3 *X, float3 *V, float3 *Force, int number,
    float dt, float damp, float mass, float floorY)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number)
        return;

    // Explicit Euler
    V[i] += Force[i] * dt / mass;
    V[i] *= damp;
    X[i] += V[i] * dt;

    // Collision
    if (X[i].y < floorY)
    {
        V[i].y += (floorY - X[i].y) / dt;
        X[i].y = floorY;
    }
}

__global__ void _laplacianSmoothingTet(float3 *V, float3 *V_sum, int *V_num, int *Tet, int tet_number)
{
    int tet = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet >= tet_number)
        return;

    float3 v = V[Tet[tet * 4 + 0]] + V[Tet[tet * 4 + 1]] + V[Tet[tet * 4 + 2]] + V[Tet[tet * 4 + 3]];
    atomicAdd(&V_sum[Tet[tet * 4 + 0]], v - V[Tet[tet * 4 + 0]]);
    atomicAdd(&V_sum[Tet[tet * 4 + 1]], v - V[Tet[tet * 4 + 1]]);
    atomicAdd(&V_sum[Tet[tet * 4 + 2]], v - V[Tet[tet * 4 + 2]]);
    atomicAdd(&V_sum[Tet[tet * 4 + 3]], v - V[Tet[tet * 4 + 3]]);
    atomicAdd(&V_num[Tet[tet * 4 + 0]], 1);
    atomicAdd(&V_num[Tet[tet * 4 + 1]], 1);
    atomicAdd(&V_num[Tet[tet * 4 + 2]], 1);
    atomicAdd(&V_num[Tet[tet * 4 + 3]], 1);
}

// Laplacian smoothing for vertices, omega is the relaxation factor (0.0 - 1.0, 0 for no smoothing)
__global__ void _laplacianSmoothingVert(float3 *V, float3 *V_sum, int *V_num, int number, float omega)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number)
        return;

    V[i] = omega * V_sum[i] / (V_num[i] * 3) + (1 - omega) * V[i];
}

// General update function, accepting a function pointer for the update function
template <typename T>
__global__ void verticesUpdate(float3 *arr, int number, T update)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number)
        return;

    update(arr, i);
}

#define CALC_STRESS_MODEL_CASE(m)                                                                                  \
    case m:                                                                                                        \
        _calcStress<<<grid_size_tet, 256>>>(d_X, Force, d_Tet, d_inv_Dm, d_det_Dm, tet_number, s0, s1, model_##m); \
        break;

void _update()
{
    int grid_size_vert = (number + 255) / 256;
    int grid_size_tet = (tet_number + 255) / 256;

    _preUpdate<<<grid_size_vert, 256>>>(Force, number, useGravity);
    cudaDeviceSynchronize();

    auto model_StVK = [] __device__(float3 l, float s0, float s1)
    {
        float I = sqrMagnitude(l);
        float3 S = l * l;
        float3 dWdL = make_float3(
            0.5f * l.x * (I - 3) * s0 + 2 * l.x * (S.x - 1) * s1,
            0.5f * l.y * (I - 3) * s0 + 2 * l.y * (S.y - 1) * s1,
            0.5f * l.z * (I - 3) * s0 + 2 * l.z * (S.z - 1) * s1);
        return dWdL;
    };

    auto model_neoHookean = [] __device__(float3 l, float s0, float s1)
    {
        float3 S = l * l;
        float I = sqrMagnitude(l),
              III = S.x * S.y * S.z,
              rcbrtIII = rcbrt(III);
        float3 dIdL = make_float3(2 * l.x, 2 * l.y, 2 * l.z),
               dIIIdL = make_float3(S.y * S.z, S.x * S.z, S.x * S.y) * dIdL;
        float dWdI = s0 * rcbrtIII,
              dWdIII = s1 * (1 - rsqrt(III)) - dWdI * I / 3 / III;
        float3 dWdL = dWdI * dIdL + dWdIII * dIIIdL;
        return dWdL;
    };

    switch (modelType)
    {
        CALC_STRESS_MODEL_CASE(StVK)
        CALC_STRESS_MODEL_CASE(neoHookean)
    default:
        _calcStress<<<grid_size_tet, 256>>>(d_X, Force, d_Tet, d_inv_Dm, d_det_Dm, tet_number, s0, s1, model_StVK);
        break;
    }
    cudaDeviceSynchronize();

    _particleUpdate<<<grid_size_vert, 256>>>(d_X, V, Force, number, dt, damp, mass, floorY);
    cudaDeviceSynchronize();

    if (laplacianSmoothing && V_sum && V_num)
    {
        cudaMemset(V_sum, 0, number * sizeof(float3));
        cudaMemset(V_num, 0, number * sizeof(int));
        _laplacianSmoothingTet<<<grid_size_tet, 256>>>(V, V_sum, V_num, d_Tet, tet_number);
        cudaDeviceSynchronize();
        _laplacianSmoothingVert<<<grid_size_vert, 256>>>(V, V_sum, V_num, number, omega);
        cudaDeviceSynchronize();
    }
}

extern "C"
{
    __export__ char *CUDA_device_name()
    {
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, 0);
        char *label = new char[256];
        sprintf(label, "(%s) -- %d, %d", modelType == StVK ? "StVK" : "neoHookean", number, tet_number);
        return label;
    }

    __export__ bool GetDebugInfo(char *info)
    {
        memcpy(info, debug_info, MAX_DEBUG_BUFFER_SIZE);
        memset(debug_info, 0, MAX_DEBUG_BUFFER_SIZE);
        return true;
    }

    __export__ void SetDebugTet(int tet_id)
    {
        debug_tet_id = tet_id;
    }

    __export__ void Initialize(
        int *Tet, float3x3 *inv_Dm, float *det_Dm,
        int number, int tet_number, bool useGravity, bool laplacianSmoothing,
        float dt, float s0, float s1, float damp, float mass, float floorY,
        HyperelasticModelType modelType)
    {
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        // if (number != ::number)
        {
            if (::number)
            {
                cudaFree(Force);
                cudaFree(V);
                cudaFree(V_sum);
                cudaFree(V_num);
                cudaFree(d_X);
            }
            ::number = number;
            cudaMalloc(&V, number * sizeof(float3));
            cudaMalloc(&Force, number * sizeof(float3));
            cudaMalloc(&V_sum, number * sizeof(float3));
            cudaMalloc(&V_num, number * sizeof(int));
            cudaMalloc(&d_X, number * sizeof(float3));
        }

        // if (tet_number != ::tet_number)
        {
            if (::tet_number)
            {
                cudaFree(d_Tet);
                cudaFree(d_inv_Dm);
                cudaFree(d_det_Dm);
            }
            ::tet_number = tet_number;
            cudaMalloc(&d_Tet, tet_number * 4 * sizeof(int));
            cudaMalloc(&d_inv_Dm, tet_number * sizeof(float3x3));
            cudaMalloc(&d_det_Dm, tet_number * sizeof(float));
        }

        cudaMemset(V, 0, number * sizeof(float3));
        cudaMemset(Force, 0, number * sizeof(float3));
        cudaMemset(V_sum, 0, number * sizeof(float3));
        cudaMemset(V_num, 0, number * sizeof(int));

        cudaMemcpy(d_Tet, Tet, tet_number * 4 * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inv_Dm, inv_Dm, tet_number * sizeof(float3x3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_det_Dm, det_Dm, tet_number * sizeof(float), cudaMemcpyHostToDevice);

        ::dt = dt;
        ::s0 = s0;
        ::s1 = s1;
        ::damp = damp;
        ::mass = mass;
        ::floorY = floorY;
        ::useGravity = useGravity;
        ::laplacianSmoothing = laplacianSmoothing;
        ::modelType = modelType;
    }

    __export__ void Update(float3 *X, int iteration_number)
    {
        cudaMemcpy(d_X, X, number * sizeof(float3), cudaMemcpyHostToDevice);

        for (int i = 0; i < iteration_number; i++)
        {
            _update();
        }

        cudaMemcpy(X, d_X, number * sizeof(float3), cudaMemcpyDeviceToHost);
    }

    __export__ void Impulse(float3 impulse)
    {
        verticesUpdate<<<(number + 255) / 256, 256>>>(
            V, number,
            [=] __device__(float3 * V, int i)
            { V[i] += impulse; });
        cudaDeviceSynchronize();
    }

    __export__ void SetLaplacianOmega(float _omega)
    {
        omega = _omega;
    }
}