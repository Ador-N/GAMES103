#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "float3x3.hu"
#include "float3operators.hu"
#include "math.h"

#include "cstring_device.hu"
#include <cstdio>
#include "printf/printf.hu"

float3 *Force, *d_X, *V;

int *d_Tet;
float3x3 *d_inv_Dm;
float *d_det_Dm;

float3 *V_sum;
int *V_num;

int number = 0, tet_number = 0;
float dt, s0, s1, damp, mass, floorY;

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

__device__ float2 __solve2x2_sym(float l1, float l2, float2 b)
{
    float detA = l1 * l1 - l2 * l2;
    return make_float2(
        (l1 * b.x - l2 * b.y) / detA,
        (l1 * b.y - l2 * b.x) / detA);
}

__global__ void _calcStress(float3 *X, float3 *Force, int *Tet, float3x3 *inv_Dm, float *det_Dm, int tet_number, float s0, float s1)
{
    int tet = blockIdx.x * blockDim.x + threadIdx.x;
    if (tet >= tet_number)
        return;

    float3x3 F = Build_Edge_Matrix(X, Tet, tet) * inv_Dm[tet];

    float3x3 U, V;
    float3 L;
    F.svd(U, L, V);

    // Perturb L to avoid instabilities when lambdas are close to each other
    const float eps = 1e-6;
    if (L.x - L.y < eps)
        L.x += eps;
    if (L.y - L.z < eps)
        L.y += eps;
    if (L.x - L.y < eps)
        L.x += eps;

    float I = sqrMagnitude(L);
    float3 S = L * L;
    float3x3 dWdLd = float3x3(
        0.5f * L.x * (I - 3) * s0 + 2 * L.x * (S.x - 1) * s1,
        0.5f * L.y * (I - 3) * s0 + 2 * L.y * (S.y - 1) * s1,
        0.5f * L.z * (I - 3) * s0 + 2 * L.z * (S.z - 1) * s1);
    float3x3 P = U * dWdLd * V.transpose();
    float3x3 force = (-det_Dm[tet] / 6) * P * inv_Dm[tet].transpose();

    // Atomic add
    atomicAdd(&Force[Tet[tet * 4 + 0]], -force * make_float3(1, 1, 1));
    atomicAdd(&Force[Tet[tet * 4 + 1]], force.getColumn(0));
    atomicAdd(&Force[Tet[tet * 4 + 2]], force.getColumn(1));
    atomicAdd(&Force[Tet[tet * 4 + 3]], force.getColumn(2));

    float3x3 d2WdL2 = s0 * outer(L, L) + float3x3(0.5f * s0 * (I - 3)) + s1 * float3x3(3 * S - make_float3(1, 1, 1));
    float3x3 Ld = float3(L), dfmi_dFnl[3][3]; // Where the subscripts are n, l

    float3x3 debug0[3][3], debug1[3][3];

    for (int k = 0; k < 3; k++)
        for (int l = 0; l < 3; l++)
        {
            float3x3 dPdFkl, omegaU, omegaVt;

            // Calculate \frac{\partial\lambda_d}{\partial F_{kl}},
            // leading to central addend of \frac{\partial f}{\partial F_{kl}}
            float3x3 Ut_dFdFkl_V = outer(U.getRow(k), V.getRow(l));
            dPdFkl += float3x3(d2WdL2 * Ut_dFdFkl_V.diag());

            // Calculate U\frac{\partial U}{\partial F_{kl}} and \frac{\partial V^T}{\partial F_{kl}}V,
            // leading to the first and last addend of \frac{\partial f}{\partial F_{kl}}
            float2 U_Vt_01 = __solve2x2_sym(L.y, L.x, make_float2(Ut_dFdFkl_V.m01, -Ut_dFdFkl_V.m10));
            float2 U_Vt_02 = __solve2x2_sym(L.z, L.x, make_float2(Ut_dFdFkl_V.m02, -Ut_dFdFkl_V.m20));
            float2 U_Vt_12 = __solve2x2_sym(L.z, L.y, make_float2(Ut_dFdFkl_V.m12, -Ut_dFdFkl_V.m21));
            omegaU.m01 = U_Vt_01.x, omegaVt.m01 = U_Vt_01.y;
            omegaU.m02 = U_Vt_02.x, omegaVt.m02 = U_Vt_02.y;
            omegaU.m12 = U_Vt_12.x, omegaVt.m12 = U_Vt_12.y;
            omegaU += -omegaU.transpose(), omegaVt += -omegaVt.transpose();
            dPdFkl += omegaU * dWdLd + dWdLd * omegaVt;
            dPdFkl = U * dPdFkl * V.transpose();
            debug1[k][l] = dPdFkl;

            // Work out \frac{\partial f}{\partial F_{kl}} and add it to the total derivative
            dfmi_dFnl[k][l] = (-det_Dm[tet] / 6) * U * dPdFkl * V.transpose() * inv_Dm[tet].transpose();
            debug0[k][l] = Ut_dFdFkl_V - omegaU * Ld - Ld * omegaVt;
        }

    // Work out Hessian = (\frac{\vec{f}_i}{\vec{x}_j})_{mn} = \frac{\partial f_{mi}}{\partial F_{nl}} * X^{-T}_{lj}
    // for all {i, j}s, where i, j are the indices of the vertices of the tetrahedron and m, n are the indices of Hessians
    // Which means we'd do 9 matrix multiplications to get 9 Hessian blocks.
    float3x3 Ht[3][3]; // Ht[i][j] = \frac{\vec{f}_i}{\vec{x}_j}
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
            float3x3 H;
            for (int m = 0; m < 3; m++)
                for (int n = 0; n < 3; n++)
                {
                    H(m, n) = dfmi_dFnl[n][0](m, i) * inv_Dm[tet](j, 0)    //
                              + dfmi_dFnl[n][1](m, i) * inv_Dm[tet](j, 1)  //
                              + dfmi_dFnl[n][2](m, i) * inv_Dm[tet](j, 2); //
                }
            Ht[i][j] = H;
        }

    if (debug_tet_id != -1 && tet == debug_tet_id)
    {
        char *tail = debug_info + strlen_d(debug_info);
        /*tail = strcat_d(tail, "inv_Dm: ");
        tail = to_string(inv_Dm[tet], tail);
        tail = strcat_d(tail, "\nF: ");
        tail = to_string(F, tail);
        tail = strcat_d(tail, "\nF - UlVt: ");
        tail = to_string(F - U * float3x3(L) * V.transpose(), tail);
        tail = strcat_d(tail, "\nI: ");
        tail += sprintf_(tail, "%f", I);
        tail = strcat_d(tail, "\nS: ");
        tail = to_string(S, tail);
        tail = strcat_d(tail, "\ndWdL: ");
        tail = to_string(dWdL, tail);
        tail = strcat_d(tail, "\nP: ");
        tail = to_string(P, tail);
        tail = strcat_d(tail, "\nforce: ");
        tail = to_string(force, tail);
        tail = strcat_d(tail, "\n");*/
        tail = strcat_d(tail, "Ht: \n");
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                tail = to_string(Ht[i][j], tail);
                tail = strcat_d(tail, "\n");
            }
        /*tail = strcat_d(tail, "debug0: \n");
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                tail = to_string(debug0[i][j], tail);
                tail = strcat_d(tail, "\n");
            }*/
        tail = strcat_d(tail, "debug1: \n");
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                tail = to_string(debug1[i][j], tail);
                tail = strcat_d(tail, "\n");
            }
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

void _update()
{
    int grid_size_vert = (number + 255) / 256;
    int grid_size_tet = (tet_number + 255) / 256;

    _preUpdate<<<grid_size_vert, 256>>>(Force, number, useGravity);
    cudaDeviceSynchronize();

    _calcStress<<<grid_size_tet, 256>>>(d_X, Force, d_Tet, d_inv_Dm, d_det_Dm, tet_number, s0, s1);
    cudaDeviceSynchronize();

    _particleUpdate<<<grid_size_vert, 256>>>(d_X, V, Force, number, dt, damp, mass, floorY);
    cudaDeviceSynchronize();

    if (laplacianSmoothing && V_sum && V_num)
    {
        cudaMemset(V_sum, 0, number * sizeof(float3));
        cudaMemset(V_num, 0, number * sizeof(int));
        _laplacianSmoothingTet<<<grid_size_tet, 256>>>(V, V_sum, V_num, d_Tet, tet_number);
        cudaDeviceSynchronize();
        _laplacianSmoothingVert<<<grid_size_vert, 256>>>(V, V_sum, V_num, number, 0.67);
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
        sprintf(label, "(%s) -- %d, %d", device.name, number, tet_number);
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
        float dt, float s0, float s1, float damp, float mass, float floorY)
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
}