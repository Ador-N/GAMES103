using System;
using Unity.Mathematics;
using UnityEngine;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

public class CUDAProviderExplicit : Singleton<CUDAProviderExplicit>, ICUDAFunctionProvider
{
    /************************************
     * CUDA functions imported from dll *
     ************************************/
    [DllImport("Parallel_Explicit_SVD.dll", EntryPoint = "CUDA_device_name")]
    protected static extern IntPtr _CUDA_device_name();

    [DllImport("Parallel_Explicit_SVD.dll", EntryPoint = "GetDebugInfo")]
    protected static unsafe extern bool _GetDebugInfo(byte* info);

    [DllImport("Parallel_Explicit_SVD.dll", EntryPoint = "SetDebugTet")]
    protected static extern void _SetDebugTet(int tet);

    [DllImport("Parallel_Explicit_SVD.dll", EntryPoint = "Initialize")]
    protected static unsafe extern void _Initialize(
        int* Tet, float3x3* inv_Dm, float* det_Dm,
        int number, int tet_number, bool useGravity, bool laplacianSmoothing,
        float dt, float s0, float s1, float damp, float mass, float floorY,
        HyperelasticModelType hyperelasticModelType);

    [DllImport("Parallel_Explicit_SVD.dll", EntryPoint = "Update")]
    protected static unsafe extern void __Update(Vector3* X, int iteration_number);

    [DllImport("Parallel_Explicit_SVD.dll", EntryPoint = "Impulse")]
    protected static extern void _Impulse(Vector3 impulse);

    [DllImport("Parallel_Explicit_SVD.dll", EntryPoint = "SetLaplacianOmega")]
    protected static extern void _SetLaplacianOmega(float omega);

    /**************************
     * CUDA functions wrapper *
     **************************/

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public IntPtr CUDA_device_name() => _CUDA_device_name();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe bool GetDebugInfo(byte* info) => _GetDebugInfo(info);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetDebugTet(int tet) => _SetDebugTet(tet);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void Initialize(
        int* Tet, float3x3* inv_Dm, float* det_Dm,
        int number, int tet_number, bool useGravity, bool laplacianSmoothing,
        float dt, float s0, float s1, float damp, float mass, float floorY,
        HyperelasticModelType hyperelasticModelType)
        => _Initialize(
            Tet, inv_Dm, det_Dm,
            number, tet_number, useGravity, laplacianSmoothing,
            dt, s0, s1, damp, mass, floorY,
            hyperelasticModelType);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void _Update(Vector3* X, int iteration_number) => __Update(X, iteration_number);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Impulse(Vector3 impulse) => _Impulse(impulse);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetLaplacianOmega(float omega) => _SetLaplacianOmega(omega);

}