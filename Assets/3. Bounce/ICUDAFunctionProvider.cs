using System;
using Unity.Mathematics;
using UnityEngine;

public interface ICUDAFunctionProvider
{
    abstract IntPtr CUDA_device_name();

    abstract unsafe bool GetDebugInfo(byte* info);

    abstract void SetDebugTet(int tet);

    abstract unsafe void Initialize(
        int* Tet, float3x3* inv_Dm, float* det_Dm,
        int number, int tet_number, bool useGravity, bool enableLaplacianSmoothing,
        float dt, float s0, float s1, float damp, float mass, float floorY = -3);

    abstract unsafe void _Update(Vector3* X, int iteration_number);

    abstract void Impulse(Vector3 impulse);

}