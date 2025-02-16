using UnityEngine;
public class FVM_Implicit_SVD_Parallel : FVM_Parallel_base<CUDAProviderImplicit>
{
    new void Start()
    {
        updatesPerFixedUpdate = 1;
        base.Start();
    }
    public int iterationsPerFixedUpdate = 60;
    protected override void FixedUpdate()
    {
        if (debug)
            provider.SetDebugTet(0);
        unsafe
        {
            fixed (Vector3* xPtr = X)
                provider._Update(xPtr, iterationsPerFixedUpdate);
        }
    }
}
