//#define DEBUG_SINGLE_TETRAHEDRON

using System;
using Unity.Mathematics;
using System.IO;
using UnityEngine;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

public abstract class FVM_Parallel_base<T> : MonoBehaviour where T : Singleton<T>, ICUDAFunctionProvider, new()
{
    static ICUDAFunctionProvider provider = Singleton<T>.Instance;
    byte[] debug_info = new byte[2048];
    string caption;
    float dt = 0.0005f;
    int updatesPreFixedUpdate = 10;
    float slowDownFactor = 1;
    float mass = 1;
    float stiffness_0 = 20000.0f;
    float stiffness_1 = 5000.0f;
    float damp = 0.99999f;

    int[] Tet;
    int tet_number;         //The number of tetrahedra

    /*public */
    Vector3[] X;
    int number;             //The number of vertices

    float3x3[] inv_Dm;
    float[] det_Dm;

    public HyperelasticModelType hyperelasticModelType = HyperelasticModelType.StVK;
    public bool debug = false;
    public bool useGravity = true;
    public bool laplacianSmoothing = true;

    // Start is called before the first frame update
    void Start()
    {
#if !DEBUG_SINGLE_TETRAHEDRON
        // FILO IO: Read the house model from files.
        // The model is from Jonathan Schewchuk's Stellar lib.
        {
            string fileContent = File.ReadAllText("Assets/3. Bounce/house2.ele");
            string[] Strings = fileContent.Split(" \t\r\n".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);

            tet_number = int.Parse(Strings[0]);
            Tet = new int[tet_number * 4];

            for (int tet = 0; tet < tet_number; tet++)
            {
                Tet[tet * 4 + 0] = int.Parse(Strings[tet * 5 + 4]) - 1;
                Tet[tet * 4 + 1] = int.Parse(Strings[tet * 5 + 5]) - 1;
                Tet[tet * 4 + 2] = int.Parse(Strings[tet * 5 + 6]) - 1;
                Tet[tet * 4 + 3] = int.Parse(Strings[tet * 5 + 7]) - 1;
            }
        }
        {
            string fileContent = File.ReadAllText("Assets/3. Bounce/house2.node");
            string[] Strings = fileContent.Split(" \t\r\n".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
            number = int.Parse(Strings[0]);
            X = new Vector3[number];
            for (int i = 0; i < number; i++)
            {
                X[i].x = float.Parse(Strings[i * 5 + 5]) * 0.4f;
                X[i].y = float.Parse(Strings[i * 5 + 6]) * 0.4f;
                X[i].z = float.Parse(Strings[i * 5 + 7]) * 0.4f;
            }
            //Centralize the model.
            Vector3 center = Vector3.zero;
            for (int i = 0; i < number; i++) center += X[i];
            center /= number;
            for (int i = 0; i < number; i++)
            {
                X[i] -= center;
                (X[i].z, X[i].y) = (X[i].y, X[i].z);
            }
        }
#else
        tet_number = 1;
        Tet = new int[tet_number * 4];
        Tet[0] = 0;
        Tet[1] = 1;
        Tet[2] = 2;
        Tet[3] = 3;

        number = 4;
        X = new Vector3[number];
        X[0] = new Vector3(0, 0, 0);
        X[1] = new Vector3(1, 0, 0);
        X[2] = new Vector3(0, 1, 0);
        X[3] = new Vector3(0, 0, 1);

#endif

        //Create triangle mesh.
        Vector3[] vertices = new Vector3[tet_number * 12];
        int vertex_number = 0;
        for (int tet = 0; tet < tet_number; tet++)
        {
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
        }

        int[] triangles = new int[tet_number * 12];
        for (int t = 0; t < tet_number * 4; t++)
        {
            triangles[t * 3 + 0] = t * 3 + 0;
            triangles[t * 3 + 1] = t * 3 + 1;
            triangles[t * 3 + 2] = t * 3 + 2;
        }
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();

        //TODO: Need to allocate and assign inv_Dm
        inv_Dm = new float3x3[tet_number];
        det_Dm = new float[tet_number];
        for (int i = 0; i < tet_number; i++)
        {
            Matrix4x4 Dm = Build_Edge_Matrix(i);
            // Deal with row-major/column-major struct definition between Unity and CUDA.
            inv_Dm[i] = new float3x3(Dm.inverse.transpose);
            det_Dm[i] = Dm.determinant;
        }

        // Init CUDA
        unsafe
        {
            fixed (int* TetPtr = Tet)
            fixed (float3x3* inv_DmPtr = inv_Dm)
            fixed (float* det_DmPtr = det_Dm)
                provider.Initialize(
                    TetPtr, inv_DmPtr, det_DmPtr,
                    number, tet_number, useGravity, laplacianSmoothing,
                    dt, stiffness_0, stiffness_1, damp, mass, -3,
                    hyperelasticModelType);
        }

        // Set fixed delta time.
        Time.fixedDeltaTime = updatesPreFixedUpdate * dt * slowDownFactor;

        style = new GUIStyle()
        {
            normal = new GUIStyleState()
            {
                textColor = Color.white
            },
            font = Font.CreateDynamicFontFromOSFont("Source Code Pro", 15),
        };

        provider.SetLaplacianOmega(0.5f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    Matrix4x4 Build_Edge_Matrix(int tet)
    {
        Matrix4x4 ret = Matrix4x4.identity;
        Vector3 X0 = X[Tet[tet * 4 + 0]];
        Vector3 X1 = X[Tet[tet * 4 + 1]];
        Vector3 X2 = X[Tet[tet * 4 + 2]];
        Vector3 X3 = X[Tet[tet * 4 + 3]];

        ret.SetColumn(0, X0 - X1);
        ret.SetColumn(1, X0 - X2);
        ret.SetColumn(2, X0 - X3);

        return ret;
    }
    GUIStyle style;

    void OnGUI()
    {
        if (!debug) return;
        //caption = Marshal.PtrToStringUTF8(CUDA_device_name());
        unsafe
        {
            fixed (byte* debug_info = this.debug_info)
                provider.GetDebugInfo(debug_info);
        }
        if (debug_info[0] != 0)
            caption /*+*/= System.Text.Encoding.ASCII.GetString(debug_info).TrimEnd('\0');

        GUILayout.Label(caption, style);
    }

    public void SetDebugTet(string text)
    {
        provider.SetDebugTet(int.Parse(text));
    }

    void Update()
    {
        // Jump up.
        if (Input.GetKeyDown(KeyCode.Space)/* || Input.GetMouseButtonDown(0)*/)
        {
            provider.Impulse(new Vector3(0, 5f, 0));
        }

        // Dump the vertex array for rendering.
        Vector3[] vertices = new Vector3[tet_number * 12];
        int vertex_number = 0;
        for (int tet = 0; tet < tet_number; tet++)
        {
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
        }
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.vertices = vertices;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }

    void FixedUpdate()
    {
        if (debug)
            provider.SetDebugTet(0);
        unsafe
        {
            fixed (Vector3* xPtr = X)
                provider._Update(xPtr, updatesPreFixedUpdate);
        }
    }
}