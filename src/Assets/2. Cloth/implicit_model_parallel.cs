using System;
using System.Threading.Tasks;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public class implicit_model_parallel : MonoBehaviour
{
	public bool useChebyshev = true;
	public ComputeShader computer;
	public int iterationCount = 2048;
	/*//float mass = 1;*/
	float dt = 0.015f;
	float damping = 0.99f;
	const float rho = 0.995f;
	float spring_k = 8000;
	int[] E;
	float[] L;
	Vector3[] V;
	AdjacencyData[] vertices;
	Transform sphere;
	ComputeBuffer adjacencyDataBuffer, positionOldBuffer, positionHatBuffer;
	ComputeBuffer[] positionBuffer = new ComputeBuffer[2];
	// Avoid repeating using string to set buffer which causes GC.
	int omegaId, verticesPositionInId, verticesPositionOutId;

	// Start is called before the first frame update
	void Start()
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;

		//Resize the mesh.
		int n = 21;
		Vector3[] X = new Vector3[n * n];
		Vector2[] UV = new Vector2[n * n];
		int[] triangles = new int[(n - 1) * (n - 1) * 6];
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++)
			{
				X[j * n + i] = new Vector3(5 - 10.0f * i / (n - 1), 0, 5 - 10.0f * j / (n - 1));
				UV[j * n + i] = new Vector3(i / (n - 1.0f), j / (n - 1.0f));
			}
		int t = 0;
		for (int j = 0; j < n - 1; j++)
			for (int i = 0; i < n - 1; i++)
			{
				triangles[t * 6 + 0] = j * n + i;
				triangles[t * 6 + 1] = j * n + i + 1;
				triangles[t * 6 + 2] = (j + 1) * n + i + 1;
				triangles[t * 6 + 3] = j * n + i;
				triangles[t * 6 + 4] = (j + 1) * n + i + 1;
				triangles[t * 6 + 5] = (j + 1) * n + i;
				t++;
			}
		mesh.vertices = X;
		mesh.triangles = triangles;
		mesh.uv = UV;
		mesh.RecalculateNormals();


		//Construct the original E
		int[] _E = new int[triangles.Length * 2];
		for (int i = 0; i < triangles.Length; i += 3)
		{
			_E[i * 2 + 0] = triangles[i + 0];
			_E[i * 2 + 1] = triangles[i + 1];
			_E[i * 2 + 2] = triangles[i + 1];
			_E[i * 2 + 3] = triangles[i + 2];
			_E[i * 2 + 4] = triangles[i + 2];
			_E[i * 2 + 5] = triangles[i + 0];
		}
		//Reorder the original edge list
		for (int i = 0; i < _E.Length; i += 2)
			if (_E[i] > _E[i + 1])
				Swap(ref _E[i], ref _E[i + 1]);
		//Sort the original edge list using quicksort
		Quick_Sort(ref _E, 0, _E.Length / 2 - 1);

		int e_number = 0;
		for (int i = 0; i < _E.Length; i += 2)
			if (i == 0 || _E[i + 0] != _E[i - 2] || _E[i + 1] != _E[i - 1])
				e_number++;

		E = new int[e_number * 2];
		for (int i = 0, e = 0; i < _E.Length; i += 2)
			if (i == 0 || _E[i + 0] != _E[i - 2] || _E[i + 1] != _E[i - 1])
			{
				E[e * 2 + 0] = _E[i + 0];
				E[e * 2 + 1] = _E[i + 1];
				e++;
			}

		L = new float[E.Length / 2];
		for (int e = 0; e < E.Length / 2; e++)
		{
			int v0 = E[e * 2 + 0];
			int v1 = E[e * 2 + 1];
			L[e] = (X[v0] - X[v1]).magnitude;
		}

		V = new Vector3[X.Length];
		for (int i = 0; i < V.Length; i++)
			V[i] = new Vector3(0, 0, 0);

		// Get sphere transform
		sphere = GameObject.Find("Sphere").transform;

		// Set Fixed Delta Time
		Time.fixedDeltaTime = dt;

		// Construct VertexData array, which means convert E to adjacency list.
		vertices = new AdjacencyData[X.Length];
		int[] edgeCount = new int[X.Length];
		for (int i = 0; i < X.Length; i++)
		{
			vertices[i] = new AdjacencyData
			{
				Id = new(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
				Length = float4x4.zero
			};

		}
		for (int i = 0; i < L.Length; i++)
		{
			int u = E[i * 2];
			int v = E[i * 2 + 1];

			// Add adjacency information for vertex u
			if (edgeCount[u] < 16)
			{
				int j = edgeCount[u];
				vertices[u].Id[j / 4][j % 4] = v;
				vertices[u].Length[j / 4][j % 4] = L[i];
				edgeCount[u]++;
			}

			// Add adjacency information for vertex v
			if (edgeCount[v] < 16)
			{
				int j = edgeCount[v];
				vertices[v].Id[j / 4][j % 4] = u;
				vertices[v].Length[j / 4][j % 4] = L[i];
				edgeCount[v]++;
			}
		}
		positionBuffer[0] = new(X.Length, sizeof(float) * 3);
		positionBuffer[1] = new(X.Length, sizeof(float) * 3);
		positionOldBuffer = new(X.Length, sizeof(float) * 3);
		positionHatBuffer = new(X.Length, sizeof(float) * 3);
		adjacencyDataBuffer = new(X.Length, sizeof(uint) * 16 + sizeof(float) * 16/*,
								 ComputeBufferType.Constant, ComputeBufferMode.Immutable*/);
		adjacencyDataBuffer.SetData(vertices);

		int kernel = computer.FindKernel("CSMain");
		//computer.SetConstantBuffer("VerticesAdjacency", adjacencyDataBuffer, 0, n * n);
		computer.SetBuffer(kernel, "VerticesPositionInOld", positionOldBuffer);
		computer.SetBuffer(kernel, "VerticesPositionInHat", positionHatBuffer);
		computer.SetBuffer(kernel, "VerticesAdjacency", adjacencyDataBuffer);
		computer.SetFloat("dt", dt);
		computer.SetFloat("spring_k", spring_k);
		computer.SetFloat("inv_H", 1 / (1 / dt / dt + 4 * spring_k));

		omegaId = Shader.PropertyToID("omega");
		verticesPositionInId = Shader.PropertyToID("VerticesPositionIn");
		verticesPositionOutId = Shader.PropertyToID("VerticesPositionOut");

	}

	void Quick_Sort(ref int[] a, int l, int r)
	{
		int j;
		if (l < r)
		{
			j = Quick_Sort_Partition(ref a, l, r);
			Quick_Sort(ref a, l, j - 1);
			Quick_Sort(ref a, j + 1, r);
		}
	}

	int Quick_Sort_Partition(ref int[] a, int l, int r)
	{
		int pivot_0, pivot_1, i, j;
		pivot_0 = a[l * 2 + 0];
		pivot_1 = a[l * 2 + 1];
		i = l;
		j = r + 1;
		while (true)
		{
			do ++i; while (i <= r && (a[i * 2] < pivot_0 || a[i * 2] == pivot_0 && a[i * 2 + 1] <= pivot_1));
			do --j; while (a[j * 2] > pivot_0 || a[j * 2] == pivot_0 && a[j * 2 + 1] > pivot_1);
			if (i >= j) break;
			Swap(ref a[i * 2], ref a[j * 2]);
			Swap(ref a[i * 2 + 1], ref a[j * 2 + 1]);
		}
		Swap(ref a[l * 2 + 0], ref a[j * 2 + 0]);
		Swap(ref a[l * 2 + 1], ref a[j * 2 + 1]);
		return j;
	}

	void Swap(ref int a, ref int b)
	{
		int temp = a;
		a = b;
		b = temp;
	}

	void Collision_Handling()
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] X = mesh.vertices;

		// Handle collision.
		Vector3 c = sphere.position;
		float r = 2.7f;
		for (int i = 0; i < X.Length; i++)
		{
			// Remove two corners from being updated.
			if (i == 0 || i == 20)
				continue;

			// Detect Collision.
			Vector3 n = X[i] - c;
			float phi = n.magnitude - r;
			if (phi >= 0)
				continue; // Skip if no collision.

			// Impulse
			Vector3 Xi0 = X[i];
			X[i] -= n.normalized * phi;
			if (Vector3.Dot(V[i], n) < 0) V[i] += (X[i] - Xi0) / dt;
		}

		mesh.vertices = X;
	}

	struct AdjacencyData
	{
		public int4x4 Id;
		public float4x4 Length;
	}

	void FixedUpdate()
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] X = mesh.vertices;
		Vector3[] X_hat = new Vector3[X.Length];
		Vector3[] last_X = new Vector3[X.Length];
		uint[] realIterationCounts = new uint[X.Length];

		ComputeBuffer iterationCountBuffer = new ComputeBuffer(X.Length, sizeof(uint));
		iterationCountBuffer.SetData(realIterationCounts);
		computer.SetBuffer(computer.FindKernel("CSMain"), "RealIterationCounts", iterationCountBuffer);


		// Initial Setup.
		for (int i = 0; i < X.Length; i++)
		{
			// Remove two corners from being updated.
			if (i == 0 || i == 20)
				continue;
			V[i] *= damping;
			X_hat[i] = X[i] = X[i] + V[i] * dt;
		}

		// Call compute shader to do newton iteration.
		int kernel = computer.FindKernel("CSMain");
		float omega;
		// Set buffer and parameters.
		positionBuffer[0].SetData(X);
		positionHatBuffer.SetData(X_hat);

		GraphicsFence fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);

		for (int k = 0; k < iterationCount; k++)
		{
			int i = k % 2, o = (k + 1) % 2;

			if (!useChebyshev || k == 0) omega = 1;
			else if (k == 1) omega = 2 / (2 - rho * rho);
			else omega = 4 / (4 - rho * rho);

			computer.SetBuffer(kernel, verticesPositionInId, positionBuffer[i]);
			computer.SetBuffer(kernel, verticesPositionOutId, positionBuffer[o]);
			computer.SetFloat(omegaId, omega);

			// Dispatch compute shader，each thread handle 64 vertices.
			/*if (k > 0)
			{
				Graphics.WaitOnAsyncGraphicsFence(fence);
			}*/
			int threadGroups = Mathf.CeilToInt((float)X.Length / 64);
			computer.Dispatch(kernel, threadGroups, 1, 1);
			//GL.Flush();
			//fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);
			/*AsyncGPUReadback.Request(positionBuffer[i], (AsyncGPUReadbackRequest request) =>
				{
					if (request.hasError)
						Debug.Log("GPU readback error detected.");
				});*/

		}
		// Get data back.
		positionBuffer[iterationCount % 2].GetData(X);
		iterationCountBuffer.GetData(realIterationCounts);
		iterationCountBuffer.Release();


		// Finishing.
		for (int i = 0; i < X.Length; i++)
			V[i] += (X[i] - X_hat[i]) / dt;

		mesh.vertices = X;

		Collision_Handling();
		mesh.RecalculateNormals();
		mesh.RecalculateBounds();
	}

	void OnDestroy()
	{
		positionBuffer[0]?.Release();
		positionBuffer[1]?.Release();
		positionOldBuffer?.Release();
		adjacencyDataBuffer?.Release();
	}
}