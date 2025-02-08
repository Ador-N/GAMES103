using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class FVM : MonoBehaviour
{
	float dt = 0.003f;
	int updatesPreFixedUpdate = 5;
	float mass = 1;
	float stiffness_0 = 20000.0f;
	float stiffness_1 = 5000.0f;
	float damp = 0.999f;

	int[] Tet;
	int tet_number;         //The number of tetrahedra

	Vector3[] Force;
	Vector3[] V;
	Vector3[] X;
	int number;             //The number of vertices

	Matrix4x4[] inv_Dm;
	float[] det_Dm;

	//For Laplacian smoothing.
	Vector3[] V_sum;
	int[] V_num;

	SVD svd = new SVD();

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
			center = center / number;
			for (int i = 0; i < number; i++)
			{
				X[i] -= center;
				float temp = X[i].y;
				X[i].y = X[i].z;
				X[i].z = temp;
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
		V = new Vector3[number];
		Force = new Vector3[number];
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


		V = new Vector3[number];
		Force = new Vector3[number];
		V_sum = new Vector3[number];
		V_num = new int[number];

		//TODO: Need to allocate and assign inv_Dm
		inv_Dm = new Matrix4x4[tet_number];
		det_Dm = new float[tet_number];
		for (int i = 0; i < tet_number; i++)
		{
			var Dm = Build_Edge_Matrix(i);
			inv_Dm[i] = Dm.inverse;
			det_Dm[i] = Dm.determinant;
		}

		// Set fixed delta time.
		Time.fixedDeltaTime = updatesPreFixedUpdate * dt * 2.5f;
	}

	Matrix4x4 Build_Edge_Matrix(int tet)
	{
		Matrix4x4 ret = Matrix4x4.identity;
		//TODO: Need to build edge matrix here.
		Vector3 X0 = X[Tet[tet * 4 + 0]];
		Vector3 X1 = X[Tet[tet * 4 + 1]];
		Vector3 X2 = X[Tet[tet * 4 + 2]];
		Vector3 X3 = X[Tet[tet * 4 + 3]];

		ret.SetColumn(0, X0 - X1);
		ret.SetColumn(1, X0 - X2);
		ret.SetColumn(2, X0 - X3);

		return ret;
	}


	void _Update()
	{
		for (int i = 0; i < number; i++)
		{
			Force[i] = Physics.gravity * mass;
			//Force[i] = Vector3.zero;
		}

		for (int tet = 0; tet < tet_number; tet++)
		{
			//TODO: Deformation Gradient
			Matrix4x4 F = Build_Edge_Matrix(tet) * inv_Dm[tet];
			//TODO: Green Strain
			Matrix4x4 G = Utils.Mul(0.5f, Utils.Sub(F.transpose * F, Matrix4x4.identity));
			//TODO: Second PK Stress
			Matrix4x4 S = Utils.Add(Utils.Mul(2 * stiffness_1, G),
									Utils.Mul(stiffness_0 * G.Trace(), Matrix4x4.identity));
			//TODO: Elastic Force
			Matrix4x4 force = Utils.Mul(-det_Dm[tet] / 6, F * S * inv_Dm[tet].transpose);
			Force[Tet[tet * 4 + 0]] -=
			(Vector3)(force.GetColumn(0) + force.GetColumn(1) + force.GetColumn(2));
			//Force[Tet[tet * 4 + 0]] -= force.MultiplyVector(Vector3.one);
			Force[Tet[tet * 4 + 1]] += (Vector3)force.GetColumn(0);
			Force[Tet[tet * 4 + 2]] += (Vector3)force.GetColumn(1);
			Force[Tet[tet * 4 + 3]] += (Vector3)force.GetColumn(2);
		}

		for (int i = 0; i < number; i++)
		{
			//TODO: Update X and V here.
			V[i] += Force[i] / mass * dt;
			V[i] *= damp;
			X[i] += V[i] * dt;

			//TODO: (Particle) collision with floor.
			if (X[i].y < -3)
			{
				float oldY = X[i].y;
				X[i].y = -3;
				V[i].y += (-3 - oldY) / dt;
				//Force[i].y += (-3 - oldY) / dt / dt / mass;
			}
		}

		// Laplacian smoothing
		for (int i = 0; i < number; i++)
		{
			V_sum[i] = Vector3.zero;
			V_num[i] = 0;
		}

		for (int tet = 0; tet < tet_number; tet++)
		{
			Vector3 v = V[Tet[tet * 4 + 0]]
					  + V[Tet[tet * 4 + 1]]
					  + V[Tet[tet * 4 + 2]]
					  + V[Tet[tet * 4 + 3]];
			V_sum[Tet[tet * 4 + 0]] += v - V[Tet[tet * 4 + 0]];
			V_sum[Tet[tet * 4 + 1]] += v - V[Tet[tet * 4 + 1]];
			V_sum[Tet[tet * 4 + 2]] += v - V[Tet[tet * 4 + 2]];
			V_sum[Tet[tet * 4 + 3]] += v - V[Tet[tet * 4 + 3]];
			V_num[Tet[tet * 4 + 0]]++;
			V_num[Tet[tet * 4 + 1]]++;
			V_num[Tet[tet * 4 + 2]]++;
			V_num[Tet[tet * 4 + 3]]++;
		}

		for (int i = 0; i < number; i++)
			V[i] = 0.67f * V_sum[i] / (V_num[i] * 3) + 0.33f * V[i];

	}

	void Update()
	{
		// Jump up.
		if (Input.GetKeyDown(KeyCode.Space))
		{
			for (int i = 0; i < number; i++)
				V[i].y += 2.5f;
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
	}

	void FixedUpdate()
	{
		for (int l = 0; l < updatesPreFixedUpdate; l++)
			_Update();
	}
}
