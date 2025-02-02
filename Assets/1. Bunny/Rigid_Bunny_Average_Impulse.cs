using UnityEngine;
using System.Collections;
using UnityEngine.Video;
using Unity.VisualScripting;
using System.Collections.Generic;

public class Rigid_Bunny_Average_Impulse : MonoBehaviour
{
	bool launched = false;
	float dt = 0.015f;
	Vector3 v = new(0, 0, 0);   // velocity
	Vector3 w = new(0, 0, 0);   // angular velocity

	Vector3[] vertices;

	float mass;                                 // mass
	Matrix4x4 I_ref, I_ref_inv;                            // reference inertia

	float lineardecay = 0.999f;                // for velocity decay
	float angulardecay = 0.98f;
	float restitution = 0.5f;                   // for collision
	float friction = 0.5f;
	float mu_n => restitution switch
	{
		_ when Mathf.Abs((v + Physics.gravity * dt).y) < 0.36f => 0.2f,
		_ => restitution
	};

	const float samplingRatio = 1f;
	const int samplingLimit = 800;


	// Use this for initialization
	void Start()
	{
		Time.fixedDeltaTime = dt;

		Mesh mesh = GetComponent<MeshFilter>().mesh;
		vertices = mesh.vertices;

		float m = 1;
		mass = 0;
		for (int i = 0; i < vertices.Length; i++)
		{
			mass += m;
			float diag = m * vertices[i].sqrMagnitude;
			I_ref[0, 0] += diag;
			I_ref[1, 1] += diag;
			I_ref[2, 2] += diag;
			I_ref[0, 0] -= m * vertices[i][0] * vertices[i][0];
			I_ref[0, 1] -= m * vertices[i][0] * vertices[i][1];
			I_ref[0, 2] -= m * vertices[i][0] * vertices[i][2];
			I_ref[1, 0] -= m * vertices[i][1] * vertices[i][0];
			I_ref[1, 1] -= m * vertices[i][1] * vertices[i][1];
			I_ref[1, 2] -= m * vertices[i][1] * vertices[i][2];
			I_ref[2, 0] -= m * vertices[i][2] * vertices[i][0];
			I_ref[2, 1] -= m * vertices[i][2] * vertices[i][1];
			I_ref[2, 2] -= m * vertices[i][2] * vertices[i][2];
		}
		I_ref[3, 3] = 1;
		I_ref_inv = I_ref.inverse;
	}

	Matrix4x4 Get_Cross_Matrix(Vector3 a)
	{
		//Get the cross product matrix of vector a
		Matrix4x4 A = Matrix4x4.zero;
		A[0, 0] = 0;
		A[0, 1] = -a[2];
		A[0, 2] = a[1];
		A[1, 0] = a[2];
		A[1, 1] = 0;
		A[1, 2] = -a[0];
		A[2, 0] = -a[1];
		A[2, 1] = a[0];
		A[2, 2] = 0;
		A[3, 3] = 1;
		return A;
	}

	// In this function, update v and w by the impulse due to the collision with
	//a plane <P, N>
	void Collision_Impulse(Vector3 P, Vector3 N)
	{
		transform.GetPositionAndRotation(out Vector3 x, out Quaternion q);
		Matrix4x4 R = Matrix4x4.Rotate(q), I_inv = R * I_ref_inv * R.transpose;
		Vector3 impulse = new(), impulseMoment = new(), Rr_avg = new();
		List<Vector3> verticesSelected = new();
		int collision_cnt = 0;

		foreach (var ri in vertices)
		{
			// Calculate vertex dynamic info
			Vector3 Rri = R.MultiplyVector(ri),
					xi = x + Rri,
					vi = v + Vector3.Cross(w, Rri);

			// Detect Collision
			float phi = Vector3.Dot(xi - P, N); // Signaled Distance Field
			if (phi >= 0 || Vector3.Dot(vi, N) >= 0) continue;              // Skip if no penetration
			++collision_cnt;
			Rr_avg += Rri;
			if (Random.Range(0, 1) > samplingRatio) continue;               // Skip if not sampled
			verticesSelected.Add(Rri);
			if (collision_cnt >= samplingLimit) break;                      // Break if too many collisions
		}

		// Do nothing if there's no collision
		if (verticesSelected.Count == 0) return;

		Rr_avg /= collision_cnt;
		verticesSelected.Add(Rr_avg);

		foreach (var Rri in verticesSelected)
		{
			// Calculate new vertex speed
			Vector3 vi = v + Vector3.Cross(w, Rri),
					vn = Vector3.Dot(vi, N) * N,
					vt = vi - vn;
			Vector3 vn_new = -mu_n * vn,
					vt_new = Vector3.zero;
			if (vt.sqrMagnitude >= 1e-6f)
				vt_new = vt * Mathf.Clamp01(1 - friction * (1 + mu_n) * vn.magnitude / vt.magnitude);
			Vector3 vi_new = vn_new + vt_new;

			Matrix4x4 Rri_astro = Get_Cross_Matrix(Rri),
					  K = Utils.Sub(Matrix4x4.Scale(Vector3.one / mass), Rri_astro * I_inv * Rri_astro);
			Vector3 j = K.inverse.MultiplyVector(vi_new - vi);
			impulse += j;
			impulseMoment += Vector3.Cross(Rri, j);
		}

		// Update v and w
		impulse /= verticesSelected.Count;
		impulseMoment /= verticesSelected.Count;
		v += impulse / mass;
		w += I_inv.MultiplyVector(impulseMoment);
	}

	void FixedUpdate()
	{
		// Game Control
		if (Input.GetKey(KeyCode.R))
		{
			transform.position = new Vector3(0, 0.6f, 0);
			restitution = 0.5f;
			launched = false;
		}
		if (Input.GetKey(KeyCode.L))
		{
			v = new Vector3(5, 2, 0);
			launched = true;
		}

		if (!launched) return;

		// Part I: Update velocities
		v += Physics.gravity * dt;
		v *= lineardecay;

		// No need to calculate torque for gravity.
		// w += Vector3.zero;
		w *= angulardecay;


		// Part II: Collision Impulse
		Collision_Impulse(new Vector3(0, 0.01f, 0), new Vector3(0, 1, 0));
		Collision_Impulse(new Vector3(2, 0, 0), new Vector3(-1, 0, 0));

		// Part III: Update position & orientation
		transform.GetPositionAndRotation(out Vector3 x, out Quaternion q);
		// Update linear status
		x += v * dt;
		// Update angular status
		Vector3 wDtDiv2 = w * dt / 2;
		q = new Quaternion(wDtDiv2.x, wDtDiv2.y, wDtDiv2.z, 1) * q;
		q.Normalize();

		// Part IV: Assign to the object
		transform.SetPositionAndRotation(x, q);
	}
}
