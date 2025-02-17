using UnityEngine;

public class Rigid_Bunny_by_Shape_Matching : MonoBehaviour
{
	public bool launched = false;
	Vector3[] X;
	Vector3[] Q;
	Vector3[] V;
	Matrix4x4 QQt = Matrix4x4.zero, QQt_inv = Matrix4x4.zero;
	//public Vector3 immediateImpulse;
	public Vector3 immediateSetVelocity;


	// Start is called before the first frame update
	void Start()
	{
		// Set Fixed Delta Time
		Time.fixedDeltaTime = 0.025f;

		Mesh mesh = GetComponent<MeshFilter>().mesh;
		V = new Vector3[mesh.vertices.Length];
		X = mesh.vertices;
		Q = mesh.vertices;

		//Centerizing Q.
		Vector3 c = Vector3.zero;
		for (int i = 0; i < Q.Length; i++)
			c += Q[i];
		c /= Q.Length;
		for (int i = 0; i < Q.Length; i++)
			Q[i] -= c;

		//Get QQ^t ready.
		for (int i = 0; i < Q.Length; i++)
		{
			QQt[0, 0] += Q[i][0] * Q[i][0];
			QQt[0, 1] += Q[i][0] * Q[i][1];
			QQt[0, 2] += Q[i][0] * Q[i][2];
			QQt[1, 0] += Q[i][1] * Q[i][0];
			QQt[1, 1] += Q[i][1] * Q[i][1];
			QQt[1, 2] += Q[i][1] * Q[i][2];
			QQt[2, 0] += Q[i][2] * Q[i][0];
			QQt[2, 1] += Q[i][2] * Q[i][1];
			QQt[2, 2] += Q[i][2] * Q[i][2];
		}
		QQt[3, 3] = 1;
		QQt_inv = QQt.inverse;

		for (int i = 0; i < X.Length; i++)
			V[i][0] = 4.0f;

		Update_Mesh(transform.position, Matrix4x4.Rotate(transform.rotation), 0);
		transform.position = Vector3.zero;
		transform.rotation = Quaternion.identity;
	}

	// Polar Decomposition that returns the rotation from F.
	Matrix4x4 Get_Rotation(Matrix4x4 F)
	{
		Matrix4x4 C = Matrix4x4.zero;
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				for (int kk = 0; kk < 3; kk++)
					C[ii, jj] += F[kk, ii] * F[kk, jj];

		Matrix4x4 C2 = Matrix4x4.zero;
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				for (int kk = 0; kk < 3; kk++)
					C2[ii, jj] += C[ii, kk] * C[jj, kk];

		float det = F[0, 0] * F[1, 1] * F[2, 2] +
						F[0, 1] * F[1, 2] * F[2, 0] +
						F[1, 0] * F[2, 1] * F[0, 2] -
						F[0, 2] * F[1, 1] * F[2, 0] -
						F[0, 1] * F[1, 0] * F[2, 2] -
						F[0, 0] * F[1, 2] * F[2, 1];

		float I_c = C[0, 0] + C[1, 1] + C[2, 2];
		float I_c2 = I_c * I_c;
		float II_c = 0.5f * (I_c2 - C2[0, 0] - C2[1, 1] - C2[2, 2]);
		float III_c = det * det;
		float k = I_c2 - 3 * II_c;

		Matrix4x4 inv_U = Matrix4x4.zero;
		if (k < 1e-10f)
		{
			float inv_lambda = 1 / Mathf.Sqrt(I_c / 3);
			inv_U[0, 0] = inv_lambda;
			inv_U[1, 1] = inv_lambda;
			inv_U[2, 2] = inv_lambda;
		}
		else
		{
			float l = I_c * (I_c * I_c - 4.5f * II_c) + 13.5f * III_c;
			float k_root = Mathf.Sqrt(k);
			float value = l / (k * k_root);
			if (value < -1.0f) value = -1.0f;
			if (value > 1.0f) value = 1.0f;
			float phi = Mathf.Acos(value);
			float lambda2 = (I_c + 2 * k_root * Mathf.Cos(phi / 3)) / 3.0f;
			float lambda = Mathf.Sqrt(lambda2);

			float III_u = Mathf.Sqrt(III_c);
			if (det < 0) III_u = -III_u;
			float I_u = lambda + Mathf.Sqrt(-lambda2 + I_c + 2 * III_u / lambda);
			float II_u = (I_u * I_u - I_c) * 0.5f;


			float inv_rate, factor;
			inv_rate = 1 / (I_u * II_u - III_u);
			factor = I_u * III_u * inv_rate;

			Matrix4x4 U = Matrix4x4.zero;
			U[0, 0] = factor;
			U[1, 1] = factor;
			U[2, 2] = factor;

			factor = (I_u * I_u - II_u) * inv_rate;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					U[i, j] += factor * C[i, j] - inv_rate * C2[i, j];

			inv_rate = 1 / III_u;
			factor = II_u * inv_rate;
			inv_U[0, 0] = factor;
			inv_U[1, 1] = factor;
			inv_U[2, 2] = factor;

			factor = -I_u * inv_rate;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					inv_U[i, j] += factor * U[i, j] + inv_rate * C[i, j];
		}

		Matrix4x4 R = Matrix4x4.zero;
		for (int ii = 0; ii < 3; ii++)
			for (int jj = 0; jj < 3; jj++)
				for (int kk = 0; kk < 3; kk++)
					R[ii, jj] += F[ii, kk] * inv_U[kk, jj];
		R[3, 3] = 1;
		return R;
	}

	// Update the mesh vertices according to translation c and rotation R.
	// It also updates the velocity.
	void Update_Mesh(Vector3 c, Matrix4x4 R, float inv_dt)
	{
		for (int i = 0; i < Q.Length; i++)
		{
			Vector3 x = (Vector3)(R * Q[i]) + c;

			V[i] += (x - X[i]) * inv_dt;
			X[i] = x;
		}
		// This will disturb all the mesh normals and make rendering issue.
		// And recalculating normals will cost a lot.
		// So let's just use simple transform.
		//Mesh mesh = GetComponent<MeshFilter>().mesh;
		//mesh.vertices = X;
		transform.SetPositionAndRotation(c, R.rotation);
	}

	void Collision(float inv_dt)
	{
		Collision_Impulse(new Vector3(0, 0.01f, 0), new Vector3(0, 1, 0));
		Collision_Impulse(new Vector3(2, 0, 0), new Vector3(-1, 0, 0));
	}

	// In this function, update v and w by the impulse due to the collision with
	//a plane <P, N>
	void Collision_Impulse(Vector3 P, Vector3 N)
	{
		float mu_n = 0.8f, mu_t = 0.5f;

		transform.GetPositionAndRotation(out Vector3 x, out Quaternion q);

		for (int i = 0; i < Q.Length; i++)
		{
			Vector3 xi = X[i], vi = V[i];
			// Detect Collision
			float phi = Vector3.Dot(xi - P, N); // Signaled Distance Field
			if (phi >= 0 || Vector3.Dot(vi, N) >= 0)
				continue; // Skip if no penetration

			// Calculate new vertex speed
			Vector3 vn = Vector3.Dot(vi, N) * N,
					vt = vi - vn;
			Vector3 vn_new = -mu_n * vn,
					vt_new = vt * Mathf.Clamp01(1 - mu_t * (1 + mu_n) * vn.magnitude);

			// Update X and V
			X[i] -= phi * N;
			V[i] = vn_new + vt_new;
		}
	}

	// Update is called once per frame
	void FixedUpdate()
	{
		if (Input.GetKey(KeyCode.L))
		{
			if (!launched)
				launched = true;
			else immediateSetVelocity = new Vector3(5, 2, 0);
		}
		else if (Input.GetKey(KeyCode.R))
		{
			for (int i = 0; i < V.Length; i++)
			{
				X[i] = new Vector3(0, 0.6f, 0) + Q[i];
				V[i] = new Vector3(4, 0, 0);
				transform.position = new Vector3(0, 0.6f, 0);
			}
			launched = false;
		}
		if (!launched) return;
		float dt = 0.025f;

		float time0 = Time.realtimeSinceStartup * 1000;
		Vector3 gravity = Physics.gravity;
		//Step 1: run a simple particle system.
		for (int i = 0; i < V.Length; i++)
		{
			if (immediateSetVelocity != Vector3.zero)
				V[i] = immediateSetVelocity;
			V[i] += gravity * dt;
			X[i] += V[i] * dt;
		}
		immediateSetVelocity = Vector3.zero;
		float time1 = Time.realtimeSinceStartup * 1000;

		//Step 2: Perform simple particle collision.
		Collision(1 / dt);
		float time2 = Time.realtimeSinceStartup * 1000;

		// Step 3: Use shape matching to get new translation c and 
		// new rotation R. Update the mesh by c and R.
		//Shape Matching (translation)
		Vector3 c = Vector3.zero;
		for (int i = 0; i < Q.Length; i++)
			c += X[i];
		c /= X.Length;

		float time3 = Time.realtimeSinceStartup * 1000;

		//Shape Matching (rotation)
		Matrix4x4 A = Matrix4x4.zero, R;
		for (int i = 0; i < Q.Length; i++)
		{
			Vector3 zi = X[i] - c, ri = Q[i];
			A.m00 += zi.x * ri.x;
			A.m01 += zi.x * ri.y;
			A.m02 += zi.x * ri.z;
			A.m10 += zi.y * ri.x;
			A.m11 += zi.y * ri.y;
			A.m12 += zi.y * ri.z;
			A.m20 += zi.z * ri.x;
			A.m21 += zi.z * ri.y;
			A.m22 += zi.z * ri.z;
		}
		A.m33 = 1;
		float time4 = Time.realtimeSinceStartup * 1000;
		A *= QQt_inv;
		R = Get_Rotation(A);
		float time5 = Time.realtimeSinceStartup * 1000;

		Update_Mesh(c, R, 1 / dt);
		float time6 = Time.realtimeSinceStartup * 1000;
		//Debug.Log($"Time: {time1 - time0}, {time2 - time1}, {time3 - time2}, {time4 - time3}, {time5 - time4}, {time6 - time5}");
	}
}
