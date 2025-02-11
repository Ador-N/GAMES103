using UnityEngine;
using System.Collections;

public class Rigid_Bunny : MonoBehaviour
{
	public bool launched = false;
	public bool useGravity = false;
	float dt = 0.015f;
	public Vector3 v = new(0, 0, 0);   // velocity
	public Vector3 w = new(0, 0, 0);   // angular velocity

	Vector3[] vertices;

	float mass;                                 // mass
	Matrix4x4 I_ref, I_ref_inv;                 // reference inertia

	public float linear_decay = 0.999f;                // for velocity decay
	public float angular_decay = 0.98f;
	public float restitution = 0.5f;                   // for collision
	public float friction = 0.5f;
	//float punishment = 1;
	public int collisionCount;
	Vector3 avgCollisionPoint;
	Vector3 impulse;
	Vector3 pointVelocity;

	float mu_t => friction;
	float mu_n => restitution switch
	{
		_ when Mathf.Abs((v + Physics.gravity * dt).y) < 0.36f => 0,
		_ => restitution
	};

	public float speed;
	public float energy;


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
	void Collision_Impulse(Vector3 P, Vector3 N, bool isGround = false)
	{
		transform.GetPositionAndRotation(out Vector3 x, out Quaternion q);
		Matrix4x4 R = Matrix4x4.Rotate(q),
				  I_inv = R * I_ref.inverse * R.transpose;
		int collision_cnt = 0;
		Vector3 Rr_avg = new();

		foreach (var r_i in vertices)
		{
			// Calculate vertex dynamic info
			Vector3 Rr_i = R.MultiplyVector(r_i),
					x_i = x + Rr_i,
					v_i = v + Vector3.Cross(w, Rr_i);

			// Detect Collision
			float phi = Vector3.Dot(x_i - P, N); // Signaled Distance Field
			if (phi >= 0 || Vector3.Dot(v_i, N) >= 0)
				continue; // Skip if no penetration
			++collision_cnt;
			Rr_avg += Rr_i;
		}

		if (collision_cnt > 0)
		{
			Rr_avg /= collision_cnt;

			// Calculate new vertex speed
			Vector3 v_i = v + Vector3.Cross(w, Rr_avg);
			pointVelocity = v_i;
			if (isGround)
			{
				collisionCount = collision_cnt;
				avgCollisionPoint = x + Rr_avg;
			}
			Vector3 v_n = Vector3.Dot(v_i, N) * N,
					v_t = v_i - v_n;

			Vector3 v_n_new = -mu_n * v_n,
					v_t_new = Vector3.zero;
			if (v_t.sqrMagnitude >= 1e-6f)
				v_t_new = v_t * Mathf.Clamp01(1 - mu_t * (1 + mu_n) * v_n.magnitude / v_t.magnitude);
			Vector3 v_i_new = v_n_new + v_t_new;

			Matrix4x4 Rr_i_astro = Get_Cross_Matrix(Rr_avg),
					  K = Utils.Sub(Matrix4x4.Scale(Vector3.one / mass), Rr_i_astro * I_inv * Rr_i_astro);
			Vector3 impulse = K.inverse * (v_i_new - v_i),
					impulseMoment = Vector3.Cross(Rr_avg, impulse);
			this.impulse = impulse;

			// Update v and w
			v += impulse / mass;
			w += I_inv.MultiplyVector(impulseMoment);
		}
	}

	void OnDrawGizmos()
	{

		if (collisionCount > 0)
		{
			Gizmos.color = Color.red;
			Gizmos.DrawSphere(avgCollisionPoint, 0.01f);
			Gizmos.DrawLine(avgCollisionPoint, avgCollisionPoint + impulse / mass);
			Gizmos.color = Color.green;
			Gizmos.DrawLine(avgCollisionPoint, avgCollisionPoint + pointVelocity);
		}
	}

	void FixedUpdate()
	{
		// Game Control
		if (Input.GetKey(KeyCode.R))
		{
			transform.position = new Vector3(0, 0.6f, 0);
			launched = false;
		}
		if (Input.GetKey(KeyCode.L))
		{
			v = new Vector3(5, 2, 0);
			launched = true;
		}

		if (!launched) return;

		// Part I: Update velocities
		if (useGravity)
			v += Physics.gravity * dt;
		v *= linear_decay;

		// No need to calculate torque for gravity.
		// w += Vector3.zero;
		w *= angular_decay;

		// Part II: Collision Impulse
		Collision_Impulse(new Vector3(0, 0.01f, 0), new Vector3(0, 1, 0), true);
		Collision_Impulse(new Vector3(2, 0, 0), new Vector3(-1, 0, 0));

		// Decay when velocity is small enough
		if (v.sqrMagnitude < 1e-6f)
			v *= 0.5f;

		// Part III: Update position & orientation
		transform.GetPositionAndRotation(out Vector3 x, out Quaternion q);
		// Update linear status
		x += dt * v;
		// Update angular status
		Vector3 wDtDiv2 = dt / 2 * w;
		q = new Quaternion(wDtDiv2.x, wDtDiv2.y, wDtDiv2.z, 1) * q;
		q.Normalize();

		// Part IV: Assign to the object
		transform.SetPositionAndRotation(x, q);

		speed = v.magnitude;
		var wi = transform.InverseTransformDirection(w);
		energy = 0.5f * mass * v.sqrMagnitude
			+ 0.5f * Vector3.Dot(wi, I_ref.MultiplyVector(wi))
			- mass * Physics.gravity.y * x.y;
	}
}
