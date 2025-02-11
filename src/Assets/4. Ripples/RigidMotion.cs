using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class RigidMotion : MonoBehaviour
{
	bool pressed = false;
	public bool move = false;
	bool lifted = false;
	Vector3 offset;

	float mass = 1.0f;
	float dt = 0.02f;
	public bool useGravity = true;
	public float linear_decay = 0.999f;                // for velocity decay
	public float angular_decay = 0.98f;
	/*public float restitution = 0.5f;                   // for collision
	public float friction = 0.5f;*/

	public Vector3 principleInertia = Vector3.one;

	Vector3 v;
	Vector3 w;

	Matrix4x4 I_ref;
	Matrix4x4 I_ref_inv;

	// Start is called before the first frame update
	void Start()
	{
		I_ref = Matrix4x4.Scale(principleInertia);
		I_ref_inv = I_ref.inverse;
	}

	void Update()
	{
		if (Input.GetMouseButtonDown(0))
		{
			pressed = true;
			Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
			if (Vector3.Cross(ray.direction, transform.position - ray.origin).magnitude < 0.8f) move = true;
			else move = false;
			offset = Input.mousePosition - Camera.main.WorldToScreenPoint(transform.position);
		}
		if (Input.GetMouseButtonUp(0))
		{
			pressed = false;
			move = false;
		}
		lifted = Input.GetMouseButton(1);
		_FixedUpdate();
	}

	void _FixedUpdate()
	{
		if (pressed)
		{
			if (move)
			{
				Vector3 mouse = Input.mousePosition;
				mouse -= offset;
				mouse.z = Camera.main.WorldToScreenPoint(transform.position).z;
				Vector3 p = Camera.main.ScreenToWorldPoint(mouse);
				p.y = lifted ? 3 : transform.position.y;
				transform.position = p;
			}
		}
		else
		{
			// Only update position when not moving.
			if (useGravity)
				v += Physics.gravity * dt;
			v *= linear_decay;

			// Decay when velocity is small enough
			if (v.sqrMagnitude < 1e-4f)
				v *= 0.5f;

			transform.position += dt * v;
		}
		// Update angular status whenever.
		Quaternion q = transform.rotation;
		w *= angular_decay;
		Vector3 wDtDiv2 = dt / 2 * w;
		q = new Quaternion(wDtDiv2.x, wDtDiv2.y, wDtDiv2.z, 1) * q;
		q.Normalize();
		transform.rotation = q;
	}

	public void Impulse(Vector3 position, Vector3 impulse)
	{
		Matrix4x4 R = Matrix4x4.Rotate(transform.rotation),
				I_inv = R * I_ref_inv * R.transpose;
		Vector3 r = position - transform.position,
				impulseMoment = Vector3.Cross(r, impulse);

		// Ignore positional impulse when pressed
		if (!pressed) v += impulse / mass;
		w += I_inv.MultiplyVector(impulseMoment);
	}
}
