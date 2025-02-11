using UnityEngine;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Data;
using System.Collections.Generic;
using UnityEngine.Rendering.Universal;
using UnityEngine.Profiling;

public class WaveMotion : MonoBehaviour
{
	int size = 100;
	float rate = 0.005f;
	float gamma = 0.004f;
	float damping = 0.996f;
	float dx = 0.1f;
	float dA = 0.01f;
	float dt = 0.02f;
	float[,] old_h;
	float[,] h;
	float[,] low_h;
	float[,] vh;
	float[,] b;

	float[,] cg_p;
	float[,] cg_r;
	float[,] cg_Ap;

	struct Mask
	{
		public bool[,] mask;
		public int li, ui, lj, uj;

		public Mask(bool[,] mask, int li, int ui, int lj, int uj)
		{
			this.mask = mask;
			this.li = li;
			this.ui = ui;
			this.lj = lj;
			this.uj = uj;
		}

		public bool this[int i, int j]
		{
			get => mask[i, j];
			set => mask[i, j] = value;
		}
	}
	List<RigidMotion> rigidbodies;
	Dictionary<RigidMotion, Mask> rigidbody_mask = new();
	//bool tag = true;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	Vector3 GetPosition(int i, int j)
	{
		return new Vector3(
			i * 0.1f - size * 0.05f,
			0,
			j * 0.1f - size * 0.05f);
	}

	// Use this for initialization
	void Start()
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		mesh.Clear();

		rigidbodies = new(FindObjectsOfType<RigidMotion>());

		Vector3[] X = new Vector3[size * size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				X[i * size + j] = GetPosition(i, j);
			}

		int[] T = new int[(size - 1) * (size - 1) * 6];
		int index = 0;
		for (int i = 0; i < size - 1; i++)
			for (int j = 0; j < size - 1; j++)
			{
				T[index * 6 + 0] = (i + 0) * size + (j + 0);
				T[index * 6 + 1] = (i + 0) * size + (j + 1);
				T[index * 6 + 2] = (i + 1) * size + (j + 1);
				T[index * 6 + 3] = (i + 0) * size + (j + 0);
				T[index * 6 + 4] = (i + 1) * size + (j + 1);
				T[index * 6 + 5] = (i + 1) * size + (j + 0);
				index++;
			}
		mesh.vertices = X;
		mesh.triangles = T;
		mesh.RecalculateNormals();

		low_h = new float[size, size];
		old_h = new float[size, size];
		h = new float[size, size];
		vh = new float[size, size];
		b = new float[size, size];

		cg_p = new float[size, size];
		cg_r = new float[size, size];
		cg_Ap = new float[size, size];

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				low_h[i, j] = 99999;
				old_h[i, j] = 0;
				vh[i, j] = 0;
			}

		Time.fixedDeltaTime = dt;
	}

	void A_Times(bool[,] mask, float[,] x, float[,] Ax, int li, int ui, int lj, int uj)
	{
		for (int i = li; i <= ui; i++)
			for (int j = lj; j <= uj; j++)
				if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
				{
					Ax[i, j] = 0;
					if (i != 0) Ax[i, j] -= x[i - 1, j] - x[i, j];
					if (i != size - 1) Ax[i, j] -= x[i + 1, j] - x[i, j];
					if (j != 0) Ax[i, j] -= x[i, j - 1] - x[i, j];
					if (j != size - 1) Ax[i, j] -= x[i, j + 1] - x[i, j];
				}
	}

	float Dot(bool[,] mask, float[,] x, float[,] y, int li, int ui, int lj, int uj)
	{
		float ret = 0;
		for (int i = li; i <= ui; i++)
			for (int j = lj; j <= uj; j++)
				if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
				{
					ret += x[i, j] * y[i, j];
				}
		return ret;
	}

	void Conjugate_Gradient(bool[,] mask, float[,] b, float[,] x, int li, int ui, int lj, int uj)
	{
		//Solve the Laplacian problem by CG.
		A_Times(mask, x, cg_r, li, ui, lj, uj);

		for (int i = li; i <= ui; i++)
			for (int j = lj; j <= uj; j++)
				if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
				{
					cg_p[i, j] = cg_r[i, j] = b[i, j] - cg_r[i, j];
				}

		float rk_norm = Dot(mask, cg_r, cg_r, li, ui, lj, uj);

		for (int k = 0; k < 128; k++)
		{
			if (rk_norm < 1e-10f) break;
			A_Times(mask, cg_p, cg_Ap, li, ui, lj, uj);
			float alpha = rk_norm / Dot(mask, cg_p, cg_Ap, li, ui, lj, uj);

			for (int i = li; i <= ui; i++)
				for (int j = lj; j <= uj; j++)
					if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
					{
						x[i, j] += alpha * cg_p[i, j];
						cg_r[i, j] -= alpha * cg_Ap[i, j];
					}

			float _rk_norm = Dot(mask, cg_r, cg_r, li, ui, lj, uj);
			float beta = _rk_norm / rk_norm;
			rk_norm = _rk_norm;

			for (int i = li; i <= ui; i++)
				for (int j = lj; j <= uj; j++)
					if (i >= 0 && j >= 0 && i < size && j < size && mask[i, j])
					{
						cg_p[i, j] = cg_r[i, j] + beta * cg_p[i, j];
					}
		}

	}

	void CalcVhForBlock(RigidMotion body, float[,] h)
	{
		//TODO: for block, calculate low_h.
		//TODO: then set up b and cg_mask for conjugate gradient.
		Vector3 size3 = new(size, 0, size);
		Vector3Int size3i = new(size - 1, 0, size - 1);
		Collider collider = body.GetComponent<Collider>();
		var (li, _, lj) = Vector3Int.Max(Utils.FloorToInt(collider.bounds.min * 10 + 0.5f * size3), Vector3Int.zero);
		var (ui, _, uj) = Vector3Int.Min(Utils.CeilToInt(collider.bounds.max * 10 + 0.5f * size3), size3i);

		if (!rigidbody_mask.TryGetValue(body, out var mask))
			mask = rigidbody_mask[body] = new Mask(new bool[size, size], li, ui, lj, uj);

		for (int i = mask.li; i <= mask.ui; i++)
			for (int j = mask.lj; j <= mask.uj; j++)
				mask[i, j] = false;

		mask = rigidbody_mask[body] = new Mask(mask.mask, li, ui, lj, uj);

		for (int i = li; i <= ui; i++)
			for (int j = lj; j <= uj; j++)
				if (collider.Raycast(
						new Ray(GetPosition(i, j) + Vector3.down * 10, Vector3.up),
						out var hitInfo,
						10 + h[i, j]))
				{
					low_h[i, j] = hitInfo.point.y;
					b[i, j] = (h[i, j] - low_h[i, j]) / rate;
					mask[i, j] = true;
				}

		//TODO: Solve the Poisson equation to obtain vh (virtual height).
		Conjugate_Gradient(mask.mask, b, vh, li, ui, lj, uj);

	}

	void Shallow_Wave(float[,] old_h, float[,] h, float[,] new_h)
	{
		//Step 1:
		//TODO: Compute new_h based on the shallow wave model.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				new_h[i, j] = h[i, j] + damping * (h[i, j] - old_h[i, j]);
				if (i != 0) new_h[i, j] += rate * (h[i - 1, j] - h[i, j]);
				if (j != 0) new_h[i, j] += rate * (h[i, j - 1] - h[i, j]);
				if (i != size - 1) new_h[i, j] += rate * (h[i + 1, j] - h[i, j]);
				if (j != size - 1) new_h[i, j] += rate * (h[i, j + 1] - h[i, j]);
			}

		//Step 2: Block->Water coupling

		Profiler.BeginSample("Block->Water coupling");
		// Calculate low_h for rigidbodies
		foreach (var body in rigidbodies)
		{
			CalcVhForBlock(body, h);
		}
		Profiler.EndSample();

		//TODO: Diminish vh.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				vh[i, j] *= gamma;
			}

		//TODO: Update new_h by vh.
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				if (i != 0) new_h[i, j] += rate * (vh[i - 1, j] - vh[i, j]);
				if (j != 0) new_h[i, j] += rate * (vh[i, j - 1] - vh[i, j]);
				if (i != size - 1) new_h[i, j] += rate * (vh[i + 1, j] - vh[i, j]);
				if (j != size - 1) new_h[i, j] += rate * (vh[i, j + 1] - vh[i, j]);
			}

		//Step 3
		//TODO: old_h <- h; h <- new_h;
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				(old_h[i, j], h[i, j]) = (h[i, j], new_h[i, j]);

		//Step 4: Water->Block coupling.
		//More TODO here.
		foreach (var (cube, mask) in rigidbody_mask)
		{
			for (int i = mask.li; i <= mask.ui; i++)
				for (int j = mask.lj; j <= mask.uj; j++)
				{
					if (mask[i, j])
					{
						cube.Impulse(
							GetPosition(i, j) + Vector3.up * h[i, j],
							0.05f * dA * vh[i, j] * Vector3.up
						);
					}
				}
		}

	}

	void Update()
	{
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] X = mesh.vertices;

		if (Input.GetKey("r"))
		{
			//TODO: Add random water.
			int i = Random.Range(1, size - 1), j = Random.Range(1, size - 1);
			float r = Random.Range(0.075f, 0.15f);
			h[i, j] += 4 * r;
			h[i - 1, j] -= r;
			h[i + 1, j] -= r;
			h[i, j - 1] -= r;
			h[i, j + 1] -= r;
		}

		_FixedUpdate();

		//TODO: Store h back into X.y and recalculate normal.

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				X[i * size + j].y = h[i, j];
			}

		mesh.vertices = X;
		mesh.RecalculateNormals();

	}

	void _FixedUpdate()
	{
		float[,] new_h = new float[size, size];

		for (int l = 0; l < 8; l++)
		{
			Shallow_Wave(old_h, h, new_h);
		}

	}
}
