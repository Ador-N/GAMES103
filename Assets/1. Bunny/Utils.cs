using UnityEditor;
using UnityEngine;

public static class Utils
{
    public static Quaternion Add(Quaternion l, Quaternion r)
    {
        return new Quaternion(l.x + r.x, l.y + r.y, l.z + r.z, l.w + r.w);
    }
    public static Vector3 Multiply(Matrix4x4 l, Vector3 r)
    {
        Vector4 r4 = new Vector4(r.x, r.y, r.z, 1);
        r4 = l * r4;
        return new Vector3(r4.x, r4.y, r4.z);
    }

    public static Matrix4x4 Add(Matrix4x4 l, Matrix4x4 r)
    {
        Matrix4x4 res = Matrix4x4.zero;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res[i, j] = l[i, j] + r[i, j];
        res[3, 3] = 1;
        return res;
    }

    public static Matrix4x4 Sub(Matrix4x4 l, Matrix4x4 r)
    {
        Matrix4x4 res = Matrix4x4.zero;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res[i, j] = l[i, j] - r[i, j];
        res[3, 3] = 1;
        return res;
    }
}