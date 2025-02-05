using UnityEditor;
using UnityEngine;

public static class Utils
{
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