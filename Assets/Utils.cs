using UnityEngine;

public static class Utils
{
    public static Matrix4x4 Add(Matrix4x4 l, Matrix4x4 r)
    {
        Matrix4x4 res = Matrix4x4.identity;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res[i, j] = l[i, j] + r[i, j];
        return res;
    }

    public static Matrix4x4 Sub(Matrix4x4 l, Matrix4x4 r)
    {
        Matrix4x4 res = Matrix4x4.identity;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res[i, j] = l[i, j] - r[i, j];
        return res;
    }

    public static Matrix4x4 Mul(float l, Matrix4x4 r)
    {
        Matrix4x4 res = Matrix4x4.identity;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                res[i, j] = l * r[i, j];
        return res;
    }

    public static float Trace(this Matrix4x4 mat)
    {
        return mat.m00 + mat.m11 + mat.m22;
    }
}

public class Singleton<T> where T : new()
{
    private static T instance;
    public static T Instance
    {
        get
        {
            instance ??= new();
            return instance;
        }
    }
}