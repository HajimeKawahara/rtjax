import numpy as np

def qmatrix(a, b, c, d, e, f, g, h, w, x, y, z):
    """Q matrix

    Notes:
        Given A = ((a,b),(c,d)), B=((e,f),(g,h)), X = ((A, BJ), (AJ, B)), and L = diag(w,x,y,z), then provides  Q = X L X^-1

    Args:
        a to z (float): See the above definition. 

    Returns:
        4x4 array: Q matrix
    """
    fac1 = (a * g**2 - a * h**2 - c * e * g + c * f * h + d * e * h -
            d * f * g)
    fac2 = (b * g**2 - b * h**2 + c * e * h - c * f * g - d * e * g +
            d * f * h)
    fac3 = (a * c * g - a * d * h + b * c * h - b * d * g - c**2 * e +
            d**2 * e)
    fac4 = (a * c * h - a * d * g + b * c * g - b * d * h - c**2 * f +
            d**2 * f)

    fac5 = (a * e * g - a * f * h + b * e * h - b * f * g - c * e**2 +
            c * f**2)
    fac6 = (a * e * h - a * f * g + b * e * g - b * f * h - d * e**2 +
            d * f**2)
    fac7 = (a**2 * g - a * c * e - a * d * f - b**2 * g + b * c * f +
            b * d * e)
    fac8 = (a**2 * h - a * c * f - a * d * e - b**2 * h + b * c * e +
            b * d * f)

    Q = np.array(
        [[(a * w * fac1 - b * x * fac2 - e * z * fac3 + f * y * fac4),
          (-a * w * fac5 + b * x * fac6 + e * z * fac7 - f * y * fac8),
          (-a * w * fac2 + b * x * fac1 + e * z * fac4 - f * y * fac3),
          (a * w * fac6 - b * x * fac5 - e * z * fac8 + f * y * fac7)],
         [(c * w * fac1 - d * x * fac2 - g * z * fac3 + h * y * fac4),
          (-c * w * fac5 + d * x * fac6 + g * z * fac7 - h * y * fac8),
          (-c * w * fac2 + d * x * fac1 + g * z * fac4 - h * y * fac3),
          (c * w * fac6 - d * x * fac5 - g * z * fac8 + h * y * fac7)],
         [(-a * x * fac2 + b * w * fac1 + e * y * fac4 - f * z * fac3),
          (a * x * fac6 - b * w * fac5 - e * y * fac8 + f * z * fac7),
          (a * x * fac1 - b * w * fac2 - e * y * fac3 + f * z * fac4),
          (-a * x * fac5 + b * w * fac6 + e * y * fac7 - f * z * fac8)],
         [(-c * x * fac2 + d * w * fac1 + g * y * fac4 - h * z * fac3),
          (c * x * fac6 - d * w * fac5 - g * y * fac8 + h * z * fac7),
          (c * x * fac1 - d * w * fac2 - g * y * fac3 + h * z * fac4),
          (-c * x * fac5 + d * w * fac6 + g * y * fac7 - h * z * fac8)]])

    denom = (a**2 * g**2 - a**2 * h**2 - 2 * a * c * e * g +
             2 * a * c * f * h + 2 * a * d * e * h - 2 * a * d * f * g -
             b**2 * g**2 + b**2 * h**2 - 2 * b * c * e * h +
             2 * b * c * f * g + 2 * b * d * e * g - 2 * b * d * f * h +
             c**2 * e**2 - c**2 * f**2 - d**2 * e**2 + d**2 * f**2)

    return Q / denom

if __name__ == "__main__":
    np.random.seed(1)
    a, b, c, d, e, f, g, h, w, x, y, z = np.random.randn(12)

    print(qmatrix(a, b, c, d, e, f, g, h, w, x, y, z) - _qmatrix(a, b, c, d, e, f, g, h, w, x, y, z) )
