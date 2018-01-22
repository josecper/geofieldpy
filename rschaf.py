import scha
import numpy


def calculate_degrees(kmax_int, kmax_ext, m_max, theta_0p):

    maxk = max(kmax_int, kmax_ext)
    ms = numpy.arange(0, maxk+1)
    roots = scha.degree(ms, theta0=theta_0p, max_k=maxk)
    k, m, n = numpy.array(scha.join_roots(roots))
    km_even = ((k-numpy.abs(m)) % 2 == 0) & (k <= maxk)
    k_even, m_even, n_even = k[km_even], m[km_even], n[km_even]
    
    k_int = k_even[k_even <= kmax_int]
    m_int = m_even[k_even <= kmax_int]
    n_int = n_even[k_even <= kmax_int]
    
    k_ext = k_even[(k_even > 0) & (k_even <= kmax_ext)]
    m_ext = m_even[(k_even > 0) & (k_even <= kmax_ext)]
    n_ext = n_even[(k_even > 0) & (k_even <= kmax_ext)]
    
    m_mehler = [0]
    for mm in numpy.arange(1, m_max+1):
        m_mehler.extend([mm, -mm])
    m_mehler = numpy.array(m_mehler)
    return (k_int, m_int, n_int, k_ext, m_ext, n_ext, m_mehler)


def remove_outliers(D, I, F, a95, F_err):

    out_a95 = 13
    out_F_err = 15
    
