#ifndef PYBNESIAN_UTIL_UNIROOT_HPP
#define PYBNESIAN_UTIL_UNIROOT_HPP

// Extracted from https://www.netlib.org/c/brent.shar

// A variant is also available in R source:
// https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/zeroin.c

/* This version includes some changes to allow different floating point types and is more
    similar to C++ code.
*/

/*************************************************************************
 *			    C math library
 * function ZEROIN - obtain a function zero within the given range
 *
 * Input
 *  template<typename F, typename T>
 *	double uniroot(F function, T ax, T bx, T tol, int maxit)
 *  F function          Function to be maximized, can be a lambda or function object.
 *	T ax;			Root will be seeked for within
 *	T bx;			a range [ax,bx]
 *	double tol;			Acceptable tolerance for the root
 *					value.
 *					May be specified as 0.0 to cause
 *					the program to find the root as
 *					accurate as possible
 *
 *	int maxit;			Max. iterations
 *
 *
 * Output
 *	Zeroin returns an estimate for the root with accuracy
 *	4*EPSILON*abs(x) + tol
 *
 *  It throws a runtime_exception if a solution could not be found.
 *
 * Algorithm
 *	G.Forsythe, M.Malcolm, C.Moler, Computer methods for mathematical
 *	computations. M., Mir, 1980, p.180 of the Russian edition
 *
 *	The function makes use of the bisection procedure combined with
 *	the linear or quadric inverse interpolation.
 *	At every step program operates on three abscissae - a, b, and c.
 *	b - the last and the best approximation to the root
 *	a - the last but one approximation
 *	c - the last but one or even earlier approximation than a that
 *		1) |f(b)| <= |f(c)|
 *		2) f(b) and f(c) have opposite signs, i.e. b and c confine
 *		   the root
 *	At every step Zeroin selects one of the two new approximations, the
 *	former being obtained by the bisection procedure and the latter
 *	resulting in the interpolation (if a,b, and c are all different
 *	the quadric interpolation is utilized, otherwise the linear one).
 *	If the latter (i.e. obtained by the interpolation) point is
 *	reasonable (i.e. lies within the current interval [b,c] not being
 *	too close to the boundaries) it is accepted. The bisection result
 *	is used in the other case. Therefore, the range of uncertainty is
 *	ensured to be reduced at least by the factor 1.6
 *
 ************************************************************************
 */

namespace util {

template <typename F, typename T>
T uniroot(F function,  /* Function under investigation	*/
          T ax,        /* Left border | of the range	*/
          T bx,        /* Right border| the root is seeked*/
          T tol,       /* Acceptable tolerance		*/
          int maxit) { /* Max # of iterations */

    T a = ax;
    T b = bx;
    T c = a;

    T fa = function(a);
    T fb = function(b);

    T fc = fa;
    ++maxit;

    /* First test if we have found a root at an endpoint */
    if (fa == 0.0) {
        return a;
    }

    if (fb == 0.0) {
        return b;
    }

    while (maxit--) {        /* Main iteration loop	*/
        T prev_step = b - a; /* Distance from the last but one
                         to the last approximation	*/
        T tol_act;           /* Actual tolerance		*/
        T p;                 /* Interpolation step is calcu- */
        T q;                 /* lated in the form p/q; divi-
                              * sion operations is delayed
                              * until the last moment	*/
        T new_step;          /* Step at this iteration	*/

        if (fabs(fc) < fabs(fb)) { /* Swap data for b to be the	*/
            a = b;
            b = c;
            c = a; /* best approximation		*/
            fa = fb;
            fb = fc;
            fc = fa;
        }

        tol_act = 2. * std::numeric_limits<T>::epsilon() * fabs(b) + tol / 2;
        new_step = (c - b) / 2;

        if (fabs(new_step) <= tol_act || fb == static_cast<T>(0)) {
            return b; /* Acceptable approx. is found	*/
        }

        /* Decide if the interpolation can be tried	*/
        if (fabs(prev_step) >= tol_act /* If prev_step was large enough*/ &&
            fabs(fa) > fabs(fb)) { /* and was in true direction,
                                    * Interpolation may be tried	*/
            T cb = c - b;

            if (a == c) {       /* If we have only two distinct	*/
                                /* points linear interpolation	*/
                T t1 = fb / fa; /* can only be applied		*/
                p = cb * t1;
                q = 1.0 - t1;
            } else { /* Quadric inverse interpolation*/
                q = fa / fc;
                T t1 = fb / fc;
                T t2 = fb / fa;
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0));
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0);
            }

            if (p > static_cast<T>(0)) /* p was calculated with the */
                q = -q;                /* opposite sign; make p positive */
            else                       /* and assign possible minus to	*/
                p = -p;                /* q				*/

            if (p < (0.75 * cb * q - fabs(tol_act * q) / 2) /* If b+p/q falls in [b,c]*/ &&
                p < fabs(prev_step * q / 2)) /* and isn't too large	*/
                new_step = p / q;            /* it is accepted
                                              * If p/q is too large then the
                                              * bisection procedure can
                                              * reduce [b,c] range to more
                                              * extent */
        }

        if (fabs(new_step) < tol_act) {       /* Adjust the step to be not less*/
            if (new_step > static_cast<T>(0)) /* than tolerance		*/
                new_step = tol_act;
            else
                new_step = -tol_act;
        }

        a = b;
        fa = fb; /* Save the previous approx. */
        b += new_step;
        fb = function(b); /* Do step to a new approxim. */
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            /* Adjust c for it to have a sign opposite to that of b */
            c = a;
            fc = fa;
        }
    }
    /* failed! */
    throw std::invalid_argument("A solution could not be found with enough precision!");
}

}  // namespace util

#endif  // PYBNESIAN_UTIL_UNIROOT_HPP