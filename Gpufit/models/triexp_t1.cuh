#ifndef GPUFIT_TRIEXP_T1_CUH_INCLUDED
#define GPUFIT_TRIEXP_T1_CUH_INCLUDED

/* Description of the calculate_triexp function
* ===================================================
*
* This function calculates the values of triexponential functions
* and their partial derivatives with respect to the model parameters.
*
* The triexponential function is: y = (a*exp(-bx) + c*exp(-dx) + e*exp(-fx))*(1 - exp(-TR/g)) 
* TR: repetition time
* The derivatives are:
* dy/da = exp(-bx)
* dy/db = -a*x*exp(-b*x)         // a*(d/(d*b)exp(-bx))
* dy/dc = exp(-dx)
* dy/dd = -c*x*exp(-d*x)         // c*(d/(d*d)exp(-dx))
* dy/de = exp(-fx)
* dy/df = -e*x*exp(-f*x)         // e*(d/(d*f)exp(-fx))
* dy/dg = (a*exp(-bx) + c*exp(-dx) + e*exp(-fx))*(-TR/g²)*exp(-TR/g)
*
* This function makes use of the user information data to pass in the
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* Note that if no user information is provided, the (X) coordinate of the
* first data value is assumed to be (0.0).  In this case, for a fit size of
* M data points, the (X) coordinates of the data are simply the corresponding
* array index values of the data array, starting from zero.
*
* There are three possibilities regarding the X values:
*
*   No X values provided:
*
*       If no user information is provided, the (X) coordinate of the
*       first data value is assumed to be (0.0).  In this case, for a
*       fit size of M data points, the (X) coordinates of the data are
*       simply the corresponding array index values of the data array,
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then
*       the same X values will be used for all fits.  In this case, the
*       size of the user_info array (in bytes) must equal
*       sizeof(REAL) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array
*       (in bytes) must equal sizeof(REAL) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: a   p[1]: b     p[2]: c
*             p[3]: d   p[4]: e     p[5]: f
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_triexp function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_triexp(
    REAL const* parameters,
    int const n_fits,
    int const n_points,
    REAL* value,
    REAL* derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char* user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL* user_info_float = (REAL*)user_info;

    // Read the last entry (TR value)
    REAL const TR = user_info_float[user_info_size / sizeof(REAL) - 1];
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters
    REAL const* p = parameters;

    /* value
    y = (a*exp(-bx) + c*exp(-dx) + e*exp(-fx))*(1 - exp(-TR/g))
    p[0]: a  (f1)   p[1]: b  (D1)   p[2]: c  (f2)  p[3]: d  (D2)  p[4]: e   (f3)  p[5]: f (D3)    p[6]: g (T1) */


    // value

    value[point_index] = 
        (p[0] * exp(-p[1] * x) +
        p[2] * exp(-p[3] * x) +
        p[4] * exp(-p[5] * x)) * 
        (1 - exp(-TR / p[6]));

    /* derivatives
    dy/da = exp(-bx)
    dy/db = -a*x*exp(-b*x)         // a*(d/(d*b)exp(-bx))
    dy/dc = exp(-dx)
    dy/dd = -c*x*exp(-d*x)         // c*(d/(d*d)exp(-dx))
    dy/de = exp(-fx)
    dy/df = -e*x*exp(-f*x)         // e*(d/(d*f)exp(-fx))   
    dy/dg = (a*exp(-bx) + c*exp(-dx) + e*exp(-fx))*(-TR/g²)*exp(-TR/g)
    */

    REAL* current_derivatives = derivative + point_index;
    current_derivatives[0 * n_points] = exp(-p[1] * x);
    current_derivatives[1 * n_points] = p[0] * x * exp(-p[1] * x);
    current_derivatives[2 * n_points] = exp(-p[3] * x);
    current_derivatives[3 * n_points] = p[2] * x * exp(-p[3] * x);
    current_derivatives[4 * n_points] = exp(-p[5] * x);
    current_derivatives[5 * n_points] = p[4] * x * exp(-p[5] * x);
    // IVIM * (-TR/(g*g))*exp(-TR/g)
    current_derivatives[6 * n_points] =
        (
            p[0] * exp(-p[1] * x) +
            p[2] * exp(-p[3] * x) +
            p[4] * exp(-p[5] * x)) *
            (-TR / (p[6] * p[6])) * exp(-TR / p[6]);
}

#endif
