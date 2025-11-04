#ifndef GPUFIT_MONOEXP_T1_STEAM_CUH_INCLUDED
#define GPUFIT_MONOEXP_T1_STEAM_CUH_INCLUDED

/* Description of the calculate_monoexp function
* ===================================================
*
* This function calculates the values of monoexponential functions
* and their partial derivatives with respect to the model parameters.
*
* The monoexponential function is: y = a*exp(-b*x) * (1 - exp(-TR/c)) * (exp(-t_mix/c))
* The derivatives are:
* dy/da = exp(-b*x) * (1 - exp(-TR/c)) * (exp(-t_mix/c))
* dy/db = a*(-x)*exp(-b*x) * (1 - exp(-TR/c)) * (exp(-t_mix/c))
* dy/dc = a*exp(-b*x) * (1 - exp(-TR/c)) * (TR/c²) * (t_mix/c²) * (exp(-t_mix/c))
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
*             p[0]: a   p[1]: b     
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
* Calling the calculate_monoexp function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_monoexp_t1_steam(
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
    // reduce user_info_size by one since the last entry is reserved for TR
    std::size_t const reduced_user_info_size = user_info_size - sizeof(REAL);

    REAL* user_info_float = (REAL*)user_info;

    // Read the last entry (TR value)
    REAL const TR = user_info_float[user_info_size / sizeof(REAL) - 2];    
    REAL const t_mix = user_info_float[user_info_size / sizeof(REAL) - 1];
    
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (reduced_user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (reduced_user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters
    REAL const* p = parameters;

    /* value
    a*exp(-b*x)*(1-exp(-TR/c))
    p[0]: a   p[1]: b   p[2]: c     */
    value[point_index] = p[0] * exp(-p[1] * x) * (1 - exp(-TR / p[2])) * (exp(-t_mix / p[2]));
    
    /* derivatives
    dy/da = exp(-b*x)
    dy/db = a*(-x)*exp(-b*x)        
    dy/dc = a*exp(-b*x)*(TR/(c*c))*exp(-TR/c)
    */
    REAL* current_derivatives = derivative + point_index;
    current_derivatives[0 * n_points] = exp(-p[1] * x) * (1 - exp(-TR / p[2])) * (exp(-t_mix / p[2]));
    current_derivatives[1 * n_points] = p[0] * (-x) * exp(-p[1] * x) * (1 - exp(-TR / p[2])) * (exp(-t_mix / p[2]));
    current_derivatives[2 * n_points] = p[0] * exp(-p[1] * x) * (TR / (p[2] * p[2])) * (t_mix / (p[2] * p[2])) * exp(-TR / p[2]);
}

#endif // GPUFIT_MONOEXP_T1_STEAM_CUH_INCLUDED