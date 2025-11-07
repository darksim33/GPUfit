#ifndef GPUFIT_MODELS_CUH_INCLUDED
#define GPUFIT_MODELS_CUH_INCLUDED

#include <assert.h>
#include "linear_1d.cuh"
#include "gauss_1d.cuh"
#include "gauss_2d.cuh"
#include "gauss_2d_elliptic.cuh"
#include "gauss_2d_rotated.cuh"
#include "cauchy_2d_elliptic.cuh"
#include "fletcher_powell_helix.cuh"
#include "brown_dennis.cuh"
#include "spline_1d.cuh"
#include "spline_2d.cuh"
#include "spline_3d.cuh"
#include "spline_3d_multichannel.cuh"
#include "spline_3d_phase_multichannel.cuh"
#include "spline_4d.cuh"
#include "spline_5d.cuh"
#include "triexp.cuh"
#include "triexp_red.cuh"
#include "triexp_s0.cuh"
#include "triexp_t1.cuh"
#include "triexp_t1_steam.cuh"
#include "triexp_s0_t1.cuh"
#include "triexp_s0_t1_steam.cuh"
#include "biexp.cuh"
#include "biexp_red.cuh"
#include "biexp_s0.cuh"
#include "biexp_t1.cuh"
#include "biexp_t1_steam.cuh"
#include "biexp_s0_t1.cuh"
#include "biexp_s0_t1_steam.cuh"
#include "monoexp.cuh"
#include "monoexp_red.cuh"
#include "monoexp_t1.cuh"
#include "monoexp_t1_steam.cuh"

__device__ void calculate_model(
    ModelID const model_id,
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    int const user_info_size)
{
    switch (model_id)
    {
    case GAUSS_1D:
        calculate_gauss1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D:
        calculate_gauss2d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D_ELLIPTIC:
        calculate_gauss2delliptic(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case GAUSS_2D_ROTATED:
        calculate_gauss2drotated(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case CAUCHY_2D_ELLIPTIC:
        calculate_cauchy2delliptic(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case LINEAR_1D:
        calculate_linear1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case FLETCHER_POWELL_HELIX:
        calculate_fletcher_powell_helix(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BROWN_DENNIS:
        calculate_brown_dennis(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_1D:
        calculate_spline1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_2D:
        calculate_spline2d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_3D:
        calculate_spline3d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_3D_MULTICHANNEL:
        calculate_spline3d_multichannel(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_3D_PHASE_MULTICHANNEL:
        calculate_spline3d_phase_multichannel(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_4D:
        calculate_spline4d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case SPLINE_5D:
        calculate_spline5d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case TRIEXP:
        calculate_triexp(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case TRIEXP_RED:
        calculate_triexp_red(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case TRIEXP_S0:
        calculate_triexp_s0(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case TRIEXP_T1:
        calculate_triexp_t1(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case TRIEXP_T1_STEAM:
        calculate_triexp_t1_steam(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case TRIEXP_S0_T1:
        calculate_triexp_s0_t1(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
        case TRIEXP_S0_T1_STEAM:
    calculate_triexp_s0_t1_steam(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);;
        break;
    case BIEXP:
        calculate_biexp(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BIEXP_RED:
        calculate_biexp_red(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BIEXP_S0:
        calculate_biexp_s0(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BIEXP_T1:
        calculate_biexp_t1(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BIEXP_T1_STEAM:
        calculate_biexp_t1_steam(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BIEXP_S0_T1:
        calculate_biexp_s0_t1(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case BIEXP_S0_T1_STEAM:
        calculate_biexp_s0_t1_steam(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case MONOEXP:
        calculate_monoexp(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case MONOEXP_RED:
        calculate_monoexp_red(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case MONOEXP_T1:
        calculate_monoexp_t1(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    case MONOEXP_T1_STEAM:
        calculate_monoexp_t1_steam(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
    default:
        assert(0); // unknown model ID
    }
}

void configure_model(ModelID const model_id, int & n_parameters, int & n_dimensions)
{
    switch (model_id)
    {
    case GAUSS_1D:              n_parameters = 4; n_dimensions = 1; break;
    case GAUSS_2D:              n_parameters = 5; n_dimensions = 2; break;
    case GAUSS_2D_ELLIPTIC:     n_parameters = 6; n_dimensions = 2; break;
    case GAUSS_2D_ROTATED:      n_parameters = 7; n_dimensions = 2; break;
    case CAUCHY_2D_ELLIPTIC:    n_parameters = 6; n_dimensions = 2; break;
    case LINEAR_1D:             n_parameters = 2; n_dimensions = 1; break;
    case FLETCHER_POWELL_HELIX: n_parameters = 3; n_dimensions = 1; break;
    case BROWN_DENNIS:          n_parameters = 4; n_dimensions = 1; break;
    case SPLINE_1D:             n_parameters = 3; n_dimensions = 1; break;
    case SPLINE_2D:             n_parameters = 4; n_dimensions = 2; break;
    case SPLINE_3D:             n_parameters = 5; n_dimensions = 3; break;
    case SPLINE_3D_MULTICHANNEL:         n_parameters = 5; n_dimensions = 4; break;
    case SPLINE_3D_PHASE_MULTICHANNEL:   n_parameters = 6; n_dimensions = 4; break;
    case SPLINE_4D:             n_parameters = 6; n_dimensions = 4; break;
    case SPLINE_5D:             n_parameters = 7; n_dimensions = 5; break;
    case TRIEXP:                n_parameters = 6; n_dimensions = 1; break;
    case TRIEXP_RED:            n_parameters = 5; n_dimensions = 1; break;
    case TRIEXP_S0:             n_parameters = 6; n_dimensions = 1; break;
    case TRIEXP_T1:             n_parameters = 7; n_dimensions = 1; break;
    case TRIEXP_T1_STEAM:       n_parameters = 7; n_dimensions = 1; break;
    case TRIEXP_S0_T1:          n_parameters = 7; n_dimensions = 1; break;
    case TRIEXP_S0_T1_STEAM:    n_parameters = 7; n_dimensions = 1; break;
    case BIEXP:                 n_parameters = 4; n_dimensions = 1; break;
    case BIEXP_RED:             n_parameters = 3; n_dimensions = 1; break;
    case BIEXP_S0:              n_parameters = 4; n_dimensions = 1; break;
    case BIEXP_T1:              n_parameters = 5; n_dimensions = 1; break;
    case BIEXP_T1_STEAM:        n_parameters = 5; n_dimensions = 1; break;
    case BIEXP_S0_T1:           n_parameters = 5; n_dimensions = 1; break;
    case BIEXP_S0_T1_STEAM:     n_parameters = 5; n_dimensions = 1; break;
    case MONOEXP:               n_parameters = 2; n_dimensions = 1; break;
    case MONOEXP_RED:           n_parameters = 1; n_dimensions = 1; break;
    case MONOEXP_T1:            n_parameters = 3; n_dimensions = 1; break;
    case MONOEXP_T1_STEAM:      n_parameters = 3; n_dimensions = 1; break;
    default: throw std::runtime_error("unknown model ID");
    }
}

#endif // GPUFIT_MODELS_CUH_INCLUDED