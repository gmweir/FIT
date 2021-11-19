


import numpy as _np
import matplotlib.pyplot as _plt
from astropy.modeling import models, fitting
import scipy.optimize

# Make plots display in notebooks



def test_astropy_quickfit(model=None, fitter=None):
    try:
        # this is for downloading existing data sets
        from astroquery.vizier import Vizier

        catalog = Vizier.get_catalogs('J/A+A/605/A100')


        # periods and magnitudes
        period = np.array(catalog[0]['Period'])
        log_period = np.log10(period)
        k_mag = np.array(catalog[0]['__Ksmag_'])
        k_mag_err = np.array(catalog[0]['e__Ksmag_'])        # error bars in magnitude measurements

    except:
        # make your own model data
        from numpy.random import default_rng

        Npts= 100
        slope = -2.0
        intercept = 14.0

        rng = default_rng(0)
        # period = 39.0*rng.random((Npts,)) + 1.0
        # period = _np.sort(period)
        # log_period = _np.log10(period)

        log_period = 1.6*rng.random((Npts,))
        log_period = _np.sort(log_period)
        period = 10.0**(log_period)

        # period = _np.linspace(start=1.0, stop=40.0, num=Npts, endpoint=True)
        # period += 10.0**(0.1*rng.random((Npts,))) #
        # log_period = _np.log10(period)
        # log_period += 0.1*rng.random((Npts,))  # distributed between 0 and 0.1

        rng = default_rng(1)
        k_mag = slope*log_period + intercept
        k_mag += 1.0*(rng.random((Npts,))-0.5) # distributed between -0.5 and 0.5

        rng = default_rng(2)
        k_mag_err = 0.01*k_mag + 0.1*(rng.random((Npts,))-0.5) # distributed between -0.05 and 0.05
    # end try

    _plt.figure()
    _plt.errorbar(log_period, k_mag, k_mag_err, fmt='k.')
    _plt.xlabel(r'$\log_{10}$(Period [days])')
    _plt.ylabel('Ks')
    _plt.Text(0, 0.5, 'Ks')

    # Choose model to fit
    if model is None:
        model = models.Linear1D()

    # Choose method to fit
    if fitter is None:
        fitter = fitting.LinearLSQFitter()

    # Fit the data
    best_fit = fitter(model, log_period, k_mag, weights=1.0/k_mag_err**2)
    print(best_fit)

    _plt.errorbar(log_period,k_mag,k_mag_err,fmt='k.')
    _plt.plot(log_period, best_fit(log_period), color='g', linewidth=3)
    _plt.xlabel(r'$\log_{10}$(Period [days])')
    _plt.ylabel('Ks')

    _plt.show()
    return best_fit
# end def test_astropy_quickfit

def test_astropy_poly(model=None, fitter=None):
    N = 100
    x1 = _np.linspace(0, 4, N)  # Makes an array from 0 to 4 of N elements
    y1 = x1**3.0 - 6.0*x1**2.0 + 12.0*x1 - 9.0

    # Now we add some noise to the data
    y1 += _np.random.normal(0, 2.0, size=len(y1)) #One way to add random gaussian noise
    sigma = 1.5
    y1_err = _np.ones(N)*sigma

    _plt.figure()
    _plt.errorbar(x1, y1, yerr=y1_err,fmt='k.')
    _plt.xlabel(r'$x_1$')
    _plt.ylabel(r'$y_1$')
    _plt.Text(0, 0.5, '$y_1$')

    model_poly = models.Polynomial1D(degree=3)
    fitter_poly = fitting.LinearLSQFitter()
    best_fit_poly = fitter_poly(model_poly, x1, y1, weights = 1.0/y1_err**2)
    print(best_fit_poly)

    fitter_poly_2 = fitting.SimplexLSQFitter()
    best_fit_poly_2 = fitter_poly_2(model_poly, x1, y1, weights = 1.0/y1_err**2)
    print(best_fit_poly_2)

    reduced_chi_squared = calc_reduced_chi_square(best_fit_poly(x1), x1, y1, y1_err, N, 4)
    print('Reduced Chi Squared with LinearLSQFitter: {}'.format(reduced_chi_squared))

    reduced_chi_squared2 = calc_reduced_chi_square(best_fit_poly_2(x1), x1, y1, y1_err, N, 4)
    print('Reduced Chi Squared with SimplexLSQFitter: {}'.format(reduced_chi_squared2))

    _plt.figure()
    _plt.errorbar(x1, y1, yerr=y1_err,fmt='k.')
    _plt.plot(x1, best_fit_poly(x1), color='r', linewidth=3, label='LinearLSQFitter()')
    _plt.plot(x1, best_fit_poly_2(x1), color='g', linewidth=3, label='SimplexLSQFitter()')
    _plt.xlabel(r'$\log_{10}$(Period [days])')
    _plt.ylabel('Ks')

    if 0:
        # Choose model to fit
        if model is None:
            model = models.Polynomial1D(degree=2)

        # Choose method to fit
        if fitter is None:
            fitter = fitting.LinearLSQFitter()

        best_fit_alt = fitter(model, x1, y1, weights = 1.0/y1_err**2)

        print(best_fit_alt)

        reduced_chi_squared_alt = calc_reduced_chi_square(best_fit_poly_alt(x1), x1, y1, y1_err, N, 4)
        print('Reduced Chi Squared with Alt. Model in LSQFitter: {}'.format(reduced_chi_squared_alt))

        _plt.plot(x1, best_fit_poly_alt(x1), color='b', linewidth=3, label='Alt')
    # end if
    _plt.legend()

    return best_fit_poly

def test_guassian1d():

    x2, y2, y2_err = gaussian_data()
    N2 = len(y2)

    _plt.figure()
    _plt.errorbar(x2, y2, yerr=y2_err, fmt='k.')
    _plt.xlabel('$x_2$')
    _plt.ylabel('$y_2$')
    #_plt.Text(0, 0.5, '$y_2$')

    model_gauss = models.Gaussian1D()
    fitter_gauss = fitting.LevMarLSQFitter()
    best_fit_gauss = fitter_gauss(model_gauss, x2, y2, weights=1/y2_err**2)

    print(best_fit_gauss)
    print(model_gauss.param_names)

    cov_diag = _np.diag(fitter_gauss.fit_info['param_cov'])
    print(cov_diag)

    print('Amplitude: {} +\- {}'.format(best_fit_gauss.amplitude.value, _np.sqrt(cov_diag[0])))
    print('Mean: {} +\- {}'.format(best_fit_gauss.mean.value, _np.sqrt(cov_diag[1])))
    print('Standard Deviation: {} +\- {}'.format(best_fit_gauss.stddev.value, _np.sqrt(cov_diag[2])))

    reduced_chi_squared = calc_reduced_chi_square(best_fit_gauss(x2), x2, y2, y2_err, N2, 3)
    print('Reduced Chi Squared using astropy.modeling: {}'.format(reduced_chi_squared))

    _plt.errorbar(x2, y2, yerr=y2_err, fmt='k.')
    _plt.plot(x2, best_fit_gauss(x2), 'g-', linewidth=6, label='astropy.modeling')
    if 1:
        def f(x,a,b,c):
            return a * np.exp(-(x-b)**2/(2.0*c**2))

        import scipy
        p_opt, p_cov = scipy.optimize.curve_fit(f,x2, y2, sigma=y1_err)
        a,b,c = p_opt
        best_fit_gauss_2 = f(x2,a,b,c)

        print(p_opt)

        print('Amplitude: {} +\- {}'.format(p_opt[0], _np.sqrt(p_cov[0,0])))
        print('Mean: {} +\- {}'.format(p_opt[1], _np.sqrt(p_cov[1,1])))
        print('Standard Deviation: {} +\- {}'.format(p_opt[2], _np.sqrt(p_cov[2,2])))

        reduced_chi_squared_scipy = calc_reduced_chi_square(best_fit_gauss_2, x2, y2, y2_err, N2, 3)
        print('Reduced Chi Squared using scipy: {}'.format(reduced_chi_squared_scipy))

        _plt.plot(x2, best_fit_gauss_2, 'r-', linewidth=2, label='scipy')
    # end if
    _plt.xlabel('$x_2$')
    _plt.ylabel('$y_2$')
    _plt.legend()

    return best_fit_gauss


def calc_reduced_chi_square(fit, x, y, yerr, N, n_free):
    '''
    fit (array) values for the fit
    x,y,yerr (arrays) data
    N total number of points
    n_free number of parameters we are fitting
    '''
    return 1.0/(N-n_free)*sum(((fit - y)/yerr)**2)


def gaussian_data(mu=0.0, sigma=10.0, amplitude=10.0, N=100):
    # mu, sigma, amplitude = 0.0, 10.0, 10.0
    x2 = _np.linspace(-30, 30, N)
    y2 = amplitude * _np.exp(-(x2-mu)**2 / (2*sigma**2))
    y2 = _np.array([y_point + _np.random.normal(0, 1) for y_point in y2])   #Another way to add random gaussian noise
    sigma = 1
    y2_err = _np.ones(N)*sigma

    return x2, y2, y2_err

if __name__=="__main__":

    best_fit_line = test_astropy_quickfit()
    best_fit_poly = test_astropy_poly()
    best_fit_gauss = test_guassian1d()

# end if