import torch
import numpy as np
from typing import Tuple
from torch import nn
from scipy.special import eval_jacobi


def generate_gaussian_psf_from_sigma(sigma, xi0,xi1,xi2):
    sig = -sigma/2
    Gridxi0, Gridxi1, Gridxi2 = torch.meshgrid(xi0,xi1,xi2,indexing = 'ij')
    ems = torch.exp(sig * (Gridxi0**2 + Gridxi1**2+ Gridxi2**2))
    return ems


def generate_gaussian_psf(sigma, xi0,xi1,xi2):
    Gridxi0, Gridxi1, Gridxi2 = torch.meshgrid(xi0,xi1,xi2,indexing = 'ij')
    ems = torch.exp(-(Gridxi0*(sigma[0,0]*Gridxi0+sigma[0,1]*Gridxi1+sigma[0,2]*Gridxi2) \
        + Gridxi1*(sigma[1,0]*Gridxi0+sigma[1,1]*Gridxi1+sigma[1,2]*Gridxi2) \
            + Gridxi2*(sigma[2,0]*Gridxi0+sigma[2,1]*Gridxi1+sigma[2,2]*Gridxi2))/2)
    return ems


class Gibson_Lanni_psf(nn.Module):
    # Torch version of http://kmdouglass.github.io/posts/implementing-a-fast-gibson-lanni-psf-solver-in-python/
    # Precision control
    #num_basis: Number of rescaled Bessels that approximate the phase function
    #num_samples: Number of pupil samples along radial direction
    #min_wavelength: microns
    """Meaning of the parameters: 
    NA: Numerical Aperture of the objective lens. NA = ni sin(Θ) where Θ is one half of the angular aperture of the lens.
    wavelength: microns
    M: magnification
    ns: specimen refractive index (RI)
    ng0: coverslip RI design value
    ng: coverslip RI experimental value
    ni0: immersion medium RI design value
    ni: immersion medium RI experimental value
    ti0: microns, working distance (immersion medium thickness) design value
    tg0: microns, coverslip thickness design value
    tg: microns, coverslip thickness experimental value
    res_lateral: microns
    res_axial: microns
    pZ: microns, particle distance from coverslip
    
    """

    def __init__(self,box_shape, NA=1.4,M=100., wavelength=0.610, res_lateral= 0.1, res_axial= 0.1,
                 ns=1.33, ng0=1.499, ng=1.5, ni0=1.499, ni=1.5, ti0=150, tg0=170, tg=169.99,
                pZ=2.0, min_wavelength = 0.436, num_basis = 100, num_samples = 1000, device = 'cpu'):
        super().__init__()

        self.device = device
        self.NA = self.make_param(NA)
        #self.M = self.make_param(M)
        self.wavelength = self.make_param(wavelength)
        self.res_lateral = self.make_param(res_lateral)
        self.res_axial = self.make_param(res_axial)
        self.ns = self.make_param(ns)
        self.ng0 = self.make_param(ng0)
        self.ng = self.make_param(ng)
        self.ni0 = self.make_param(ni0)
        self.ni = self.make_param(ni)
        self.ti0 = self.make_param(ti0)
        self.tg0 = self.make_param(tg0)
        self.tg = self.make_param(tg)
        self.pZ = self.make_param(pZ)
    
        # Place the origin at the center of the final PSF array
        self.size_x, self.size_y, self.size_z = box_shape
        self.eps = 2.22045e-16#1.19209e-07 #For double:2.22045e-16     For double: 1.19209e-07
        self.Seps = np.sqrt(self.eps)
        self.min_wavelength = min_wavelength
        self.num_basis = num_basis
        self.num_samples = 1000


    def forward(self):
        
        x0 = (self.size_x - 1) / 2
        y0 = (self.size_y - 1) / 2

        # Scaling factors for the Fourier-Bessel series expansion
        scaling_factor = self.NA * (3 * torch.arange(1, self.num_basis + 1,device = self.device,dtype=torch.double) - 2) * self.min_wavelength / self.wavelength

        # Radial coordinates, pupil space
        rhomax = min([self.NA, self.ns, self.ni, self.ni0, self.ng, self.ng0]) / self.NA
        rho = torch.linspace(0, rhomax.item(), self.num_samples,device = self.device)

        #Some verifications because of numerical precision.
        thresh = -self.Seps
        assert ((self.ns * self.ns - self.NA * self.NA * rhomax * rhomax).min()) >=thresh
        assert ((self.ni * self.ni - self.NA * self.NA * rhomax * rhomax).min()) >=thresh
        assert ((self.ni0 * self.ni0 - self.NA * self.NA * rho * rho).min()) >=thresh
        assert ((self.ns * self.ns - self.NA * self.NA * rho * rho).min()) >=thresh
        assert ((self.ng0 * self.ng0 - self.NA * self.NA * rho * rho).min()) >=thresh

        # Define the wavefront aberration

        # Sample the phase
        # Shape is (number of z samples by number of rho samples)
        #phase = np.cos(W) + 1j * np.sin(W)
        #zv : todo : define it logically with positive and negative bounds
        # I think this is good now
         # Stage displacements away from best focus
        zv = (self.res_axial * torch.arange(-self.size_z / 2, self.size_z /2, device = self.device) \
            + self.res_axial / 2).type(torch.double)
        #ti = zv.reshape(-1,1) + self.ti0

        #a = NA * zd0 / M #
        #a = self.NA * self.zd0 / torch.sqrt(self.M*self.M + self.NA*self.NA) 
        #a = min([NA, ns, ni, ni0, ng, ng0]) / NA
        #I have found the two expressions, do not know which one is the right one.
        #OPDs = self.pZ * torch.sqrt((self.ns * self.ns - self.NA * self.NA * rho * rho).clamp(self.Seps)) # OPD in the sample.
        #OPDi = ti * torch.sqrt((self.ni * self.ni - self.NA * self.NA * rho * rho).clamp(self.Seps)) - self.ti0 * torch.sqrt((self.ni0 * self.ni0 - self.NA * self.NA * rho * rho).clamp(self.Seps)) # OPD in the immersion medium.
        #OPDg = self.tg * torch.sqrt((self.ng * self.ng - self.NA * self.NA * rho * rho).clamp(self.Seps)) - self.tg0 * torch.sqrt((self.ng0 * self.ng0 - self.NA * self.NA * rho * rho).clamp(self.Seps)) # OPD in the coverslip.
        #OPDt = a * a * (self.zd0 - self.zd) * rho * rho / (2.0 * self.zd0 * self.zd) # OPD in camera position.
        #W= 2 * np.pi / self.wavelength * (OPDs + OPDi + OPDg + OPDt)

        OPDs = self.pZ * torch.sqrt((self.ns * self.ns - self.NA * self.NA * rho * rho).clamp(self.Seps)) # OPD in the sample
        OPDi = (zv.reshape(-1,1) + self.ti0) * torch.sqrt((self.ni * self.ni - self.NA * self.NA * rho * rho).clamp(self.Seps)) \
            - self.ti0 * torch.sqrt((self.ni0 * self.ni0 - self.NA * self.NA * rho * rho).clamp(self.Seps)) # OPD in the immersion medium
        OPDg = self.tg * torch.sqrt((self.ng * self.ng - self.NA * self.NA * rho * rho).clamp(self.Seps)) - self.tg0 \
            * torch.sqrt((self.ng0 * self.ng0 - self.NA * self.NA * rho * rho).clamp(self.Seps))
        W = 2 * np.pi / self.wavelength * (OPDs + OPDi + OPDg)

        # Sample the phase
        phase = torch.cos(W) + 1j * torch.sin(W)

        # Define the basis of Bessel functions
        # Shape is (number of basis functions by number of rho samples)
        J = torch.special.bessel_j0( scaling_factor.reshape(-1, 1) * rho).type(torch.complex128)

        # Compute the approximation to the sampled pupil phase by finding the least squares
        # solution to the complex coefficients of the Fourier-Bessel expansion.
        # Shape of C is (number of basis functions by number of z samples).
        # Note the matrix transpose to get the dimensions correct.
        C, _, _, _ = torch.linalg.lstsq(J.T, phase.T)

        GridX,GridY = torch.meshgrid(torch.arange(self.size_x,device = self.device,dtype=torch.double),\
            torch.arange(self.size_y,device = self.device,dtype=torch.double),indexing = 'ij')
        r_pixel = torch.sqrt((GridX - x0) * (GridX - x0) + (GridY - y0) * (GridY - y0)) * self.res_lateral

        b = 2 * np. pi * r_pixel.reshape(r_pixel.shape[0],r_pixel.shape[1],1) * self.NA / self.wavelength
        denom = scaling_factor * scaling_factor - b * b
        R = ((scaling_factor * torch.special.bessel_j1(scaling_factor * rhomax) * torch.special.bessel_j0(b * rhomax) * rhomax -
             b * torch.special.bessel_j0(scaling_factor * rhomax) * torch.special.bessel_j1(b * rhomax) * rhomax)/denom).type(torch.complex128)
        PSF_nonnorm = (torch.abs(R@C)**2)
        PSF = PSF_nonnorm/torch.max(PSF_nonnorm)
        
        return PSF


    def make_param(self,x):
        return(torch.nn.parameter.Parameter(torch.tensor(x, dtype = torch.double,device = self.device,requires_grad = True)))


class Hanser_psf(nn.Module):

    #from the implementation of https://github.com/david-hoffman/pyOTF

    """
    To fully describe a PSF or OTF of an objective lens, assuming no
    abberation, we generally need a few parameters:
    - The wavelength of operation (assume monochromatic light)
    - the numerical aperature of the objective
    - the index of refraction of the medium
    For numerical calculations we'll also want to know the x/y resolution
    and number of points. Note that it is assumed that z is the optical
    axis of the objective lens

    Based on the following work

    [(1) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
    Phase-Retrieved Pupil Functions in Wide-Field Fluorescence Microscopy.
    Journal of Microscopy 2004, 216 (1), 32–48.](dx.doi.org/10.1111/j.0022-2720.2004.01393.x)
    [(2) Hanser, B. M.; Gustafsson, M. G. L.; Agard, D. A.; Sedat, J. W.
    Phase Retrieval for High-Numerical-Aperture Optical Systems.
    Optics Letters 2003, 28 (10), 801.](dx.doi.org/10.1364/OL.28.000801)
    """

    def __init__(self, box_shape,xi0,xi1,xi2, device= 'cpu',NA=1.4, wavelength=0.610, ni=1.5, res_lateral=0.1,
                res_axial=0.25, num_zernike = 21, vec_corr="none", condition="sine",with_grad = True):
        """Generate a PSF object.
        wl=520, na=0.85, ni=1.0, res=130, zres=300
        Parameters
        ----------
        box_shape : int
            x/y/z size of the simulation
        NA : numeric
            Numerical aperature of the simulation
        wavelength : numeric
            Emission wavelength of the simulation in nm
        ni : numeric
            index of refraction for the media
        res_lateral : numeric
            x/y resolution of the simulation, must have same units as wavelength
        res_axial : numeric
            z resolution of simuation, must have same units a wavelength
            
        mcoefs : ndarray (num_zernike, )
            The magnitude coefficiencts
        pcoefs : ndarray (num_zernike, )
            The phase coefficients
            
        We assume that the mcoefs and pcoefs are Noll ordered
        
        vec_corr : str
            keyword to indicate whether to include vectorial effects
                Valid options are: "none", "x", "y", "z", "total"
                Default is: "none"
        condition : str
            keyword to indicate whether to model the sine or herschel conditions
            **Herschel's Condition** invariance of axial magnification
            **Abbe's Sine Condition** invariance of lateral magnification
            conditions
                Valid options are: "none", "sine", "herschel"
                Default is: "sine"
                Note: "none" is not a physical solution
        """

        super().__init__()
        self.with_grad = with_grad
        self.size_x, self.size_y, self.size_z = box_shape
        self.device = device
        self.NA = self.make_param(NA)
        self.wavelength = self.make_param(wavelength)
        self.ni = self.make_param(ni)
        self.res_lateral = self.make_param(res_lateral)
        self.res_axial = self.make_param(res_axial)
        self.mcoefs = torch.nn.parameter.Parameter(torch.zeros(num_zernike,dtype = torch.float, requires_grad=self.with_grad,device = self.device))
        self.pcoefs = torch.nn.parameter.Parameter(torch.zeros(num_zernike,dtype = torch.float, requires_grad=self.with_grad,device = self.device))
        self.xi0 = xi0
        self.xi1 = xi1
        self.xi2 = xi2

        self.vec_corr = vec_corr
        self.condition = condition

        """zrange : array-like
        An alternate way to specify the z range for the calculation
        must be expressed in the same units as wavelength
        """

    def forward(self):
        self._gen_zrange()
        self._gen_kr()
        kr = self._kr
        theta = self._phi
        # make zernikes (need to convert kr to r where r = 1 when kr is at
        # diffraction limit)
        r = kr *self.wavelength / self.NA
        with torch.no_grad():
            zerns = torch.tensor(zernike(r.detach().cpu().numpy(), theta.detach().cpu().numpy(),
                    *noll2degrees(np.arange(len(self.mcoefs.detach().cpu())) + 1)),
                    dtype = torch.float,device = self.device)


        pupil_phase = (zerns * self.pcoefs[:, None, None]).sum(0)
        pupil_mag = (zerns * self.mcoefs[:, None, None]).sum(0)

        # apply aberrations to unaberrated pupil (N.B. the unaberrated phase is 0)
        pupil_total = (torch.abs(self._gen_pupil()) + pupil_mag) * torch.exp(1j * pupil_phase)
        # generate the PSF, assign to attribute
        PSFa = self._gen_psf(pupil_total).permute(0,2,3,1)

        return (torch.abs(PSFa[0])**2).contiguous()


    def make_param(self,x):
        return(torch.nn.parameter.Parameter(torch.tensor(x, dtype = torch.float,requires_grad = self.with_grad,device = self.device)))

    
    def _gen_zrange(self):
        """Generate the zrange from size_z and res_axial."""
        self.zrange = (torch.arange(self.size_z,device = self.device) - (self.size_z + 1) // 2) * self.res_axial


    def _gen_kr(self):
        """Generate coordinate system and other internal parameters."""
        kx = torch.fft.fftfreq(self.size_x,dtype = torch.float, device = self.device)
        ky = torch.fft.fftfreq(self.size_y,dtype = torch.float, device = self.device)
        self._kx = kx/self.res_lateral
        self._ky = ky/self.res_lateral
        kxx, kyy = torch.meshgrid(kx, ky,indexing = 'ij')
        """Convert from cartesian to polar coordinates."""
        self._phi = torch.atan2(kyy, kxx)
        kr = torch.sqrt(kyy**2 + kxx**2)

        self._kr = kr/self.res_lateral
        # kmag is the radius of the spherical shell of the OTF
        self._kmag = self.ni / self.wavelength
        # because the OTF only exists on a spherical shell we can calculate
        # a kz value for any pair of kx and ky values
        """Take the positive square root, negative values will be set to zero."""
        self._kz = torch.sqrt(torch.nn.functional.relu(self._kmag**2 - self._kr**2))


    def _gen_pupil(self):
        """Generate an ideal pupil."""
        kr = self._kr
        # define the diffraction limit
        # remember we"re working with _coherent_ data _not_ intensity,
        # so drop the factor of 2
        diff_limit = self.NA / self.wavelength
        # return a circle of intensity 1 over the ideal passband of the
        # objective make sure data is complex
        return (kr <= diff_limit).type(torch.complex64)


    def _calc_defocus(self):
        """Calculate the defocus to apply to the base pupil."""
        kz = self._kz
        return torch.exp(2 * np.pi * 1j * kz * (self.zrange.unsqueeze(1).unsqueeze(2)))


    def _gen_psf(self, pupil_base=None):
        """Generate the PSF.

        kwargs
        ------
        pupil_base : ndarray
            provided so that phase retrieval algorithms can hook into this
            method.

        NOTE: that the internal state is created with fftfreq, which creates
        _unshifted_ frequences
        """
        # generate internal coordinates
        self._gen_kr()
        self._gen_zrange() #Necessary to do it again at each iteration because we change res_axial 
        # generate the pupil
        if pupil_base is None:
            pupil_base = self._gen_pupil()
        else:
            assert pupil_base.ndim == 2, f"`pupil_base` is wrong shape: {pupil_base.shape}"
            # Maybe we should do ifftshift here so user doesn't have too
        # pull relevant internal state variables
        kr = self._kr
        phi = self._phi
        kmag = self._kmag
        # apply the defocus to the base_pupil
        pupil = pupil_base * self._calc_defocus()
        # calculate theta, this is possible because we know that the
        # OTF is only non-zero on a spherical shell
        theta = torch.arcsin((kr < kmag) * kr / kmag)
        # The authors claim that the following code is unecessary as the
        # sine condition is already taken into account in the definition
        # of the pupil, but I call bullshit
        if self.condition != "none":
            if self.condition == "sine":
                a = 1.0 / torch.sqrt(torch.cos(theta))
            elif self.condition == "herschel":
                a = 1.0 / torch.cos(theta)
            else:
                raise RuntimeError("You should never see this")
            pupil *= a
        # apply the vectorial corrections, if requested

        # if no correction we still need one more axis for the following
        # code to work generally
        pupils = pupil.unsqueeze(0)
        PSFa = torch.fft.fftshift(torch.fft.ifftn(pupils, dim=(2, 3)), dim=(2, 3))
        # save the PSF internally
        return PSFa


    def otfi(self,PSFi):
        """Intensity OTF, complex array."""
        otfi =  torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(PSFi)))
        return otfi


def _radial_zernike(r, n, m):
    """Radial part of the zernike polynomial.

    Formula from http://mathworld.wolfram.com/ZernikePolynomial.html
    """
    rad_zern = np.zeros_like(r)
    # zernike polynomials are only valid for r <= 1
    valid_points = r <= 1.0
    if m == 0 and n == 0:
        rad_zern[valid_points] = 1
        return rad_zern
    rprime = r[valid_points]
    # for the radial part m is always positive
    m = abs(m)
    # calculate the coefs
    coef1 = (n + m) // 2
    coef2 = (n - m) // 2
    jacobi = eval_jacobi(coef2, m, 0, 1 - 2 * rprime**2)
    rad_zern[valid_points] = (-1) ** coef2 * rprime**m * jacobi
    return rad_zern


def _ingest_index(j):
    """Convert inputs to arrays and do type and validity checking."""
    j = np.asarray(j)

    # check inputs
    if not (j > 0).all():
        raise ValueError("Invalid Noll index")
    if not np.issubdtype(j.dtype, np.signedinteger):
        raise ValueError("Index is not integer, input = {j}")
    return j


def noll2degrees(j: int) -> Tuple[int, int]:
    """Convert the Noll index number j to the integer pair (n, m) that defines the Zernike polynomial Z_n^m(ρ, θ).

    NOTE: inputs are vectorized and outputs will be ndarrays

    Source: (https://en.wikipedia.org/wiki/Zernike_polynomials)
    https://github.com/rdoelman/ZernikePolynomials.jl/blob/2825846679607f7bf335fdb9edd3b7145d65082b/src/ZernikePolynomials.jl
    """

    n = (np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2)).astype(int)
    jr = j - (n * (n + 1) / 2).astype(int)

    # if (n % 4) < 2:
    #     m1 = jr
    #     m2 = -(jr - 1)
    #     if (n - m1) % 2 == 0:
    #         m = m1
    #     else:
    #         m = m2
    # else:  # mod(n,4) ∈ (2,3)
    #     m1 = jr - 1
    #     m2 = -(jr)
    #     if (n - m1) % 2 == 0:
    #         m = m1
    #     else:
    #         m = m2

    # below is the vectorization version of the above.

    m1 = np.zeros_like(jr)
    m2 = np.zeros_like(jr)
    m = np.zeros_like(jr)

    n_mod_4 = n % 4

    idx0 = n_mod_4 < 2
    m1[idx0] = jr[idx0]
    m2[idx0] = -(jr[idx0] - 1)

    m1[~idx0] = jr[~idx0] - 1
    m2[~idx0] = -jr[~idx0]

    idx1 = (n - m1) % 2 == 0
    m[idx1] = m1[idx1]
    m[~idx1] = m2[~idx1]

    return n, m


def zernike(r: float, theta: float, n: int, m: int, norm: bool = True) -> float:
    """Calculate the Zernike polynomial on the unit disk for the requested orders.

    Parameters
    ----------
    r : ndarray
    theta : ndarray

    Args
    ----
    Noll : numeric or numeric sequence
        Noll's Indices to generate
    (n, m) : tuple of numerics or numeric sequences
        Radial and azimuthal degrees
    n : see above
    m : see above

    Kwargs
    ------
    norm : bool (default False)
        Do you want the output normed?

    Returns
    -------
    zernike : ndarray
        The zernike polynomials corresponding to Noll or (n, m) whichever are
        provided

    Example
    -------
    >>> x = np.linspace(-1, 1, 512)
    >>> xx, yy = np.meshgrid(x, x)
    >>> r, theta = cart2pol(yy, xx)
    >>> zern = zernike(r, theta, 4, 0)  # generates the defocus zernike polynomial
    """
    n, m = np.asarray(n), np.asarray(m)
    if n.ndim > 1:
        raise ValueError("Radial degree has the wrong shape")
    if m.ndim > 1:
        raise ValueError("Azimuthal degree has the wrong shape")
    if n.shape != m.shape:
        raise ValueError("Radial and Azimuthal degrees have different shapes")

    # make sure r and theta are arrays
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # make sure that r is always greater than 0
    if not (r >= 0).all():
        raise ValueError("r must always be greater or equal to 0")
    if r.ndim > 2:
        raise ValueError("Input rho and theta cannot have more than two dimensions")

    # make sure that n and m are iterable
    n, m = n.ravel(), m.ravel()

    # make sure that n is always greater or equal to m
    if not (n >= abs(m)).all():
        raise ValueError("n must always be greater or equal to m")

    # return column of zernike polynomials
    return np.array([_zernike(r, theta, nn, mm, norm) for nn, mm in zip(n, m)]).squeeze()


def _zernike(r, theta, n, m, norm=True):
    """Calculate the full zernike polynomial."""
    # remember if m is negative
    mneg = m < 0
    # going forward m is positive (Radial zernikes are only defined for
    # positive m)
    m = abs(m)
    # if m and n aren't seperated by multiple of two then return zeros
    if (n - m) % 2:
        return np.zeros_like(r)
    zern = _radial_zernike(r, n, m)
    if mneg:
        # odd zernike
        zern *= np.sin(m * theta)
    else:
        # even zernike
        zern *= np.cos(m * theta)

    # calculate the normalization factor
    if norm:
        # https://www.gatinel.com/en/recherche-formation/wavefront-sensing/zernike-polynomials/
        if m == 0:
            # m is zero
            norm = np.sqrt(n + 1)
        else:
            # m not zero
            norm = np.sqrt(2 * (n + 1))
        zern *= norm
    return zern


class Gaussian_psf(nn.Module):

    def __init__(self, box_shape,box_size,sigma=1, device='cpu',with_grad = True,dtype = torch.float):
        """Generate a PSF object.

        Parameters
        ----------
        sigma : float
            standard deviation of the PSF
        box_shape : int
            x/y/z size of the simulation
        box_size : float
            x/y/z size of the box
        """

        super().__init__()
        self.with_grad=with_grad
        self.dtype = dtype
        self.device = device
        x = torch.linspace(-box_size[0]/2,box_size[0]/2,box_shape[0],dtype = dtype,device = device)
        y = torch.linspace(-box_size[1]/2,box_size[1]/2,box_shape[1],dtype = dtype,device = device)
        z = torch.linspace(-box_size[2]/2,box_size[2]/2,box_shape[2],dtype = dtype,device = device)
        #print('x: ', x, x.shape)
        self.Gridxi0, self.Gridxi1, self.Gridxi2 = torch.meshgrid(x,y,z,indexing = 'ij')
        self.sigma = self.make_param(np.identity(3)*sigma)


    def forward(self):
        ems = torch.exp(-(self.Gridxi0*(self.sigma[0,0]*self.Gridxi0+self.sigma[0,1]*self.Gridxi1+self.sigma[0,2]*self.Gridxi2) 
                          + self.Gridxi1*(self.sigma[1,0]*self.Gridxi0+self.sigma[1,1]*self.Gridxi1+self.sigma[1,2]*self.Gridxi2)
                          + self.Gridxi2*(self.sigma[2,0]*self.Gridxi0+self.sigma[2,1]*self.Gridxi1+self.sigma[2,2]*self.Gridxi2))/2)
        
        return ems.contiguous()


    def make_param(self,x):
        return(torch.nn.parameter.Parameter(torch.tensor(x, dtype = self.dtype,requires_grad = self.with_grad,device = self.device)))


class Gaussian_psf_logsigma(nn.Module):

    def __init__(self, box_shape,box_size,sigma=1, device='cpu',with_grad = True,dtype = torch.float):
        """Generate a PSF object.

        Parameters
        ----------
        sigma : float
            standard deviation of the PSF
        box_shape : int
            x/y/z size of the simulation
        box_size : float
            x/y/z size of the box
        """

        super().__init__()
        self.with_grad=with_grad
        self.dtype = dtype
        self.device = device
        x = torch.linspace(-box_size[0]/2,box_size[0]/2,box_shape[0],device = device)
        y = torch.linspace(-box_size[1]/2,box_size[1]/2,box_shape[1],device = device)
        z = torch.linspace(-box_size[2]/2,box_size[2]/2,box_shape[2],device = device)
        self.Gridxi0, self.Gridxi1, self.Gridxi2 = torch.meshgrid(x,y,z,indexing = 'ij')
        self.logsigma = self.make_param(np.identity(3)*np.log(sigma))


    def forward(self):
        self.sigma = torch.exp(self.logsigma)
        ems = torch.exp(-(self.Gridxi0*(self.sigma[0,0]*self.Gridxi0+self.sigma[0,1]*self.Gridxi1+self.sigma[0,2]*self.Gridxi2) 
                          + self.Gridxi1*(self.sigma[1,0]*self.Gridxi0+self.sigma[1,1]*self.Gridxi1+self.sigma[1,2]*self.Gridxi2)
                          + self.Gridxi2*(self.sigma[2,0]*self.Gridxi0+self.sigma[2,1]*self.Gridxi1+self.sigma[2,2]*self.Gridxi2))/2)
        return ems.contiguous()


    def make_param(self,x):
        return(torch.nn.parameter.Parameter(torch.tensor(x, dtype = self.dtype,requires_grad = self.with_grad,device = self.device)))


class Anisotropic_gaussian_psf(nn.Module):

    def __init__(self, box_shape,box_size,sigma=1,sigma_z=1, eps=0.01, device='cpu',with_grad = True,dtype = torch.float):
        """Generate a PSF object.

        Parameters
        ----------
        sigma : float
            standard deviation of the PSF
        box_shape : int
            x/y/z size of the simulation
        box_size : float
            x/y/z size of the box
        """

        super().__init__()
        self.with_grad=with_grad
        self.dtype = dtype
        self.device = device
        x = torch.linspace(-box_size[0]/2,box_size[0]/2,box_shape[0])
        y = torch.linspace(-box_size[1]/2,box_size[1]/2,box_shape[1])
        z = torch.linspace(-box_size[2]/2,box_size[2]/2,box_shape[2])
        self.Gridxi0, self.Gridxi1, self.Gridxi2 = torch.meshgrid(x,y,z,indexing = 'ij')
        self.sigma = self.make_param(np.identity(2)*sigma)
        self.sigma_z = self.make_param(sigma_z)
        self.eps = self.make_param(eps)


    def forward(self):
        ems = (torch.exp(-( self.Gridxi0*(self.sigma[0,0]*self.Gridxi0+self.sigma[0,1]*self.Gridxi1)
                           +self.Gridxi1*(self.sigma[1,0]*self.Gridxi0+self.sigma[1,1]*self.Gridxi1))/(2*(self.eps+(torch.abs(self.Gridxi2)*self.sigma_z)**2)))
        / ((self.eps+(torch.abs(self.Gridxi2)*self.sigma_z)**2)))

        return ems.contiguous()


    def make_param(self,x):
        return(torch.nn.parameter.Parameter(torch.tensor(x, dtype = self.dtype,requires_grad = self.with_grad,device = self.device)))
