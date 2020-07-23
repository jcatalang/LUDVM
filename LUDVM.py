import os, sys
import numpy as np
import matplotlib.pyplot as plt
import timeit

class LUDVM():
    '''
    ## -----------------------------------------------------------------------
    # \title LUDVM | LESP-modulated Unsteady Discrete Vortex Method
    # ------------------------------------------------------------------------
    # \author Juan Manuel      PhD Candidate in Unsteady Aerodynamics and CFD
    #         Catalan Gomez    Universidad Carlos III de Madrid
    #                          Bioengineering and Aerospace Engineering Dpt.
    #                          jcatalan@ing.uc3m.es (Contact)
    # ------------------------------------------------------------------------
    # \brief Code to solve 2D unsteady airfoil flow problems: using unsteady
    #        thin-airfoil theory augmented with intermittent LEV model.
    #        Proposed by Kiran Ramesh and Ashok Gopalarathnam.
    #
    #        LESP stands for Leading-Edge Suction Parameter.
    #
    #        The code is distributed in a python class called LUDVM, and its
    #        methods:
    #           - airfoil
    #           - motion_plunge | motion_sinusoidal
    #           - induced_velocity
    #           - airfoil_downwash
    #           - time_loop
    #           - compute_coefficients
    #           - flowfield
    #           - animation
    #
    #  Example of calling:
    #     self = LUDVM(t0=0, tf=20, dt=5e-2, chord=1, rho=1.225, Uinf=1, \
    #                   Npoints = 81, Ncoeffs=30, LESPcrit=0.2, Naca = '0012')
    # ------------------------------------------------------------------------
    # \date 22-06-2020 by J.M. Catalan \n
    #       Created from scratch
    # \date 13-07-2020 by J.M. Catalan \n
    #       Bug related to negative LESP modulation fixed
    #       Chord variability fixed
    #       Motion sinusoidal added
    #       Airfoil circulation re-defined
    #       Other minor changes
    # \date 16-07-2020 by J.M. Catalan \n
    #       Solving method proposed by Faure et. al. added
    #       Difference wrt. Ramesh: no need for iterating
    # \date 23-07-2020 by J.M. Catalan
    #       Propulsive efficiency computed
    # ------------------------------------------------------------------------
    # \details
    #    Publication providing details on the LDVM theory is:
    #    Kiran Ramesh, Ashok Gopalarathnam, Kenneth Granlund, Michael V. Ol and
    #    Jack R. Edwards, "Discrete-vortex method with novel shedding criterion
    #    for unsteady aerofoil flows with intermittent leading-edge vortex
    #    shedding," Journal of Fluid Mechanics, Volume 751, July 2014, pp
    #    500-538.  DOI: http://dx.doi.org/10.1017/jfm.2014.297
    #    Available from:
    #    http://www.mae.ncsu.edu/apa/publications.html#j023
    # ........................................................................
    #    Publication on the large-angle unsteady thin airfoil theory is:
    #    Ramesh, K., Gopalarathnam, A., Edwards, J. R., Ol, M. V., and
    #    Granlund, K., "An unsteady airfoil theory applied to pitching
    #    motions validated against experiment and computation,"
    #    Theor. Comput. Fluid Dyn., January 2013, DOI
    #    10.1007/s00162-012-0292-8.  Available from:
    #    http://www.mae.ncsu.edu/apa/publications.html#j021
    # ........................................................................
    #    Publication containing the details of the modified model:
    #    A modified discrete-vortex method algorithm with shedding criterion
    #    for aerodynamic coefficients prediction at high angle of attack
    #    Thierry M. Faure, Laurent Dumas, Vincent Drouet, Olivier Montagnier.
    #    Applied Mathematical Modelling, December 2018.
    # ........................................................................
    #    More details in Katz J. & Plotkin A. Low Speed Aerodynamics
    #    Chapter 13, Section 13.8 -> Unsteady Motion of a Two-Dimensional
    #    Thin Airfoil. The paper is based on this section, adding the
    #    effect of the LESP for the Leading edge shedding, the Vatista's
    #    vortex model and the placement methodology for the shed vortices
    #    of Ansari et al. (2006) and Ansari, Zbikowski & Knowles (2006).
    # ........................................................................
    #    More detailed info on PhD thesis:
    #    Kiran Ramesh. Theory and Low-Order Modeling of Unsteady Airfoil Flows
    #
    #  ----------------------------- IMPORTANT ----------------------------
    #   Needs the package 'airfoils' installed: https://pypi.org/project/airfoils/
    #   Installation:
    #     - 1st choice: pip install airfoils (if you are not in conda)
    #     - 2nd choice: if you are using anaconda, you need to install the pip
    #       package inside conda before. Thus, do: conda install pip.
    #       Now you need to use pip to install airfoils. Introduce the following:
    #            ~/tools/anaconda3/bin/pip install airfoils
    #       Where ~/tools/anaconda3/bin/pip is the path to the pip package
    #       in your PC.
    #  ---------------------------------------------------------------------

    '''
    def __init__(self, t0=0, tf=12, dt=1.5e-2, chord=1, rho=1.225, Uinf=1, \
                       Npoints=80, Ncoeffs=30, LESPcrit=0.2, Naca = '0012', \
                       foil_filename = None, G = 1, T = 2, alpha_m = 0, \
                       alpha_max = 10, k = 0.2*np.pi, phi = 90, h_max = 1, \
                       verbose = True, method = 'Faure'):
        self.t0 = t0               # Beggining of simulation
        self.tf = tf               # End of simulation
        self.dt = dt               # Time step

        self.chord    = chord      # Airfoil Chord [m]
        self.rho      = rho        # Air Density [kg/m^3]
        self.Uinf     = Uinf       # Freestream Velocity [m/s]l
        self.Npoints  = Npoints    # Number of airfoil nodes
        self.Ncoeffs  = Ncoeffs    # Number of coefficients in the Fourier expansion
        self.piv      = 1/3*chord # Pivot point for the pitching motion
        self.LESPcrit = LESPcrit   # Critical Leading Edge Suction Parameter (LESP)
        self.maxerror = 1e-10      # Maximum error of the Newton Iteration Method (only for Ramesh method)
        self.maxiter  = 50         # Maximum number of iterations in the Newton Iteration Method (only for Ramesh method)
        self.epsilon  = 1e-4       # For the numerical derivative in Newton method (only for Ramesh method)
        self.xgamma   = 0.25       # Gamma point location as % of panel, where the bound vortices are located
        self.method   = method     # Method for computing the Circulations: 'Ramesh' or 'Faure'

        self.t = np.arange(t0,tf+dt,dt)      # Time vector
        self.nt = len(self.t)                # Number of time steps

        self.verbose = verbose

        self.dt_star = dt*Uinf/chord          # dimensionless dt
        self.v_core  = 1.3*self.dt_star*chord # vortex-core radius (Vatista's model) 1.3*dtstar*chord
        # self.v_core = 1.3*0.015*chord

        self.ilev2 = 0          # for lev plotting purposes
        # self.int_gammax_old = 0 # for loads calculation

        self.start_time = timeit.default_timer()
        # Below, the methods are called
        if Naca is not None:
            self.airfoil(Naca = Naca)
        else:
            try:
             self.airfoil(Naca = None, foil_filename=foil_filename)
            except: print('Please introduce a valid foil_filename file')
        # self.motion_plunge(G = G, T = T, alpha_m = alpha_m, h0=0, x0=0)
        self.motion_sinusoidal(alpha_m = alpha_m, alpha_max = alpha_max, \
                              h_max = h_max, k = k, phi = phi, h0=0, x0=0)
        self.time_loop()
        self.compute_coefficients()
        elapsed_time=timeit.default_timer() - self.start_time
        print('Elapsed time:', elapsed_time)

        return None

    def airfoil(self, Naca = '0012', filename = None,  \
                      Npoints = None, uniform_spacing = 'theta'):
        from airfoils import Airfoil
        from airfoils.fileio  import import_airfoil_data
        from scipy.interpolate import interp1d

        if Npoints == None:
            Npoints = self.Npoints

        if Naca == None and filename is not None:
            try:
              # Reads file from UIUC database: .dat file
              # https://m-selig.ae.illinois.edu/ads/coord_database.html
              upper,lower = import_airfoil_data(filename)
              xupper1, yupper1 = upper[0,:], upper[1,:]
              xlower1, ylower1 = lower[0,:], lower[1,:]
              # Defining interpolants
              yupper_i = interp1(xupper1, yupper1, kind='cubic', bounds_error=False, \
                        fill_value="extrapolate");
              ylower_i = interp1(xlower1, ylower1, kind='cubic', bounds_error=False, \
                        fill_value="extrapolate");
              # Obtaining the airfoil with Npoints
              xupper, xlower = np.linspace(0,1,np.floor(Npoints/2)), np.linspace(0,1,np.floor(Npoints/2))
              yupper, ylower = yupper_i(xupper), ylower_i(xlower)
              xa   = self.chord*0.5*(xupper + xlower)
              etaa = self.chord*0.5*(yupper + ylower) #camberline
            except:
                print('Error reading file')
        elif Naca is not None: #builds the naca airfoil
            airfoil = Airfoil.NACA4(Naca,n_points = Npoints)

            airfoilpoints    = airfoil.all_points
            xupper, yupper   = airfoil._x_upper, airfoil._y_upper
            xlower, ylower   = airfoil._x_lower, airfoil._y_lower
            camberline_angle = airfoil.camber_line_angle(xupper)
            xa               = self.chord*0.5*(xupper + xlower)
            etaa             = self.chord*airfoil.camber_line(xupper) #camberline

        if uniform_spacing == 'theta':
            theta        = np.linspace(0,np.pi, self.Npoints) #uniform spacing in theta
            x            = self.chord/2*(1-np.cos(theta))
            eta          = np.interp(x, xa, etaa)
        else: #uniform spacing in x
            x, eta      = xa, etaa
            theta       = np.arccos(1-2*x/self.chord)

        x_panel     = x[:-1] + self.xgamma*(x[1:]-x[:-1]) # at gamma point (xgamma)
        eta_panel   = np.interp(x_panel, x, eta)
        theta_panel = np.arccos(1-2*x_panel/self.chord)

        # Computing derivatives wrt x/theta
        detadx, detadtheta = np.zeros(len(eta)), np.zeros(len(eta))
        detadx_panel, detadtheta_panel = np.zeros(len(eta_panel)), np.zeros(len(theta_panel))
        ip = 0
        for i in range(Npoints):
            if i == 0: # forward
                detadx[i] = (eta[i+1] - eta[i])/(x[i+1] - x[i])
                detadtheta[i] = (eta[i+1] - eta[i])/(theta[i+1] - theta[i])
            elif i == len(eta)-1: #backward
                detadx[i] = (eta[i] - eta[i-1]) / (x[i] - x[i-1])
                detadtheta[i] = (eta[i] - eta[i-1]) / (theta[i] - theta[i-1])
            else:
                detadx[i] = (eta[i+1] - eta[i-1]) / (2*(x[i+1] - x[i-1]))
                detadtheta[i] = (eta[i+1] - eta[i-1]) / (2*(theta[i+1] - theta[i-1]))
        for ip in range(Npoints-1):
            if ip == 0:    #forward
                detadx_panel[ip] = (eta_panel[ip+1] - eta_panel[ip])/(x_panel[ip+1] - x_panel[ip])
                detadtheta_panel[ip] = (eta_panel[ip+1] - eta_panel[ip])/(theta_panel[ip+1] - theta_panel[ip])
            elif ip == len(eta_panel)-1: #backward
                detadx_panel[ip] = (eta_panel[ip] - eta_panel[ip-1])/(x_panel[ip] - x_panel[ip-1])
                detadtheta_panel[ip] = (eta_panel[ip] - eta_panel[ip-1])/(theta_panel[ip] - theta_panel[ip-1])
            else:
                detadx_panel[ip] = (eta_panel[ip+1] - eta_panel[ip-1]) / (2*(x_panel[ip+1] - x_panel[ip-1]))
                detadtheta_panel[ip] = (eta_panel[ip+1] - eta_panel[ip-1]) / (2*(theta_panel[ip+1] - theta_panel[ip-1]))

        self.Npoints = len(eta)
        self.airfoil = {'x':x, 'theta':theta, 'eta':eta, \
                        'detadx':detadx, 'detadtheta':detadtheta, \
                        'x_panel':x_panel, 'theta_panel':theta_panel, \
                        'eta_panel':eta_panel, 'detadx_panel': detadx_panel, \
                        'detadtheta_panel':detadtheta_panel}
        return None

    def motion_sinusoidal(self, alpha_m = 0, alpha_max = 10, h_max = 1, \
                          k = 0.2*np.pi, phi = 90, h0=0, x0=0.25):
        # Definition of motion kinematics:
        # Heaving:            h(t)     = h0 + h_max*cos(2*pi*f*t)
        # Pitching:           alpha(t) = alpham + alpha_max*cos(2*pi*f*t + phi)
        # Horizontal flight:  x(t)     = x0 - Uinf*t
        #
        # Inputs: alpham (mean pitch in degrees), alpha_max (pitch amplitude in degrees)
        #         h_max (heaving amplitude),
        #         k (reduced frequency: ratio between convective time and period)
        #         phi (phase lag between heaving and pitching in degrees)
        #         x0, h0: initial position of the pivot point

        pi    = np.pi
        Uinf  = self.Uinf
        nt    = self.nt
        chord = self.chord

        f = k*Uinf/(2*pi*self.chord)
        self.f = f
        alpha_m, alpha_max, phi = alpha_m*pi/180, alpha_max*pi/180, phi*pi/180

        # Initialize arrays for the motion
        alpha, alpha_dot = np.zeros(nt), np.zeros(nt)
        h    , h_dot     = np.zeros(nt), np.zeros(nt)
        x    , x_dot     = np.zeros(nt), np.zeros(nt)

        # Defining motion of the pivot point
        for i in range(nt):
           ti = self.t[i]
           alpha[i]     =   alpha_m + alpha_max*np.cos(2*pi*f*ti + phi)
           alpha_dot[i] = - alpha_max*2*pi*f*np.sin(2*pi*f*ti + phi)
           h[i]         =   h0 + h_max*np.cos(2*pi*f*ti)
           h_dot[i]     = - h_max*2*pi*f*np.sin(2*pi*f*ti)
           x[i]         =   x0 - Uinf*ti
           x_dot[i]     = - Uinf

        xpiv, hpiv = x, h

        # Get motion of the entire airfoil as a function of time
        path_airfoil = np.zeros([self.nt, 2, self.Npoints]) # t,xy, Npoints
        for i in range(nt):
            ti = self.t[i]
            # First we compute the Leading Edge motion
            path_airfoil[i,0,0] = xpiv[i] - self.piv*np.cos(-alpha[i]) #xLE new
            path_airfoil[i,1,0] = hpiv[i] + self.piv*np.sin(-alpha[i]) #yLE new
            # The position of a new generic point Q results from rotating the
            # vector LEQ a clockwise (-) angle alpha, such that:
            # xQ_new = xLE_new + xQ*cos(-alpha) - yQ*sin(-alpha)
            # yQ_new = yLE_new + xQ*sin(-alpha) + yQ*cos(-alpha)
            path_airfoil[i,0,1:] =  path_airfoil[i,0,0] + np.cos(-alpha[i]) * self.airfoil['x'][1:] \
                                        - np.sin(-alpha[i]) * self.airfoil['eta'][1:]
            path_airfoil[i,1,1:] =  path_airfoil[i,1,0] + np.sin(-alpha[i]) * self.airfoil['x'][1:] \
                                        + np.cos(-alpha[i]) * self.airfoil['eta'][1:]

        # Gamma points are located at xgamma of each panel
        path_airfoil_gamma_points = path_airfoil[:,:,:-1] +  \
                        self.xgamma*(path_airfoil[:,:,1:]-path_airfoil[:,:,:-1])

        self.phi  , self.h_max      = phi, h_max
        self.alpha, self.alpha_dot = alpha, alpha_dot
        self.hpiv , self.h_dot     = hpiv , h_dot
        self.xpiv , self.x_dot     = xpiv , x_dot
        self.path                  = {'airfoil': path_airfoil,  \
                         'airfoil_gamma_points':path_airfoil_gamma_points}

        return None

    def motion_plunge(self, G = 1, T = 2, alpha_m = 0, h0=0, x0=0.25):
        # The plunge maneuver is defined by the following equation:
        # V(t) = -Vmax*sin^2(pi*t/T), where T is the maneuver duration and
        # Vmax is the peak plunge velocity, reached in the middle of the
        # maneuver, for t/T = 0.5. If we integrate V, we obtain the motion of
        # the airfoil in the vertical direction. The pitching is constant
        # in the plunge maneuver. Then, the whole motions is:
        # Plunging:           h(t)     = h0 - Vmax * t/2 + Vmax*T/(4*pi)*sin(2*pi*t/T)
        # Pitching:           alpha(t) = alpha_m
        # Horizontal flight:  x(t)     = x0 - Uinf*t
        #
        # Inputs: G=Vmax/Uinf (velocity ratio)
        #         T (maneuver duration)
        #         alpha_m (pitching for the simulation)
        #         x0, h0 (initial position of the pivot point)
        #
        # If the time of simulation is higher than the plunge maneuver duration
        # (tf > T), once completed the maneuver, the airfoil continues
        # in horizontal flight.

        pi    = np.pi
        Uinf  = self.Uinf
        nt    = self.nt
        chord = self.chord

        # Definition of motion kinematics
        # alpha_m = alpha_m  # Mean Pitch [degrees]
        alpha_m = alpha_m * pi / 180
        Vmax = G*Uinf
        T = T * chord/Uinf;
        h0 = h0   # initial position of 'pivot' point
        x0 = x0   # initial position of 'pivot' point

        self.G = G
        self.T = T

        # Initialize arrays for the motion
        alpha, alpha_dot = np.zeros(nt), np.zeros(nt)
        h    , h_dot     = np.zeros(nt), np.zeros(nt)
        x    , x_dot     = np.zeros(nt), np.zeros(nt)

        # Defining motion of the pivot point
        for i in range(nt):
           ti = self.t[i]
           if ti <= T:# plunge maneuver until T (duration of maneuver)
            alpha[i] = alpha_m
            alpha_dot[i] = 0
            h[i] = h0 - Vmax * ti/2 + Vmax * T/(4*pi) * np.sin(2*pi*ti/T)
            h_dot[i] = - Vmax*np.sin(pi*ti/T)**2
            x[i] = x0 - Uinf*ti
            x_dot[i] = - Uinf
           else: # from T to t_final -> horizontal flight (after plunge maneuver)
            alpha[i] = alpha_m
            alpha_dot[i] = 0
            h[i] = h[i-1]
            h_dot[i] = 0
            x[i] = x0 - Uinf*ti
            x_dot[i] = - Uinf

        xpiv, hpiv = x, h

        # Get motion of the entire airfoil as a function of time
        path_airfoil = np.zeros([self.nt, 2, self.Npoints]) # t,xy, Npoints
        for i in range(nt):
            ti = self.t[i]
            # First we compute the Leading Edge motion
            path_airfoil[i,0,0] = xpiv[i] - self.piv*np.cos(-alpha[i]) #xLE new
            path_airfoil[i,1,0] = hpiv[i] + self.piv*np.sin(-alpha[i]) #yLE new
            # The position of a new generic point Q results from rotating the
            # vector LEQ a clockwise (-) angle alpha, such that:
            # xQ_new = xLE_new + xQ*cos(-alpha) - yQ*sin(-alpha)
            # yQ_new = yLE_new + xQ*sin(-alpha) + yQ*cos(-alpha)
            path_airfoil[i,0,1:] =  path_airfoil[i,0,0] + np.cos(-alpha[i]) * self.airfoil['x'][1:] \
                                        - np.sin(-alpha[i]) * self.airfoil['eta'][1:]
            path_airfoil[i,1,1:] =  path_airfoil[i,1,0] + np.sin(-alpha[i]) * self.airfoil['x'][1:] \
                                        + np.cos(-alpha[i]) * self.airfoil['eta'][1:]

        # Gamma points are located at xgamma of each panel
        path_airfoil_gamma_points = path_airfoil[:,:,:-1] +  \
                        self.xgamma*(path_airfoil[:,:,1:]-path_airfoil[:,:,:-1])

        self.alpha, self.alpha_dot = alpha, alpha_dot
        self.hpiv , self.h_dot     = hpiv , h_dot
        self.xpiv , self.x_dot     = xpiv , x_dot
        self.path                  = {'airfoil': path_airfoil,  \
                         'airfoil_gamma_points':path_airfoil_gamma_points}
        return None

    def induced_velocity(self, circulation, xw, zw, xp, zp, viscous = True):
          # Calculates the induced velocity at points 'xp,yp', generated by
          # vortices located at 'xw,yw'. If 'viscous' is True, it uses the
          # Vatista's vortex model with core-radius v_core = 1.3*dt_star*chord.
          # If 'viscous' is not True, v_core = 0 and thus it uses point vortices.

          Np, Nw = len(xp), len(xw)
          x_dist = np.zeros([Np, Nw])
          z_dist = np.zeros([Np, Nw])
          for k in range(Np):
              x_dist[k,:] = xp[k] - xw
              z_dist[k,:] = zp[k] - zw

          if viscous == True: v_core = self.v_core
          else: v_core = 0

          Ku = z_dist/(2*np.pi*np.sqrt((x_dist**2 + z_dist**2)**2 + v_core**4))
          Kw = x_dist/(2*np.pi*np.sqrt((x_dist**2 + z_dist**2)**2 + v_core**4))

          u2, w2 = circulation*Ku, - circulation*Kw
          u , w  = np.sum(u2, axis=1), np.sum(w2, axis=1)
          return u, w

    def airfoil_downwash(self, circulation, xw, zw, i):
        # Computes induced velocity normal to the airfoil surface W(x,t).
        # i is the time-step index
        # If xw, zw are the wake vortices coordinates, computes the wake downwash
        # over the airfoil.

        alpha     = self.alpha[i]
        alpha_dot = self.alpha_dot[i]
        h_dot     = self.h_dot[i]

        xp, zp = self.path['airfoil_gamma_points'][i,0,:], self.path['airfoil_gamma_points'][i,1,:]

        u1, w1 = self.induced_velocity(circulation, xw, zw, xp, zp)

        # u1, w1 are in global coordinates, we need to rotate them to local
        u = u1*np.cos(alpha) - w1*np.sin(alpha)  # tangential to chord
        w = u1*np.sin(alpha) + w1*np.cos(alpha)  # normal to chord

        W = self.airfoil['detadx_panel']*(self.Uinf*np.cos(alpha) + h_dot*np.sin(alpha) + u \
                    - alpha_dot*self.airfoil['eta_panel']) \
                    - self.Uinf*np.sin(alpha) - alpha_dot*(self.airfoil['x_panel'] - self.piv) \
                    + h_dot*np.cos(alpha) - w

        return W

    def time_loop(self, print_dt = 50, BCcheck = False):

        pi = np.pi

        Uinf        = self.Uinf
        theta       = self.airfoil['theta']
        theta_panel = self.airfoil['theta_panel']
        LESPcrit    = self.LESPcrit
        epsilon     = self.epsilon
        chord       = self.chord
        rho         = self.rho
        dt          = self.dt

        # Initializing vortices coordinates and circulation
        nvort = self.nt-1
        # initializing paths of shed vortices
        # 1st index: time; 2nd index: x,y: 3rd index: Number of vortex
        self.path['TEV'] = np.zeros([self.nt, 2, nvort]) # At each dt, a TEV is shed
        self.path['LEV'] = np.zeros([self.nt, 2, nvort]) # There will be nt LEV shed as maximum

        # initializing circulations
        self.circulation = {'TEV': np.zeros([nvort])} #initializing dictionary
        self.circulation['LEV']     = np.zeros([nvort])
        self.circulation['bound']   = np.zeros([nvort])
        self.circulation['airfoil'] = np.zeros([nvort, self.Npoints-1])         # dGamma(x,t) = gamma(x,t)*dx
        self.BC                     = np.zeros([nvort, self.Npoints])           # Boundary condition computation (normal velocity to airfoil)
        self.circulation['gamma_airfoil'] = np.zeros([nvort, self.Npoints-1])   # gamma(x,t) = Fourier series
        self.circulation['Gamma_airfoil'] = np.zeros([nvort, self.Npoints-1])   # Gamma(x,t) = int_0^x dGamma(x,t)

        # initializing loads and pressure distribution
        self.dp = np.zeros([self.nt, self.Npoints-1])
        self.Fn = np.zeros([self.nt])
        self.Fs = np.zeros([self.nt])
        self.L  = np.zeros([self.nt])
        self.D  = np.zeros([self.nt])
        self.T  = np.zeros([self.nt])
        self.M  = np.zeros([self.nt])

        # Initializing fourier coefficients vector and LESP vector
        self.fourier = np.zeros([self.nt, 2, self.Ncoeffs]) # axis = 1 -> 0 coeffs, 1 derivatives
        self.LESP    = np.zeros(self.nt)
        self.LESP_prev = np.zeros(self.nt)

        itev = 0 #tev counter
        ilev = 0 #lev counter
        LEV_shed = -1*np.ones(self.nt) # stores the information of intermittent LEV shedding per dt

        '''----------------------------------------------------------'''
        '''----------------------- Time loop ------------------------'''
        '''----------------------------------------------------------'''
        for i in range(1,self.nt): #starting from 2nd step
            if (i == 1 or i == self.nt-1  or i/print_dt == int(i/print_dt)) and self.verbose == True:
                print('Step {} out of {}. Elapsed time {}'.format(i, self.nt-1, timeit.default_timer() - self.start_time))

            # Rewrite coordinates of the rest of vortices in the structure (not including vortices at time step i)
            self.path['TEV'][i,:,:itev]  = self.path['TEV'][i-1,:,:itev] # [:i] does not include i
            self.path['LEV'][i,:,:ilev]  = self.path['LEV'][i-1,:,:ilev]

            '''--------------------------------------------------------------'''
            '''---------------------- TEV computation -----------------------'''
            '''--------------------------------------------------------------'''
            # Compute the position of the shed TEV
            if itev == 0:
                # First TEV is located horizontally downstream at a distance 0.5*Uinf*dt from the trailing edge
                self.path['TEV'][i,:,itev] = self.path['airfoil'][0,:,-1]  + [0.5*Uinf*dt,0]
            else:
                # Shedding of the Trailing Edge Vortex (TEV)
                # (X,Z)_tev_i = (X,Z)_TE + 1/3[(X,Z)_tev_i-1 - (X,Z)_TE]
                # At 1/3 of the distance between the shedding edge and the
                # previously shed vortex (in this dt).
                self.path['TEV'][i,:,itev] = self.path['airfoil'][i,:,-1] + \
                   1/3*(self.path['TEV'][i,:,itev-1] - self.path['airfoil'][i,:,-1])

            if self.method == 'Ramesh': # iterating with Newton method
               f = 1 #initializing
               niter = 1
               shed_vortex_gamma = -1 # guess for Newton-Raphson
               while abs(f) > self.maxerror and niter < self.maxiter:
                   self.circulation['TEV'][itev] = shed_vortex_gamma
                   circulation = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
                   xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1])
                   zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1])
                   W = self.airfoil_downwash(circulation, xw, zw, i)
                   # Compute A0, A1 coefficients
                   A0 = - 1/pi * np.trapz(W/Uinf                    , theta_panel)
                   A1 =   2/pi * np.trapz(W/Uinf*np.cos(theta_panel), theta_panel)
                   # Get f for Newton-Raphson
                   circulation_bound = Uinf*chord*pi*(A0 + A1/2)
                   f = circulation_bound + sum(self.circulation['TEV'][:itev+1]) + sum(self.circulation['LEV'][:ilev+1])

                   # We set now gamma_TEV = gamma_TEV + epsilon
                   self.circulation['TEV'][itev] = shed_vortex_gamma + epsilon
                   # Get f + delta for Newton-Raphson: we need to compute again W, A0, A1
                   circulation   = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
                   # print(circulation)
                   xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1])
                   zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1])
                   W  = self.airfoil_downwash(circulation, xw, zw, i)
                   A0 = - 1/pi * np.trapz(W/Uinf                    , theta_panel)
                   A1 =   2/pi * np.trapz(W/Uinf*np.cos(theta_panel), theta_panel)
                   circulation_bound = Uinf*chord*pi*(A0 + A1/2)
                   fdelta = circulation_bound + sum(self.circulation['TEV'][:itev+1]) + sum(self.circulation['LEV'][:ilev+1])

                   # Newton-Raphson:
                   fprime = (fdelta - f) / epsilon # numerical df/dGamma
                   shed_vortex_gamma = shed_vortex_gamma - f / fprime # update solution with Newton

                   self.circulation['TEV'][itev] = shed_vortex_gamma # Restoring TEV circulation

                   if niter >= self.maxiter:
                       print('The solution did not converge during the Newton-Raphson iteration')
                       # break

                   niter = niter + 1

               # Solution after convergenge:
               circulation = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
               xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1])
               zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1])
               W  = self.airfoil_downwash(circulation, xw, zw, i)
               A0 = - 1/pi * np.trapz(W/Uinf                    , theta_panel)
               A1 =   2/pi * np.trapz(W/Uinf*np.cos(theta_panel), theta_panel)
               self.fourier[i,0,:2] = A0, A1
               self.circulation['bound'][itev] = Uinf*chord*pi*(A0 + A1/2)
               # Now we compute the rest of fourier coefficients (from A2 to An)
               for n in range(2,self.Ncoeffs):
                   self.fourier[i,0,n] = 2/pi * np.trapz(W/Uinf*np.cos(n*theta_panel), theta_panel)
               for n in range(self.Ncoeffs): # and their derivatives
                   self.fourier[i,1,n] = (self.fourier[i,0,n] - self.fourier[i-1,0,n])/dt

            elif self.method == 'Faure': # without iterating
                # Contribution of existing vortices
                circulation = np.append(self.circulation['TEV'][:itev], self.circulation['LEV'][:ilev])
                xw = np.append(self.path['TEV'][i,0,:itev], self.path['LEV'][i,0,:ilev])
                zw = np.append(self.path['TEV'][i,1,:itev], self.path['LEV'][i,1,:ilev])
                T1  = self.airfoil_downwash(circulation, xw, zw, i)

                # We compute the intensity of the shed TEV
                xa, za = self.path['airfoil_gamma_points'][i,0,:], self.path['airfoil_gamma_points'][i,1,:]
                xtevi, ztevi = self.path['TEV'][i,0,itev], self.path['TEV'][i,1,itev]
                utevi, wtevi = self.induced_velocity(np.array([1]), np.array([xtevi]), np.array([ztevi]), xa, za)
                ut = utevi*np.cos(self.alpha[i]) - wtevi*np.sin(self.alpha[i])  # tangential to chord
                un = utevi*np.sin(self.alpha[i]) + wtevi*np.cos(self.alpha[i])  # normal to chord
                T2 = self.airfoil['detadx_panel']*ut - un

                I1 = np.trapz(T1*(np.cos(theta_panel)-1), theta_panel)
                I2 = np.trapz(T2*(np.cos(theta_panel)-1), theta_panel)
                self.circulation['TEV'][itev] = - (I1 + sum(self.circulation['TEV'][:itev]) \
                    + sum(self.circulation['LEV'][:ilev]))/(1+I2)

                self.circulation['bound'][itev] = I1 + self.circulation['TEV'][itev]*I2

                J1 = - 1/np.pi*np.trapz(T1, theta_panel)
                J2 = - 1/np.pi*np.trapz(T2, theta_panel)

                W  = T1 + self.circulation['TEV'][itev]*T2
                # self.fourier[i,0,0] = J1 + self.circulation['TEV'][itev]*J2
                self.fourier[i,0,0] = - 1/pi * np.trapz(W/Uinf, theta_panel)
                for n in range(1,self.Ncoeffs):
                   self.fourier[i,0,n] = 2/pi * np.trapz(W/Uinf*np.cos(n*theta_panel), theta_panel)
                for n in range(self.Ncoeffs): # and their derivatives
                   self.fourier[i,1,n] = (self.fourier[i,0,n] - self.fourier[i-1,0,n])/dt

            self.LESP_prev[itev] = self.fourier[i,0,0] # LESP before being modulated (if it is the case)

            '''--------------------------------------------------------------'''
            '''-------------------- TEV, LEV computation --------------------'''
            '''--------------------------------------------------------------'''

            if abs(self.fourier[i,0,0]) >= abs(LESPcrit): # if A0 exceeds the LESPcrit: shedding occurs
                LEV_shed_gamma = self.circulation['TEV'][itev] # initial guess for Newton
                TEV_shed_gamma = self.circulation['TEV'][itev] # initial guess for Newton
                LEV_shed[i] = ilev    # indicator for knowing when shedding occurs
                # LEV_shed will be 'ilev' when shedding occurs and '-1' otherwise

                # Compute the position of the shed LEV
                if LEV_shed[i] == 0: # First LEV
                    #Shedding of the Leading Edge Vortex (TEV)
                    self.path['LEV'][i,:,ilev] = self.path['airfoil'][i,:,0]
                elif LEV_shed[i] > 0:
                    if LEV_shed[i-1] != -1: # if a lev has been shed previously
                        # Shedding of the Leading Edge Vortex (TEV)
                        # (X,Z)_lev_i = (X,Z)_LE + 1/3[(X,Z)_lev_i-1 - (X,Z)_LE]
                        # At 1/3 of the distance between the shedding edge and the
                        # previously shed vortex (in this dt).
                        self.path['LEV'][i,:,ilev] = self.path['airfoil'][i,:,0] + \
                        1/3*(self.path['LEV'][i,:,ilev-1] - self.path['airfoil'][i,:,0])
                    else: # not shed previously -> place it on the LE
                        self.path['LEV'][i,:,ilev] = self.path['airfoil'][i,:,0]

                if self.fourier[i,0,0] < 0: # if A0 < 0:
                    LESPcrit = -abs(LESPcrit)
                else:
                    LESPcrit =  abs(LESPcrit)

                if self.method == 'Ramesh':
                   f1, f2 = 0.1, 0.1 #initializing for the while
                   niter = 1
                   # Newton method for nonlinear systems
                   while (abs(f1) > self.maxerror or abs(f2) > self.maxerror) and \
                         niter < self.maxiter:

                         self.circulation['TEV'][itev] = TEV_shed_gamma #initial guess
                         self.circulation['LEV'][ilev] = LEV_shed_gamma #initial guess

                         circulation = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
                         xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1])
                         zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1])
                         W = self.airfoil_downwash(circulation, xw, zw, i)
                         # Compute A0, A1 coefficients
                         A0 = - 1/pi * np.trapz(W/Uinf                    , theta_panel)
                         A1 =   2/pi * np.trapz(W/Uinf*np.cos(theta_panel), theta_panel)
                         # Get f1 for Newton method
                         cbound = Uinf*chord*pi*(A0 + A1/2)
                         f1 = cbound + sum(self.circulation['TEV'][:itev+1]) + sum(self.circulation['LEV'][:ilev+1])
                         # Get f2 for Newton method
                         f2 = LESPcrit - A0

                         # Now we need to compute f1delta and f2delta
                         self.circulation['TEV'][itev] = TEV_shed_gamma + epsilon #initial guess
                         self.circulation['LEV'][ilev] = LEV_shed_gamma           #initial guess
                         circulation = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
                         xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1])
                         zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1])
                         W = self.airfoil_downwash(circulation, xw, zw, i)
                         # Compute A0, A1 coefficients
                         A0 = - 1/pi * np.trapz(W/Uinf                    , theta_panel)
                         A1 =   2/pi * np.trapz(W/Uinf*np.cos(theta_panel), theta_panel)
                         # Get f1 for Newton method
                         circulation_bound = Uinf*chord*pi*(A0 + A1/2)
                         f1_delta_TEV = circulation_bound + sum(self.circulation['TEV'][:itev+1]) + sum(self.circulation['LEV'][:ilev+1])
                         f2_delta_TEV = LESPcrit - A0

                         self.circulation['TEV'][itev] = TEV_shed_gamma             #initial guess
                         self.circulation['LEV'][ilev] = LEV_shed_gamma + epsilon   #initial guess
                         circulation = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
                         xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1])
                         zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1])
                         W = self.airfoil_downwash(circulation, xw, zw, i)
                         # Compute A0, A1 coefficients
                         A0 = - 1/pi * np.trapz(W/Uinf                    , theta_panel)
                         A1 =   2/pi * np.trapz(W/Uinf*np.cos(theta_panel), theta_panel)
                         # Get f1 for Newton method
                         circulation_bound = Uinf*chord*pi*(A0 + A1/2)
                         f1_delta_LEV = circulation_bound + sum(self.circulation['TEV'][:itev+1]) + sum(self.circulation['LEV'][:ilev+1])
                         f2_delta_LEV = LESPcrit - A0

                         # Build the Jacobian
                         # J = [J11 J12] = [df1/dGamma_LEV df1/dGamma_TEV]
                         #     [J21 J22]   [df2/dGamma_LEV df2/dGamma_TEV]
                         # J11 = df1/dGamma_LEV = (f1(Gamma_LEV+eps) - f1(Gamma_LEV))/(Gamma_LEV+eps - Gamma_LEV)
                         # J12 = df1/dGamma_TEV = (f1(Gamma_TEV+eps) - f1(Gamma_TEV))/(Gamma_TEV+eps - Gamma_TEV)
                         # J21 = df2/dGamma_LEV = (f2(Gamma_LEV+eps) - f2(Gamma_LEV))/(Gamma_LEV+eps - Gamma_LEV)
                         # J22 = df2/dGamma_TEV = (f2(Gamma_TEV+eps) - f2(Gamma_TEV))/(Gamma_TEV+eps - Gamma_TEV)
                         # Where all the denominators are equal to epsilon -> Gamma+eps-Gamma
                         # Newton for nonlinear systems:
                         # J*p_k = -f -> p_k = - J^-1 *f (solve a linear system at each iteration)
                         # p_k is the direction of search in the Newton method for nonlinear systems
                         # [Gamma_LEV, Gamma_TEV]_k+1 = [Gamma_LEV, Gamma_TEV]_k + pk

                         J11  = (f1_delta_LEV - f1) / epsilon
                         J12  = (f1_delta_TEV - f1) / epsilon
                         J21  = (f2_delta_LEV - f2) / epsilon
                         J22  = (f2_delta_TEV - f2) / epsilon
                         J    = np.array([[J11, J12], [J21, J22]])

                         pk = - np.linalg.solve(J, np.array([f1,f2])) #direction of search
                         shed_gamma = np.array([LEV_shed_gamma, TEV_shed_gamma]) + pk

                         LEV_shed_gamma = shed_gamma[0]
                         TEV_shed_gamma = shed_gamma[1]

                         self.circulation['TEV'][itev]    = TEV_shed_gamma
                         self.circulation['LEV'][ilev]    = LEV_shed_gamma
                         self.circulation['bound'][itev]  = cbound

                         if niter >= self.maxiter:
                             print('The solution did not converge when solving the LEV-TEV nonlinear system')
                             # break

                         niter = niter + 1

                   # Solution after convergence:
                   circulation = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
                   xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1])
                   zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1])
                   W = self.airfoil_downwash(circulation, xw, zw, i)
                   A0 = - 1/pi * np.trapz(W/Uinf                    , theta_panel)
                   A1 =   2/pi * np.trapz(W/Uinf*np.cos(theta_panel), theta_panel)
                   self.fourier[i,0,:2] = A0, A1
                   self.circulation['bound'][itev] = Uinf*chord*pi*(A0 + A1/2)

                   # Now we compute the rest of fourier coefficients (from A2 to An)
                   for n in range(2,self.Ncoeffs):
                       self.fourier[i,0,n] = 2/pi * np.trapz(W/Uinf*np.cos(n*theta_panel), theta_panel)

                   # Not updating the derivatives since A0(t) is no longer differentiable
                   # Use the derivatives of the coefficients before the TEV
                   # for n in range(self.Ncoeffs): # and their derivatives
                   #     self.fourier[i,1,n] = (self.fourier[i,0,n] - self.fourier[i-1,0,n])/dt

                elif self.method == 'Faure': # without iterating
                    # Contribution of existing vortices
                    circulation = np.append(self.circulation['TEV'][:itev], self.circulation['LEV'][:ilev])
                    xw = np.append(self.path['TEV'][i,0,:itev], self.path['LEV'][i,0,:ilev])
                    zw = np.append(self.path['TEV'][i,1,:itev], self.path['LEV'][i,1,:ilev])
                    T1  = self.airfoil_downwash(circulation, xw, zw, i)

                    # We compute the intensity of the shed TEV and LEV
                    xa, za = self.path['airfoil_gamma_points'][i,0,:], self.path['airfoil_gamma_points'][i,1,:]
                    xtevi, ztevi = self.path['TEV'][i,0,itev], self.path['TEV'][i,1,itev]
                    utevi, wtevi = self.induced_velocity(np.array([1]), np.array([xtevi]), np.array([ztevi]), xa, za)
                    ut_tevi = utevi*np.cos(self.alpha[i]) - wtevi*np.sin(self.alpha[i])  # tangential to chord
                    un_tevi = utevi*np.sin(self.alpha[i]) + wtevi*np.cos(self.alpha[i])  # normal to chord
                    T2 = self.airfoil['detadx_panel']*ut_tevi - un_tevi
                    xlevi, zlevi = self.path['LEV'][i,0,ilev], self.path['LEV'][i,1,ilev]
                    ulevi, wlevi = self.induced_velocity(np.array([1]), np.array([xlevi]), np.array([zlevi]), xa, za)
                    ut_levi = ulevi*np.cos(self.alpha[i]) - wlevi*np.sin(self.alpha[i])  # tangential to chord
                    un_levi = ulevi*np.sin(self.alpha[i]) + wlevi*np.cos(self.alpha[i])  # normal to chord
                    T3 = self.airfoil['detadx_panel']*ut_levi - un_levi

                    I1 = np.trapz(T1*(np.cos(theta_panel)-1), theta_panel)
                    I2 = np.trapz(T2*(np.cos(theta_panel)-1), theta_panel)
                    I3 = np.trapz(T3*(np.cos(theta_panel)-1), theta_panel)

                    J1 = - 1/np.pi*np.trapz(T1, theta_panel)
                    J2 = - 1/np.pi*np.trapz(T2, theta_panel)
                    J3 = - 1/np.pi*np.trapz(T3, theta_panel)

                    # Now we need to solve the linear system
                    A  = np.array([[1+I2, 1+I3], [J2, J3]])
                    b1 = - (I1 + sum(self.circulation['TEV'][:itev]) \
                        + sum(self.circulation['LEV'][:ilev]))
                    b2 = LESPcrit - J1
                    b  = np.array([b1, b2 ])
                    shed_gamma = np.linalg.solve(A, b)

                    self.circulation['TEV'][itev]    = shed_gamma[0]
                    self.circulation['LEV'][ilev]    = shed_gamma[1]

                    self.circulation['bound'][itev] = I1 + self.circulation['TEV'][itev]*I2 \
                          + self.circulation['LEV'][ilev]*I3
                    W  = T1 + self.circulation['TEV'][itev]*T2 + self.circulation['LEV'][ilev]*T3
                    self.fourier[i,0,0] = J1 + self.circulation['TEV'][itev]*J2 + self.circulation['LEV'][ilev]*J3
                    for n in range(1,self.Ncoeffs):
                       self.fourier[i,0,n] = 2/pi * np.trapz(W/Uinf*np.cos(n*theta_panel), theta_panel)

                    # Not updating the derivatives since A0(t) is no longer differentiable
                    # Use the derivatives of the coefficients before the LEV
                    # for n in range(self.Ncoeffs): # and their derivatives
                    #    self.fourier[i,1,n] = (self.fourier[i,0,n] - self.fourier[i-1,0,n])/dt

            else: # LEV shedding does not occur
                pass

            self.LESP[itev] = self.fourier[i,0,0]

            '''--------------------------------------------------------------'''
            '''-------------------- Airfoil circulation ---------------------'''
            '''--------------------------------------------------------------'''
            # We need compute the airfoil circulation per panel dGamma (located at xgamma of the panel)
            # Gamma_b(t) = int_0^c dGamma(t) = int_0^c gamma(x,t) dx
            # Gamma_b(t) = int_0^pi gamma(theta,t)*c/2*sin(theta)*dtheta
            # where gamma(theta,t) is the Fourier vortex distribution
            # We want to compute the integrand:
            # dGamma(t) = gamma(theta,t)*c/2*sin(theta)*dtheta
            # where c/2*sin(theta)*dtheta comes from differenciating x wrt theta
            # Then, np.sum(self.circulation['airfoil'][i,:]) should be equal to
            # self.circulation['bound'][i], so that the integral is fulfilled

            # Coefficients and derivatives
            A0, A0dot = self.fourier[i,:,0]
            A1, A1dot = self.fourier[i,:,1]
            A2, A2dot = self.fourier[i,:,2]
            A3, A3dot = self.fourier[i,:,3]

            Npanels      = self.Npoints - 1

            for j in range(Npanels):
                dtheta = theta[j+1] - theta[j]
                dxa    = self.airfoil['x'][j+1] - self.airfoil['x'][j]
                term2  = 0
                for n in range(1,self.Ncoeffs):
                    An    = self.fourier[i,0,n]
                    term2 = An*np.sin(n*theta_panel[j]) + term2

                term1  = A0*(1+np.cos(theta_panel[j]))/np.sin(theta_panel[j])
                gamma  = 2*Uinf*(term1 + term2)
                # dGamma = gamma*dxa
                dGamma = gamma*chord/2*np.sin(theta_panel[j])*dtheta
                self.circulation['airfoil'][itev,j]       = dGamma
                self.circulation['gamma_airfoil'][itev,j] = gamma

            for j in range(Npanels):
                self.circulation['Gamma_airfoil'][itev,j] = sum(self.circulation['airfoil'][itev,:j+1])

            # dxa    = self.airfoil['x'][1:] - self.airfoil['x'][:-1]
            # An = self.fourier[i,0,1:]
            # for j in range(Npanels):
            #     dtheta = theta[j+1] - theta[j]
            #     mat   = np.zeros(self.Ncoeffs-1)
            #     for n in range(1,self.Ncoeffs):
            #
            #         mat[n-1] = np.sin(n*theta[j+1])*np.sin(theta[j+1]) + \
            #                    np.sin(n*theta[j])*np.sin(theta[j])
            #
            #     dGamma = 0.5*chord*Uinf*(A0*(2+np.cos(theta[j+1])+np.cos(theta[j])) \
            #                                 + np.dot(An,mat))*dtheta
            #
            #     self.circulation['airfoil'][itev,j] = dGamma

            # self.circulation['gamma_airfoil'][itev,:] = self.circulation['airfoil'][itev,:]/dxa # gamma(x,t): fourier series
            # for j in range(Npanels):
            #     self.circulation['Gamma_airfoil'][itev,j] = sum(self.circulation['airfoil'][itev,:j+1])

            '''--------------------------------------------------------------'''
            '''----------- Compute pressure and aerodynamic loads -----------'''
            '''--------------------------------------------------------------'''

            alpha      = self.alpha[i]
            alpha_dot  = self.alpha_dot[i]
            h_dot      = self.h_dot[i]
            dGamma     = self.circulation['airfoil'][itev,:]         # dGamma(x,t)
            gamma      = self.circulation['gamma_airfoil'][itev,:]   #  gamma(x,t)
            Gamma      = self.circulation['Gamma_airfoil'][itev,:]   #  Gamma(x,t)
            if itev == 0:
                Gamma_old = np.zeros_like(Gamma)
            else:
                Gamma_old  = self.circulation['Gamma_airfoil'][itev-1,:] #  Gamma(x,t-1)
            dGamma_dt  = (Gamma - Gamma_old)/self.dt
            xa         = self.airfoil['x']
            x_gamma    = self.airfoil['x_panel']

            circulation = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
            xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1]) # x of wake vortices
            zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1]) # y of wake vortices
            xp, zp = self.path['airfoil_gamma_points'][i,0,:], self.path['airfoil_gamma_points'][i,1,:]
            xpoint, zpoint = np.array(xp), np.array(zp)
            u1, w1 = self.induced_velocity(circulation, xw, zw, xpoint, zpoint)
            # u1, w1 are in global coordinates, we need to rotate them to local
            u = u1*np.cos(alpha) - w1*np.sin(alpha)  # tangential to chord (u)
            w = u1*np.sin(alpha) + w1*np.cos(alpha)  # normal to chord (w)

            # Pressure distribution
            # term1           = (Uinf*np.cos(alpha) + h_dot*np.sin(alpha) + u)*gamma
            # int_gammax      = Gamma     # Gamma(x,t)
            # int_gammax_old  = Gamma_old # Gamma(x,t-1)
            # d_int_gammax_dt = (int_gammax - int_gammax_old)/self.dt  # d/dt (Gamma(x,t))
            # self.dp[i,:]    = rho*(term1 + d_int_gammax_dt) #something wrong when derivatives are not updated

            # Normal force on the airfoil (integral of dp along the chord)
            # self.Fn[i] = np.trapz(self.dp[i,:], x_gamma)
            # or using the Fourier coefficients and derivatives
            self.Fn[i] = rho*pi*chord*Uinf*((Uinf*np.cos(alpha) + h_dot*np.sin(alpha))*(A0 + 0.5*A1) \
                        + chord*(3/4*A0dot + 1/4*A1dot + 1/8*A2dot)) \
                        + rho*np.trapz(u*gamma, x_gamma)

            # Axial force due to leading edge suction
            self.Fs[i] = rho*pi*chord*Uinf**2*A0**2

            # Lift force
            self.L[i] = self.Fn[i]*np.cos(alpha) + self.Fs[i]*np.sin(alpha)
            # Drag force
            self.D[i] = self.Fn[i]*np.sin(alpha) - self.Fs[i]*np.cos(alpha)
            # Thurst force
            self.T[i] = -self.D[i]

            # Pitching moment
            xref = self.piv
            # self.M[i] = np.trapz(self.dp[i,:]*(xref-x_gamma), x_gamma)
            # or using the Fourier coefficients and derivatives (same result)
            self.M[i] = xref*self.Fn[i] - rho*pi*chord**2*Uinf*( \
                        (Uinf*np.cos(alpha) + h_dot*np.sin(alpha))*(1/4*A0 + 1/4*A1 - 1/8*A2) \
                    +   chord*(7/16*A0dot + 3/16*A1dot + 1/16*A2dot - 1/64*A3dot)) \
                    -   rho*np.trapz(u*gamma*x_gamma, x_gamma)

            '''--------------------------------------------------------------'''
            '''----------- Convection of vortices (wake roll-up) ------------'''
            '''--------------------------------------------------------------'''
            circulation_wake = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
            circulation_foil = self.circulation['airfoil'][itev,:]
            xw = np.append(self.path['TEV'][i,0,:itev+1], self.path['LEV'][i,0,:ilev+1]) # x of wake vortices
            zw = np.append(self.path['TEV'][i,1,:itev+1], self.path['LEV'][i,1,:ilev+1]) # y of wake vortices
            xa = self.path['airfoil_gamma_points'][i,0,:] # x of airfoil vortices
            za = self.path['airfoil_gamma_points'][i,1,:] # z of airfoil vortices

            # TEV convection
            xp = self.path['TEV'][i,0,:itev+1]
            zp = self.path['TEV'][i,1,:itev+1]
            u_wake, w_wake = self.induced_velocity(circulation_wake, xw, zw, xp, zp)
            u_foil, w_foil = self.induced_velocity(circulation_foil, xa, za, xp, zp, viscous = True)

            self.path['TEV'][i,0,:itev+1] = self.path['TEV'][i,0,:itev+1] + dt*(u_wake + u_foil)
            self.path['TEV'][i,1,:itev+1] = self.path['TEV'][i,1,:itev+1] + dt*(w_wake + w_foil)

            # LEV convection
            xp = self.path['LEV'][i,0,:ilev+1]
            zp = self.path['LEV'][i,1,:ilev+1]
            u_wake, w_wake = self.induced_velocity(circulation_wake, xw, zw, xp, zp)
            u_foil, w_foil = self.induced_velocity(circulation_foil, xa, za, xp, zp, viscous = True)

            self.path['LEV'][i,0,:ilev+1] = self.path['LEV'][i,0,:ilev+1] + dt*(u_wake + u_foil)
            self.path['LEV'][i,1,:ilev+1] = self.path['LEV'][i,1,:ilev+1] + dt*(w_wake + w_foil)

            self.ilev     = ilev
            self.itev     = itev
            self.LEV_shed = LEV_shed

            '''--------------------------------------------------------------'''
            '''------------------ Boundary condition check ------------------'''
            '''--------------------------------------------------------------'''
            # Boundary condition is: (grad(phi) - V0 - w x r)·n = 0
            # where where phi is the velocity potential, V0 is the velocity of the
            # body frame with respect to the inertial frame expressed in body
            # coordinates, w is the rate of rotation of the body frame, r is the
            # position vector of a point in the body frame about the pivot point
            # and n is a unit vector which is normal to the camberline in the
            # body frame. The velocity potential is comprised of components from
            # bound circulation and wake circulation, phi_B and phi_W respectively.
            if BCcheck == True:
               xa, za = self.path['airfoil_gamma_points'][i,0,:], self.path['airfoil_gamma_points'][i,1,:]
               u1, w1 = self.induced_velocity(circulation_wake, xw, zw, xa, za)
               # u1, w1 are in global coordinates, we need to rotate them to local
               u = u1*np.cos(alpha) - w1*np.sin(alpha)  # tangential to chord (u)
               w = u1*np.sin(alpha) + w1*np.cos(alpha)  # normal to chord (w)
               W = self.airfoil['detadx_panel']*(self.Uinf*np.cos(alpha) + h_dot*np.sin(alpha) + u \
                       - alpha_dot*self.airfoil['eta_panel']) \
                       - self.Uinf*np.sin(alpha) - alpha_dot*(self.airfoil['x'] - self.piv) \
                       + h_dot*np.cos(alpha) - w


               BCnx = self.airfoil['detadx_panel']*(- u - self.Uinf*np.cos(alpha)  \
                       - h_dot*np.sin(alpha) + alpha_dot*self.airfoil['eta_panel'])
               BCnz = W + w + self.Uinf*np.sin(alpha) - h_dot*np.cos(alpha) \
                       + alpha_dot*(self.airfoil['x_panel'] - self.piv)

               self.BC[itev,:] = BCnx + BCnz

            '''--------------------------------------------------------------'''
            '''-------------------- Vortex indices update -------------------'''
            '''--------------------------------------------------------------'''
            if LEV_shed[i] != -1: # if a lev has been shed in this dt: increase ilev
                ilev = ilev + 1

            itev = itev + 1 # increase tev after each dt

        return None

    def compute_coefficients(self):

        q = 0.5*self.rho*self.Uinf**2

        qc = q*self.chord

        self.Cp = self.dp/q
        self.Cn, self.Cs = self.Fn/qc, self.Fs/qc
        self.Cl, self.Cd, self.Ct = self.L/qc , self.D/qc, self.T/qc
        self.Cm = self.M/(qc*self.chord)

        return None

    def flowfield(self, xmin, xmax, zmin, zmax, dr):

        x1, z1 = np.arange(xmin, xmax, dr), np.arange(zmin, zmax, dr)
        x , z  = np.meshgrid(x1, z1, indexing = 'ij')
        xp, zp = np.ravel(x), np.ravel(z)
        u      = np.zeros([self.nt, np.shape(x)[0], np.shape(x)[1]])
        w      = np.zeros([self.nt, np.shape(x)[0], np.shape(x)[1]])

        for itev in range(self.nt-1):
            print('Flowfield i =', itev)
            ilev = int(self.LEV_shed[itev])
            circulation_wake = np.append(self.circulation['TEV'][:itev+1], self.circulation['LEV'][:ilev+1])
            circulation_foil = self.circulation['airfoil'][itev,:]
            xw = np.append(self.path['TEV'][itev+1,0,:itev+1], self.path['LEV'][itev+1,0,:ilev+1]) # x of wake vortices
            zw = np.append(self.path['TEV'][itev+1,1,:itev+1], self.path['LEV'][itev+1,1,:ilev+1]) # y of wake vortices
            xa = self.path['airfoil_gamma_points'][itev+1,0,:] # x of airfoil vortices
            za = self.path['airfoil_gamma_points'][itev+1,1,:] # z of airfoil vortices

            u_wake, w_wake = self.induced_velocity(circulation_wake, xw, zw, xp, zp)
            u_foil, w_foil = self.induced_velocity(circulation_foil, xa, za, xp, zp)

            u[i,:,:] = (u_wake + u_foil).reshape(np.shape(x))
            w[i,:,:] = (w_wake + w_foil).reshape(np.shape(x))

        self.xff, self.zff = x, z
        self.uff, self.wff = u, w

        return None


    def animation(self, step=1, ani_interval=10):
        from matplotlib.animation import FuncAnimation

        # Animation
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln,     = plt.plot([], [], 'k.', animated=True, markersize=2)
        ln_tev, = plt.plot([], [], 'r*', markersize=1, animated=True)
        ln_lev, = plt.plot([], [], 'b*', markersize=1, animated=True)
        tev_indices = np.arange(0,self.nt-1,step)
        # lev_time_indices = np.arange(0,self.ilev,1)

        # xmin, xmax = -6, 2
        # ymin, ymax = -6, 2
        xmin, xmax = 1.1*np.min(self.path['airfoil'][:,0,:]),1.1*np.max(self.path['airfoil'][:,0,:])
        ymin, ymax = -5*abs(np.min(self.path['airfoil'][:,1,:])),5*abs(np.max(self.path['airfoil'][:,1,:]))

        def init():
           ax.set_xlim(xmin, xmax)
           ax.set_ylim(ymin, ymax)
           ln.set_data(xdata,ydata)
           ln_tev.set_data(xdata,ydata)
           ln_lev.set_data(xdata,ydata)
           return ln,

        def update(i):
           # xdata.append(airfoil_path[frame,0,:]) #accumulative path
           # ydata.append(airfoil_path[frame,1,:]) #accumulative path

           # Airfoil motion
           ln.set_data(self.path['airfoil_gamma_points'][i+1,0,:], self.path['airfoil_gamma_points'][i+1,1,:])
           # TEV motion
           ln_tev.set_data(self.path['TEV'][i+1,0,:i+1],self.path['TEV'][i+1,1,:i+1])
           # LEV motion
           if self.LEV_shed[i] != -1:
               self.ilev2 = int(self.LEV_shed[i]) #ilev at that step
           if self.ilev2 > 0:
               ln_lev.set_data(self.path['LEV'][i+1,0,:self.ilev2+1],self.path['LEV'][i+1,1,:self.ilev2+1])
           return ln, ln_tev, ln_lev,

        ani = FuncAnimation(fig, func=update, frames=tev_indices,
                    init_func=init, blit=True, interval = ani_interval,repeat=False)
        plt.show()

        return None

    def propulsive_efficiency(self, T=None):
        # Computes the propulsive efficiency per period
        # Makes sense for pitching and heaving motion
        if T == None: T = 1/self.f

        tt = self.t/T #nondim time

        Nt  = int(np.floor(tt[-1])) # Number of full periods
        Ctm = np.zeros([Nt])
        Cpm = np.zeros([Nt])

        for ii in range(Nt):       
            indt = np.where(np.logical_and(tt >= ii-1, tt < ii)) #indices per period    
            Ctm[ii] = np.mean(self.Ct[indt])
            Cpi = abs(self.h_dot[indt]/self.Uinf * self.Cl[indt]) + abs(self.alpha_dot[indt]*self.Cm[indt] * self.chord/Uinf)
            Cpm[ii] = np.mean(Cpi)

        self.tt = tt
        self.etap = Ctm/Cpm        
        return None

# Cluster vortices by proximity into a single vortex to reduce the scalation
# of computation time as the number of vortices increase (Faure 2019)

# Implement tandem airfoils (Faure 2020): almost exactly the same as for 1
# See: Numerical study of two-airfoil arrangements by a discrete vortex method

if __name__ == "__main__":

    # Optmimum pitching case
    rho = 1.225
    chord = 1
    Uinf = 1
    k = 0.7798 #Reduced frequency k = 2*pi*c*f/U
    f = k*Uinf/(2*np.pi*chord)
    T = 1/f
    hmax = 1.4819*chord
    phi = 77.5885
    alpha_max = 46.2737

    self = LUDVM(t0=0, tf=5*T, dt=3.5e-2, chord=chord, rho=rho, Uinf=Uinf, \
         Npoints=80, Ncoeffs=30, LESPcrit=0.2, Naca = '0030', \
         alpha_m = 0, alpha_max = alpha_max, k = k, phi = phi, h_max = hmax,
         verbose = True, method = 'Faure')

    self.propulsive_efficiency()


    # self.animation(ani_interval=20)

    # # LESP with and without cutting
    # plt.figure(1)
    # plt.plot(self.t, self.LESP_prev)
    # plt.plot(self.t, self.LESP)
    # #
    # # Bound circulation check: should be the integral of airfoil dGammas
    # plt.figure(2)
    # plt.plot(self.circulation['bound'])
    # plt.plot(np.sum(self.circulation['airfoil'], axis=1), '.', markersize = 8)


    # Flow field - time evolution
    # xmin, xmax = -8, 2
    # zmin, zmax = -8, 2
    # dr = 0.1
    # self.flowfield(xmin, xmax, zmin, zmax, dr)
    # i = 350
    # plt.figure()
    # plt.plot(self.path['airfoil'][i,0,:], self.path['airfoil'][i,1,:], 'k.', markersize=2)
    # plt.contourf(self.xff, self.zff, self.uff[i,:,:], cmap = 'Spectral')
    # plt.colorbar()


    plt.show()
