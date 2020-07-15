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
    #        LESP stands for Leading-edge suction parameter.
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
