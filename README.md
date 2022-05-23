## LUDVM | LESP-modulated Unsteady Discrete Vortex Method

Author: Juan Manuel Catalan Gomez. PhD Candidate in Unsteady Aerodynamics and CFD.

### Brief description

Code to solve 2D unsteady airfoil flow problems: using unsteady thin-airfoil theory augmented with intermittent LEV model. Proposed by Kiran Ramesh and Ashok Gopalarathnam.

LESP stands for Leading-edge Suction Parameter: $\text{LESP} (t) = A_0(t)$, defined as $ A_0(t) = - \frac{1}{\pi} \int_0^\pi \frac{W(x,t)}{U} \text{d}\theta$
with $W(x,t)$ being the induced velocity normal to the aerofoil surface, computed from components of motion kinematics, $U$ being horizontal velocity of the airfoil, and $\theta$ being a variable of transformation related to the chordwise coordinate $x$. When reaching a critical value of the $\text{LESP}(t)$ such that $|\text{LESP}(t)| \geq \text{LESP}_{crit}$, which is an input of the code, the $\text{LESP}(t)$ is limited to such value and a Leading Edge Vortex (LEV) is shed at that time step.

Check out **README.pdf** to see the math symbols displayed.

The code is distributed in a python class called LUDVM, and its methods:

- airfoil
- motion_plunge | motion_sinusoidal
- induced_velocity
- airfoil_downwash
- time_loop
- compute_coefficients
- flowfield
- animation

Example of calling:

     self = LUDVM(t0=0, tf=20, dt=5e-2, chord=1, rho=1.225, Uinf=1, \
                  Npoints = 81, Ncoeffs=30, LESPcrit=0.2, Naca = '0012')

### Details:

​	   Publication providing details on the LDVM theory is:
​       Kiran Ramesh, Ashok Gopalarathnam, Kenneth Granlund, Michael V. Ol and
​       Jack R. Edwards, "Discrete-vortex method with novel shedding criterion
​       for unsteady aerofoil flows with intermittent leading-edge vortex
​       shedding," Journal of Fluid Mechanics, Volume 751, July 2014, pp
​       500-538.  DOI: http://dx.doi.org/10.1017/jfm.2014.297
​       Available from:
​       http://www.mae.ncsu.edu/apa/publications.html#j023

------

​       Publication on the large-angle unsteady thin airfoil theory is:
​       Ramesh, K., Gopalarathnam, A., Edwards, J. R., Ol, M. V., and
​       Granlund, K., "An unsteady airfoil theory applied to pitching
​       motions validated against experiment and computation,"
​       Theor. Comput. Fluid Dyn., January 2013, DOI
​       10.1007/s00162-012-0292-8.  Available from:
​       http://www.mae.ncsu.edu/apa/publications.html#j021  

------

​       Publication containing the details of the modified model:
​       A modified discrete-vortex method algorithm with shedding criterion
​       for aerodynamic coefficients prediction at high angle of attack
​       Thierry M. Faure, Laurent Dumas, Vincent Drouet, Olivier Montagnier.
​       Applied Mathematical Modelling, December 2018.

------

​       More details in Katz J. & Plotkin A. Low Speed Aerodynamics
​       Chapter 13, Section 13.8 -> Unsteady Motion of a Two-Dimensional
​       Thin Airfoil. The paper is based on this section, adding the
​       effect of the LESP for the Leading edge shedding, the Vatista's
​       vortex model and the placement methodology for the shed vortices
​       of Ansari et al. (2006) and Ansari, Zbikowski & Knowles (2006).

------

​       More detailed info on PhD thesis:
​       Kiran Ramesh. Theory and Low-Order Modeling of Unsteady Airfoil Flows

### External dependencies

Needs the package *airfoils* installed: https://pypi.org/project/airfoils/
Installation:

1. First choice (if you are not in conda): 

   ```pip install airfoils
   pip install airfoils
   ```

2. Second choice: if you are using anaconda, you need to install the *pip* package inside conda before. Thus do:

       conda install pip

   Now you need to use *pip* to install airfoils. Introduce the following:

   ```
   ~/tools/anaconda3/bin/pip install airfoils
   ```

   Where *~/tools/anaconda3/bin/pip* is the path to the *pip* package in your anaconda distribution.
