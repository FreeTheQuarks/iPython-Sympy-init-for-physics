#~ This file sets up iPython for symbollic manipulation.
from sympy.abc import *  # Imports all letters as symbols.
from sympy import (init_printing, python, latex, symbols, Symbol, lambdify,             # Sympy commands
                   simplify, expand, factor, Eq, Function, Matrix, Piecewise,           # Algebra
                   Sum, Limit, Derivative, Integral, summation, limit, diff, integrate, series, solve, dsolve,                          # Calculus
                   oo, I as i, pi, sqrt, factorial, binomial, exp, ln, gamma, zeta,                                                     # Commonly used constants and functions.
                   sin, cos, tan, sec, csc, cot, asin as arcsin, acos as arccos, atan as arctan,                                        # Trig
                   And, Or, Xor, Not, Nor, Implies, Equivalent, Unequality, to_cnf, to_dnf, satisfiable, cartes, ordered, TableForm)    # Logic
from sympy.solvers import nsolve
from sympy.physics.units import (meter, second, kilogram, newton, pascal, joule, watt,          # Classical
                                 ampere, coulomb, volt, ohm, farad, weber, henry, eV, tesla,    # E&M
                                 candela, hertz, kelvin, mole, degrees, atmosphere)             # Misc

angstrom         = meter*10**-10                # Extra units.
atomic_mass_unit = 1.660538921*10**-27*kilogram

# These three are for the graphing function, but can still called directly from iPython.
from matplotlib import pyplot
from numpy import arange, linspace
from math import floor, ceil

e          = exp(1)
C1, C2, C3 = symbols("C1 C2 C3")
hbar       = Symbol("hbar", real=True, positive=True)

# Physical Constants
# Decided to make these numerical rather than symbollic, for convenience.

# Universal
c0       = 299792458 * (meter/second)                                   # Speed of light
G        = 6.67384*10**-11 * (newton*meter**2 / (kilogram**2))          # Newton's gravitational constant
h_planck = 6.62606957*10**-34 * (joule*second)                          # Planck's constant
hBar     = h_planck/(2*pi)                                              # Reduced Planck Constant
# E&M
q_e       = 1.602176565*10**-19 * coulomb                               # Fundamental charge
mu_0      = 4*pi*10**-7 * (henry/meter)                                 # Vacuum permeability
epsilon_0 = 1/(c0**2*mu_0)                                              # Vacuum permittivity
k_Coulomb = 1/(4*pi*epsilon_0)                                          # Coulomb's constant
mElectron = 9.10938291*10**-31 * kilogram                               # Electron mass
mProton   = 1.672621777*10**-27 * kilogram                              # Proton mass
mNeutron  = 1.674927351*10**-27 * kilogram                              # Neutron mass
mu_Bohr   = q_e*hBar/(2*mElectron)                                      # Bohr magneton
nuclear_magneton = q_e*hBar/2/mProton                                   # Nuclear magneton mu_k
# Quantum
pauli_x = Matrix(([0, 1], [1, 0]))
pauli_y = Matrix(([0, -i], [i, 0]))
pauli_z = Matrix(([1, 0], [0, -1]))
pauli   = Matrix((pauli_x, pauli_y, pauli_z)).transpose()
# Thermodynamics
n_Avagadro      = 6.02214129*10**23 / mole
R_gas_constant  = 8.3144621 * joule/(mole*kelvin)
k_Boltzmann     = R_gas_constant/n_Avagadro
StefanBoltzmann = pi**2 * k_Boltzmann**4 / (60*hBar**3 * c**2)


# Misc functions.
def Ackerman(m, n):
    """Ackerman function; version used in Rosen book."""
    if m == 0:
        return 2*n
    elif m >= 1:
        if n == 0:
            return 0
        elif n == 1:
            return 2
        elif n >= 2:
            return Ackerman(m-1, Ackerman(m, n-1))


def choose(n, k):
    """Computes the binomial coefficients nCk"""
    return factorial(n) / factorial(k) / factorial(n-k)


def erf(z):
    """The Gauss error function."""
    return simplify(2/sqrt(pi) * integrate(exp(-t**2), (t, 0, z))).evalf()


def Find_Coordinate_System(func):
    """Attempts to find what coordinate system a function is using.
    Returns 'c' for cartesian and 's' for spherical."""
    coords = ''
    try:
        for element in func:
            if (x in element) or (y in element) or (z in element):
                coords = 'c'
                break
            elif (r in element) or (theta in element) or (phi in element):
                coords = 's'
                break
                # If the function is not vector valued, then the above code won't work. Hence the TypeError.
    except TypeError:
        if (x in func) or (y in func) or (z in func):
            coords = 'c'
        elif (r in func) or (theta in func) or (phi in func):
            coords = 's'

    if coords == '':
        # Will throw an error if it cannot figure out what coordinate system an equation is supposed to be using.
        raise UserWarning("Coordinate system not specified.")
    else:
        return coords


def GramSchmidt(vectorSet):
    """Given a set of vectors, returns an orthonormal set generated by the Gram-Schmidt process."""

    orthonormal = vectorSet

    for n in range(len(vectorSet)):
        orthonormal[n] = vectorSet[n]
        for m in range(n):
            orthonormal[n] -= Projection(orthonormal[m], vectorSet[n])
            orthonormal[n] = Normalize(orthonormal[n])

    return orthonormal


def graph(func, var, domain, differentiate=False, integrate=False, scale=True):
    """Graphs a function using matplotlib.
    Can be either a Sympy funtion or a normal Python function, but it must only use one variable.
    differentiate:  graph func's derivative
    integrate:      graph func's anti-derivative
    scale:          use same scaling for X and Y axes.
        """
    pyplot.show()  # Remove old graph.

    try:
        newFunc = lambdify(var, func)  # Turns Sympy expression into a lambda function.
    except SyntaxError:
        newFunc = func

        # Speed is importance for this next line since the domain can be large and func can be complex,
        codomain = [newFunc(x) for x in domain]  # but don't think there is any faster way to do this than this line right here.

    if differentiate:
        try:
            funcDiff = lambdify(var, func.diff(var))
            diffDomain = [funcDiff(x) for x in domain]
            pyplot.plot(domain, diffDomain, linewidth=1.0, color='red')
        except AttributeError:
            print("Could not differentiate: not SymPy expression.")

    if integrate:
        try:
            funcInt = lambdify(var, func.integrate(var, conds='none'))
            intDomain = [funcInt(x) for x in domain]
            pyplot.plot(domain, intDomain, linewidth=1.0, color='green')
        except AttributeError:
            print("Could not integrate: not SymPy expression.")
        except _CoeffExpValueError:
            print("Could not integrate: integral too difficult.")

    pyplot.plot(domain, codomain, linewidth=1.0, color='blue')
    pyplot.xlabel(str(var))
    pyplot.ylabel('y')
    pyplot.title(str(func))

    #~ This is just to make the graph have the same size x and y axes.
    if scale:
        smallD = min(domain)
        bigD = max(domain)
        smallCod = min(codomain)
        bigCod = max(codomain)
        minima = floor(min([smallD, smallCod]))
        maxima = ceil(max([bigD, bigCod]))
        pyplot.axis([minima, maxima, minima, maxima])

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())  # Maximizes the graph window.

    pyplot.show(block=True)  # Block stops iPython from being used while graph is up.


def Hermite(order, var=xi, physicists=True):
    """Returns the nth-order physicist's Hermite polynomial for a variable var.
    If physicists=False then it will calculate the probabilist's Hermites."""

    if physicists:
        return expand(simplify((-1)**order * exp(var**2) * diff(exp(-var**2), var, order)))
    else:
        return expand(simplify((-1)**order * exp(var**2 / 2) * diff(exp(-var**2 / 2), var, order)))


def Kronecker_Delta(i, j):
    """1 if i == j, 0 otherwise."""

    return int(i == j)


def MaxwellStressTensor(E, B):
    """Calculates the Maxwell Stress Tensor for a given E and B fields.
    Must input E and B as matrices."""

    T = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

    for k in range(0, 3):
        for j in range(0, 3):
            T[k][j] = e0*(E[k]*E[j] - Kronecker_Delta(k, j)*E.dot(E)/2) + (B[k]*B[j] - Kronecker_Delta(k, j)*B.dot(B)/2)/mu0

    return simplify(Matrix((T[0], T[1], T[2])))


def Lagrange_Equation(lagrangian, coordinate):
    """Returns a symbolic Lagrange equation."""
    dcoordinate_dt = coordinate.diff(t)
    return simplify(Eq(lagrangian.diff(coordinate), lagrangian.diff(dcoordinate_dt).diff(t)))


def Laguerre_Polynomial(n, var=x):
    """returns the nth order Laguerre Polynomial.
    Calculated with the recurrence relation."""

    if n == 0:
        return 1
    elif n == 1:
        return 1 - var
    else:
        k = 1
        Lprev = 1 - var
        Lprevprev = 1

        while k < n:
            L = ((2*k+1-var)*Lprev - k*Lprevprev)/(k+1)
            Lprevprev = Lprev
            Lprev = L
            k += 1

        return factor(L)


def LeviCivita(i, j, k):
    """The Levi-Civita symbol.
    Returns 0 if two arguments are equal.
    Returns 1 if arguments cycle "clockwise".
    Returns -1 if arguments cycle "anti-clockwise"."""

    if i == j or j == k or k == i:
        return 0
    elif (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
        return 1
    elif (i, j, k) in [(1, 3, 2), (3, 2, 1), (2, 1, 3)]:
        return -1
    else:
        raise UserWarning("Invalid arguments given: " + str((i, j, k)))


def Small_Angle_Approximation(angle=theta, replacement=theta):
    """Returns a list populated with the three small angle approximations.
    Accurate to second order.
    To be used in conjunction with the .subs() property of Sympy expressions,
        i.e. expression.selfubs(Small_Angle_approximation(theta))"""

    return [(sin(angle), replacement),
            (cos(angle), 1-replacement**2 / 2),
            (tan(angle), replacement)]


def Taylor(equation, var, a, order):
    """TODO: fix this!
    Supposed to return nth order Taylor expansion around point a.
    Not sure what is wrong here."""

    try:
        func = equation.rhs  # Allows Eq() format to be used.
    except AttributeError:
        func = equation

    toReturn = 0
    for number in range(0, order + 1):
        temp = func.diff(var, number)
        toReturn += (temp.subs(var, a)/factorial(number))*(var-a)**number

    return simplify(toReturn)


def truth_table(eq):
    """Prints the truth table for a proposition."""
    #~ FIX ME: Sloppy code, no idea what is going on here.

    free = list(ordered(eq.free_symbols))
    rows = [free+[str(eq)]]

    for n in cartes(*[[True, False]]*len(free)):
        rows.append(list(n))
        rows[-1].append(eq.subs(zip(free, n)))
        rows[-1] = ["T" if n else "F" for n in rows[-1]]

    return TableForm(rows, wipe_zeros=False)


#~  Vector Calculus
# All vector derivatives should work in both Cartesian and spherical coordinate systems. (Maybe add cylindrical if you're bored one day.)
# There is also a funcion for calculating a triple integral in spherical coordinates that defaults to integrating over all of space,
# since this is a very common integral to do in physics this seemed like a reasonable default.

def CrossProduct(a, b, magnitude=False):
    """Returns the cross product of two vectors, expressed as a Sympy matrix.
    Vectors must be defined as iterables.
    If magnitude=True, will return the length of the vector."""

    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]
    if magnitude:
        return simplify(sqrt(x*x + y*y + z*z))  # Assuming Euclidean metric.
    else:
        return simplify(Matrix([(x, y, z)]))


def Curl(func, coords):
    """Returns the curl of a  vector function.
    Needs to be checked for correctness."""
    
    if coords == 'c':
        return simplify(Matrix([(func[2].diff(y) - func[1].diff(z)),
                                (func[0].diff(z) - func[2].diff(x)),
                                (func[1].diff(x) - func[0].diff(y))]))
    elif coords == 's':
        return simplify(Matrix([(diff(func[2]*r*sin(theta), theta) - diff(func[1]*r, phi))/r/r/sin(theta),
                                (diff(func[0], phi) - diff(func[2]*r*sin(theta), r))/r/sin(theta),
                                (diff(func[1]*r, r) - diff(func[0], theta))/r]))
    else:
        return Curl(func, Find_Coordinate_System(func))


def Divergence(func, coords):
    """Returns the divergence of a vector function.
    The vector function must be defined as an iterable.
    coords = 'c' for Cartesian, 's' for spherical.
    If coordinate system is not specified, it will atempt to figure it out."""

    if coords == 'c':
        return simplify(diff(func[0], x) + diff(func[1], y) + diff(func[2], z))
    elif coords == 's':
        return simplify(r**-2*diff(func[0]*r**2, r) + (diff(func[1]*sin(theta), theta) + diff(func[2], phi))/(r*sin(theta)))
    else:
        return Divergence(func, Find_Coordinate_System(func))


def Gradient(func, coords):
    """Returns the gradient of a scalar function, expressed as a Sympy matrix..
    coords = 'c' for Cartesian, 's' for spherical.
    If coordinate system is not specified, it will atempt to figure it out."""

    if coords == 'c':
        return Matrix([[func.diff(x),
                        func.diff(y),
                        func.diff(z)]])
    elif coords == 's':
        return Matrix([[func.diff(r),
                        func.diff(theta)/r,
                        func.diff(phi)/(r*sin(theta))]])
    else:
        return Gradient(func, Find_Coordinate_System(func))


def Laplacian(func, coords):
    """Returns the Laplacian of a scalar function."""

    if coords != 'c' and coords != 's':
        coords = Find_Coordinate_System(func)

    return Divergence(Gradient(func, coords), coords)


def Magnitude(iterable):
    """Returns the magnitude, or norm, of a vector.
    Input does not necessarily need to be a vector"""

    return simplify(sqrt(sum(element*element for element in iterable)))


def Normalize(vector):
    """Normalizes a vector."""

    return vector / sqrt(vector.dot(vector))


def Projection(u, v):
    """Calculates the projection of a vector u onto the vector v."""

    return (v.dot(u))/(u.dot(u)) * u


def SphericalIntegral(func, rLimits=(0, oo), thetaLimits=(0, pi), phiLimits=(0, 2*pi)):
    """Returns triple integral of the function using spherical coordinates.
    By default it will integrate over all of space."""

    return Integral(Integral(Integral(func*r**2*sin(theta), (r, rLimits[0], rLimits[1])), (theta, thetaLimits[0], thetaLimits[1])), (phi, phiLimits[0], phiLimits[1]))


# Probability distributions

def Binomial(n, p, var=k):

    return binomial(n, var)*p**n * (1-p)**(n - var)


def Exponential(paramater, var=x):
    """Returns a Exponential distribution with lambda = parameter."""
    return paramater*exp(-var*paramater)


def Normal(mean, stddev, var=x):
    """Returns a Normal or Gaussian distribution."""
    return 1/sqrt(2*pi)/stddev * exp(- ((var-mean)/stddev)**2 / 2)


def Poisson(paramater, var=k):
    """Returns a Poisson distribution with lambda = parameter."""
    return exp(-paramater)*paramater**var / factorial(var)


def Standard_Normal(var=x):
    """Returns Normal(1,1)"""
    return Normal(1, 1, var)


def Uniform(a, b):
    """Uniform distribution."""
    return 1/(b-a)


def Expected_Value(func, var, lower, upper):
    return simplify(integrate(var*func, (var, lower, upper), conds="none"))


def Moment_Generating_Function(func, var, lower, upper):
    return simplify(integrate(exp(t*var)*func, (var, lower, upper), conds="none"))


# Quantum Mechanics

def Commutator(op1, op2, wavefunc=Function("psi")(x, y, z)):
    """Returns the commutator of two operators."""

    try:
        return simplify(op1(op2(wavefunc)) - op2(op1(wavefunc)))
    except TypeError:
        return simplify(op1*op2 - op2*op1)


def Expectation_Value(psi, operator):
    """Returns the expectation value of an operator.
    Operator must be defined as a Python function, not as a Sympy expression."""

    if x in psi:
        return integrate(psi*operator(psi), (x, -oo, oo), conds="none")
    elif r in psi:
        return simplify(SphericalIntegral(psi*operator(psi)).doit())


def Hamiltonian(psi, coords=''):
    """Returns the Hamiltonian for a time-independent wavefunction."""

    if coords == '':
        coords = Find_Coordinate_System(psi)

    if coords == 'c':
        return -hbar**2/(2*m)*Laplacian(psi, 'c') + psi*Function("V")(x, y, z)
    elif coords == 's':
        return -hbar**2/(2*m)*Laplacian(psi, 's') + psi*Function("V")(r, theta, phi)


def Kinetic_Energy(psi):
    """Kinetic energy operator.
    Works up to three dimensions."""
    return -hbar**2/(2*m) * Laplacian(psi)


def Momentum(psi):
    """Momentum operator.
    Works up to three dimensions."""

    return -i*hbar*Gradient(psi)


def Momentum_Squared(psi):
    return -hbar**2 * Laplacian(psi)


def Position(psi):
    """Position operator. (One dimensional.)"""

    if x in psi:
        return x*psi
    elif r in psi:
        return r*psi


def Position_Squared(psi):
    """Position operator. (One dimensional.)"""

    if x in psi:
        return x**2*psi
    elif r in psi:
        return r**2*psi


def Reduced_Mass(m1, m2):
    """Calculates the reduced mass for a two particle system."""
    return m1*m2/(m1+m2)


def Transmission_Coefficient_Barrier(k1, k2, L):
    """Returns the transmission coefficient for a finite-width potential barrier.
    Interpreted as the probability of penetrating the barrier."""
    return 1 / (1 + ((k1**2 + k2**2)*(2*k1*k2)*sinh(k2*L))**2)


def Transmission_Coefficient_Well(k1, k2, L):
    """Returns the transmission coefficient for a potential well.
    Interpreted as the probability of being found outside the well."""
    return (2*k1*k2)**2 / ((k1**2 + k2**2)**2 - ((k2**2 - k1**2)*cos(k2*L))**2)


#  Angular momentum operators.
def Lx(psi):
    """x component of angular momentum."""
    coords = Find_Coordinate_System(psi)
    if coords == 'c':
        return simplify(-i*hbar*(y*psi.diff(z) - z*psi.diff(y)))
    elif coords == 's':
        return simplify(-i*hbar*(-sin(phi)*psi.diff(theta) - cot(theta)*cos(phi)*psi.diff(phi)))


def Ly(psi):
    """y component of angular momentum."""
    coords = Find_Coordinate_System(psi)
    if coords == 'c':
        return simplify(-i*hbar*(z*psi.diff(x) - x*psi.diff(z)))
    elif coords == 's':
        return simplify(-i*hbar*(cos(phi)*psi.diff(theta) - cot(theta)*sin(phi)*psi.diff(phi)))


def Lz(psi):
    """z component of angular momentum."""
    coords = Find_Coordinate_System(psi)
    if coords == 'c':
        return simplify(-i*hbar*(x*psi.diff(y) - y*psi.diff(x)))
    elif coords == 's':
        return simplify(-i*hbar*psi.diff(phi))


def Lplus(psi):
    """Returns Lx + i*Ly"""
    return simplify(Lx(psi) + i*Ly(psi))


def Lminus(psi):
    """Returns Lx - i*Ly"""
    return simplify(Lx(psi) - i*Ly(psi))


#  Misc
def texprint(eqn):
    """Prints the latex code for a given Sympy formula."""
    print(latex(eqn))


#~ When using dark background, copy and paste this.
#~ The command does nothing when run in this file, unfortunately.
print("\ninit_printing(forecolor='White', backcolor='Transparent')")
