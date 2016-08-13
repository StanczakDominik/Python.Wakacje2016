import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
        description='liczy poissona')
parser.add_argument('kopytko', type=int,
    help="rozmiar siatki")
parser.add_argument('--filename', default=False, type=str)
parser.add_argument('--field', action='store_true',
    help="plots field")
parser.add_argument('--particles', action='store_true',
    help="plots particle trajectories")
args = parser.parse_args()
filename = args.filename
plot_field = args.field
particles = args.particles

epsilon_zero = 1


def L2_norm(a, b):
    return np.sqrt(np.sum((a-b)**2))


N = args.kopytko
x, dx = y, dy = np.linspace(-1, 1, N, retstep=True)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2
indices = R2 < 0.25
charge = np.zeros_like(X)
charge[indices] = 1
potential = np.zeros((len(x)+2, len(x) +2))
potential[:,0] = -1
potential[:,-1] = +1

def solve_poisson_jacobi_2d(dx, potential, charge, epsilon = 1e-6, debug=True):
    """rozwiązuje równanie płazona
    zwraca tablicę z rozwiązaniem bez warunków brzegowych
    dx: krok przestrzenny na siatce
    potential: tablica rozmiaru N + 2, krawędzie zawierają warunki brzegowe
    charge: tablica rozmiaru N, ładunek na siatce
    
    $$ \Big(\psi(x-dx) + \psi(x+dx)\Big) /2 + \rho (dx)^2 / \epsilon  =
        = \psi(x) $$
    """
    
    L2_norm_current = 1
    i = 0
    
    potential1 = potential.copy()
    potential2 = potential.copy()
    while L2_norm_current > epsilon:
        i += 1
        potential2[1:-1, 1:-1] = (potential1[:-2, 1:-1] +\
                                  potential1[2:, 1:-1] +\
                                  potential1[1:-1, :-2] +\
                                  potential1[1:-1, 2:])/4 +\
                                  charge*dx**2 / epsilon_zero
        L2_norm_current = L2_norm(potential1, potential2)
        potential1[:] = potential2
    if debug:
        print(i, L2_norm_current)
    return potential1[1:-1, 1:-1]
    
solution_2d = solve_poisson_jacobi_2d(dx, potential, charge)
plt.title("{} x {}".format(N, N))
if plot_field:
    Ey, Ex = np.gradient(solution_2d, -dx, -dy)
    plt.quiver(X, Y, Ex, Ey)
    #plt.streamplot(X, Y, Ex, Ey)
    plt.contour(X, Y, solution_2d)
elif particles:
    Ey, Ex = np.gradient(solution_2d, -dx, -dy)
    #plt.quiver(X, Y, Ex, Ey, units='xy', scale=10)
    dr = np.array([dx, dy])
    qm = 1
    
    def f(r, t):
        try:
            ind_x, ind_y = index = ((r[:2] + 1) / dr).astype(int)
            cell_location = x[index[0]], y[index[1]]
            p_left, p_bot = percentage_right = (r[:2] - cell_location)/dr
            p_right, p_top = 1 - percentage_right
            
            percentages = np.array([[p_top*p_left, p_top*p_right],
                                    [p_bot*p_left, p_bot*p_right]])
            Ex_local = Ex[ind_x:ind_x+2, ind_y:ind_y+2]
            Ex_interpolated = qm*(Ex_local * percentages).sum()
            Ey_local = Ey[ind_x:ind_x+2, ind_y:ind_y+2]
            Ey_interpolated = qm*(Ey_local * percentages).sum()
            return np.array([r[2], r[3], Ex_interpolated, Ey_interpolated])
        except IndexError:
            return np.zeros(4)
    from scipy.integrate import odeint
    
    for i in range(10):
        r = np.random.random(4)-0.5
        r[2:] = 0
        t = np.linspace(0, 1, 100)
        trajectory = odeint(f, r, t)
        plt.plot(trajectory[:,0], trajectory[:,1], "go-")
    plt.streamplot(X, Y, Ex, Ey)
    #plt.quiver(r[0], r[1], Ex_interpolated, Ey_interpolated, units='xy', scale=10)
    
    
    plt.plot(r[0], r[1], "bo")
    #plt.plot(cell_location[0], cell_location[1], "r*")
elif False:
    Ey, Ex = np.gradient(solution_2d, -dx, -dy)
    #plt.quiver(X, Y, Ex, Ey, units='xy', scale=10)
    r = np.array([0.4, 0.3])
    dr = np.array([dx, dy])
    qm = 1
    
    def f(r, t):
        try:
            ind_x, ind_y = index = ((r + 1) / dr).astype(int)
            cell_location = x[index[0]], y[index[1]]
            p_left, p_bot = percentage_right = (r - cell_location)/dr
            p_right, p_top = 1 - percentage_right
            
            percentages = np.array([[p_top*p_left, p_top*p_right],
                                    [p_bot*p_left, p_bot*p_right]])
            #import ipdb; ipdb.set_trace()
            Ex_local = Ex[ind_x:ind_x+2, ind_y:ind_y+2]
            Ex_interpolated = qm*(Ex_local * percentages).sum()
            Ey_local = Ey[ind_x:ind_x+2, ind_y:ind_y+2]
            Ey_interpolated = qm*(Ey_local * percentages).sum()
            return np.array([Ex_interpolated, Ey_interpolated])
        except ValueError:
            print(r)
            import ipdb; ipdb.set_trace()
        except IndexError:
            return np.zeros(2)
    from scipy.integrate import odeint
    t = np.linspace(0, 1, 1000)
    trajectory = odeint(f, r, t)
    plt.plot(trajectory[:,0], trajectory[:,1], "go-")
    plt.streamplot(X, Y, Ex, Ey)
    #plt.quiver(r[0], r[1], Ex_interpolated, Ey_interpolated, units='xy', scale=10)
    
    #import ipdb; ipdb.set_trace()
    
    plt.plot(r[0], r[1], "bo")
    #plt.plot(cell_location[0], cell_location[1], "r*")
    
else:
    plt.imshow(solution_2d)
if filename:
    plt.savefig(filename)
plt.show()
