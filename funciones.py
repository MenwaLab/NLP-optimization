import numpy as np

# ================= Funciones de prueba =================
def singx_f(x):
    """
    Evalúa la función SINGX (Powell singular extendida).
    x: array de longitud n (múltiplo de 4).
    """
    n = len(x)
    m = n // 4
    f = 0.0
    for j in range(m):
        u1 = x[4*j]     + 10.0*x[4*j+1]
        u2 = x[4*j+2]   - x[4*j+3]
        u3 = x[4*j+1]   - 2.0*x[4*j+2]
        u4 = x[4*j]     - x[4*j+3]
        f += u1**2 + 5.0*u2**2 + u3**4 + 10.0*u4**4
    return f
    
def singx_grad(x):
    """
    Calcula el gradiente analítico de la función SINGX.
    Devuelve un array de la misma dimensión que x.
    """
    n = len(x)
    m = n // 4
    g = np.zeros(n)
    for j in range(m):
        u1 = x[4*j]     + 10.0*x[4*j+1]
        u2 = x[4*j+2]   - x[4*j+3]
        u3 = x[4*j+1]   - 2.0*x[4*j+2]
        u4 = x[4*j]     - x[4*j+3]
        g[4*j]     = 2.0*u1 + 40.0*u4**3
        g[4*j+1]   = 20.0*u1 + 4.0*u3**3
        g[4*j+2]   = 10.0*u2 - 8.0*u3**3
        g[4*j+3]   = -10.0*u2 - 40.0*u4**3
    return g

A = np.array([[1., 10., 0., 0.],
              [0., 0., 1., -1.],
              [0., 1., -2., 0.],
              [1., 0., 0., -1.]])

def singx_hess(x):
    """Hessiana analítica de SINGX (bloque diagonal)."""
    n = len(x); m = n // 4
    H = np.zeros((n, n))
    for j in range(m):
        u = A @ x[4*j:4*j+4]
        D = np.diag([2., 10., 12.*u[2]**2, 120.*u[3]**2])
        H[4*j:4*j+4, 4*j:4*j+4] = A.T @ D @ A
    return H

def almost_f(x):
    """
    Evalúa la función Brown Almost Linear (ALMOST).
    x: array de longitud n ≥ 1.
    """
    n = len(x)
    S = np.sum(x)
    r = x[:n-1] + S - (n+1)        # primeros n-1 residuos
    prod = np.prod(x)
    return np.sum(r**2) + (prod - 1.0)**2

def almost_grad(x):
    """
    Gradiente analítico de ALMOST.
    Devuelve un array de la misma dimensión que x.
    """
    n = len(x)
    S = np.sum(x)
    r = np.zeros(n)
    r[:n-1] = x[:n-1] + S - (n+1)   # residuos lineales
    prod = np.prod(x)
    r[n-1] = prod - 1.0              # residuo del producto
    R = np.sum(r[:n-1])

    g = np.zeros(n)
    for k in range(n):
        # P_k = producto de todos los x_j con j != k (sin dividir por cero)
        if x[k] != 0:
            Pk = prod / x[k]
        else:
            Pk = np.prod(np.delete(x, k))

        if k < n-1:
            g[k] = 2.0 * (R + r[k]) + 2.0 * r[n-1] * Pk
        else:
            g[k] = 2.0 * R + 2.0 * r[n-1] * Pk
    return g

def almost_hess(x):
    """Hessiana analítica de ALMOST."""
    n = len(x); S = np.sum(x); P = np.prod(x)
    r = np.zeros(n)
    r[:n-1] = x[:n-1] + S - (n+1)
    r[n-1] = P - 1.0
    rn = r[n-1]
    Pk = np.array([P/x[k] if x[k] != 0 else np.prod(np.delete(x, k)) for k in range(n)])
    H = np.zeros((n, n))
    for k in range(n-1):
        for l in range(n-1):
            H[k, l] = 2*(n + 1 + (1 if k == l else 0)) + 2*Pk[k]*Pk[l]
            if k != l:
                H[k, l] += 2*rn*np.prod(np.delete(x, [k, l]))
        H[k, n-1] = 2*n + 2*Pk[k]*Pk[n-1] + 2*rn*np.prod(np.delete(x, [k, n-1]))
        H[n-1, k] = H[k, n-1]
    H[n-1, n-1] = 2*(n-1) + 2*Pk[n-1]**2
    return H

# ================= Métodos =================
def max_descenso(f, grad, x0, tol=1e-6, max_iter=10000,
                 c1=1e-4, rho=0.5, verbose=True):
    """
    Método de Máximo Descenso con búsqueda lineal de Armijo.

    Parámetros
    ----------
    f    : función objetivo
    grad : gradiente analítico
    x0   : punto inicial
    tol  : tolerancia sobre ||grad||
    c1   : constante de Armijo (típicamente 1e-4)
    rho  : factor de reducción del paso (típicamente 0.5)

    Devuelve
    --------
    x_opt   : aproximación al mínimo
    history : diccionario con listas de iterados, valores de f y ||grad||
    """
    x = x0.copy().astype(float)
    history = {'x': [x.copy()],
               'f': [f(x)],
               'gnorm': [np.linalg.norm(grad(x))]}

    for k in range(max_iter):
        g = grad(x)
        gnorm = np.linalg.norm(g)

        if gnorm < tol:
            if verbose:
                print(f"Convergencia en {k} iteraciones, ||g|| = {gnorm:.2e}")
            break

        d = -g / gnorm                       # dirección de máximo descenso
        fk = f(x)
        gd = g @ d                           # = -||g|| < 0
        alpha = 1.0                          # paso inicial

        # Backtracking de Armijo
        while f(x + alpha*d) > fk + c1*alpha*gd:
            alpha *= rho

        x = x + alpha*d
        history['x'].append(x.copy())
        history['f'].append(f(x))
        history['gnorm'].append(np.linalg.norm(grad(x)))

        if verbose and (k % 100 == 0 or k < 5):
            print(f"k={k:4d}  f={history['f'][-1]:.6e}  "
                  f"||g||={history['gnorm'][-1]:.3e}  alpha={alpha:.3e}")

    return x, history

def newton_method(f, grad, hess, x0, tol=1e-10, max_iter=100, verbose=True):
    """Método de Newton puro para minimización irrestricta."""
    x = x0.copy().astype(float)
    hist = {'x': [x.copy()], 'f': [f(x)], 'gnorm': [np.linalg.norm(grad(x))]}
    for k in range(max_iter):
        g = grad(x); gn = np.linalg.norm(g)
        if gn < tol:
            if verbose: print(f"Convergencia en {k} iteraciones, ||g||={gn:.2e}")
            break
        H = hess(x)
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            if verbose: print(f"Hessiana singular en k={k}, se detiene el método")
            break
        x = x + d
        hist['x'].append(x.copy()); hist['f'].append(f(x)); hist['gnorm'].append(np.linalg.norm(grad(x)))
        if verbose:
            print(f"k={k}: f={hist['f'][-1]:.6e}, ||g||={hist['gnorm'][-1]:.3e}, ||d||={np.linalg.norm(d):.3e}")
    return x, hist

def bfgs(f, grad, x0, tol=1e-10, max_iter=2000,
         c1=1e-4, c2=0.9, max_ls=60, verbose=True, print_every=10):
    """
    Método BFGS con búsqueda lineal de Armijo backtracking.
    Devuelve (x_opt, history).
    """
    x = x0.copy().astype(float)
    n = len(x)
    B = np.eye(n)                     # aproximación inicial de [∇²f]^{-1}
    g = grad(x)
    hist = {'x': [x.copy()], 'f': [f(x)],
            'gnorm': [np.linalg.norm(g)], 'alpha': []}

    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            if verbose: print(f"Convergencia en {k} iteraciones, ||g||={np.linalg.norm(g):.2e}")
            break

        d = -B @ g                     # dirección cuasi-Newton
        fk = f(x)
        gd = g @ d

        # ----- Búsqueda lineal de Armijo con backtracking -----
        alpha = 1.0
        for _ in range(max_ls):
            if f(x + alpha*d) <= fk + c1*alpha*gd:
                break
            alpha *= 0.5

        x_new = x + alpha*d
        g_new = grad(x_new)

        s = x_new - x
        y = g_new - g
        ys = y @ s

        # Actualización BFGS (solo si la condición de curvatura se cumple)
        if ys > 1e-12:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            B = V @ B @ V.T + rho * np.outer(s, s)
        # si ys ≤ 0, se omite la actualización (mantiene B definida positiva)

        x = x_new
        g = g_new
        hist['x'].append(x.copy())
        hist['f'].append(f(x))
        hist['gnorm'].append(np.linalg.norm(g))
        hist['alpha'].append(alpha)

        if verbose and (k < 5 or k % print_every == 0):
            print(f"k={k:4d}  f={hist['f'][-1]:.6e}  "
                  f"||g||={hist['gnorm'][-1]:.3e}  alpha={alpha:.3e}")

    return x, hist

def trust_region_cauchy(f, grad, hess, x0, delta0=1.0,
                        eta1=0.25, eta2=0.75, gamma1=0.25, gamma2=2.0,
                        tol=1e-8, max_iter=500, verbose=True):
    """
    Región de confianza con paso de Cauchy.
    """
    x = x0.copy().astype(float)
    delta = delta0
    hist = {'x': [x.copy()], 'f': [f(x)], 'gnorm': [np.linalg.norm(grad(x))],
            'delta': [delta], 'rho': []}

    for k in range(max_iter):
        g = grad(x); gn = np.linalg.norm(g)
        if gn < tol:
            if verbose: print(f"Convergencia en {k} iteraciones, ||g||={gn:.2e}")
            break

        B = hess(x)
        # ----- Paso de Cauchy -----
        gBg = g @ B @ g
        alpha_c = min(gn**2 / gBg, delta / gn) if gBg > 0 else delta / gn
        p = -alpha_c * g

        # ----- Cociente de reducción -----
        pred = -(g @ p + 0.5 * p @ B @ p)   # reducción predicha
        real = f(x) - f(x + p)               # reducción real
        rho = real / pred if pred != 0 else 0.0

        # ----- Actualización del radio y del iterado -----
        if rho < eta1:
            delta = gamma1 * delta
        elif rho > eta2 and abs(np.linalg.norm(p) - delta) < 1e-12:
            delta = min(gamma2 * delta, 1e6)
        # si rho > eta1 pero no en la frontera, delta no cambia

        if rho > 0:                          # aceptar paso
            x = x + p
        # si rho <= 0, rechazar (x no cambia)

        hist['x'].append(x.copy())
        hist['f'].append(f(x))
        hist['gnorm'].append(np.linalg.norm(grad(x)))
        hist['delta'].append(delta)
        hist['rho'].append(rho)

        if verbose and (k < 5 or k % 20 == 0):
            print(f"k={k:3d}  f={hist['f'][-1]:.6e}  ||g||={hist['gnorm'][-1]:.3e}  "
                  f"Δ={delta:.3e}  ρ={rho:.3e}")

    return x, hist