"""
28 - Iteración de Políticas (Policy Iteration)
===============================================
Algoritmo alternativo a la Iteración de Valores para resolver MDPs.

Dos fases por iteración:
  1. Evaluación de Política: calcular V^π resolviendo el sistema lineal
       V^π(s) = Σ_{s'} T(s,π(s),s') · [R(s,π(s),s') + γ·V^π(s')]
  2. Mejora de Política: actualizar π greedy respecto a V^π
       π'(s) = argmax_a Σ_{s'} T(s,a,s') · [R(s,a,s') + γ·V^π(s')]

Converge en menos iteraciones que Value Iteration pero cada paso es más
costoso (resolución de sistema lineal).
"""

import numpy as np


# ────────────────── MDP (reutiliza estructura del tema 27) ─────────────

class MDP:
    def __init__(self, estados, acciones, transiciones, recompensas, gamma=0.9):
        self.estados       = estados
        self.acciones      = acciones
        self.transiciones  = transiciones
        self.recompensas   = recompensas
        self.gamma         = gamma

    def T(self, s, a):
        return self.transiciones.get((s, a), [])

    def R(self, s, a, s_prima):
        return self.recompensas.get((s, a, s_prima), 0.0)


# ──────────────── Fase 1: Evaluación de Política ─────────────────────

def evaluar_politica_iterativa(mdp, politica, epsilon=1e-8, max_iter=5000):
    """
    Evalúa V^π mediante iteración (más simple que resolución matricial,
    equivalente en el límite).
    """
    V = {s: 0.0 for s in mdp.estados}
    for _ in range(max_iter):
        delta = 0.0
        V_nuevo = {}
        for s in mdp.estados:
            a = politica.get(s)
            if a is None:   # estado terminal: no hay acción
                V_nuevo[s] = V[s]
                continue
            trans = mdp.T(s, a)
            V_nuevo[s] = sum(p * (mdp.R(s, a, s2) + mdp.gamma * V[s2])
                             for p, s2 in trans) if trans else 0.0
            delta = max(delta, abs(V_nuevo[s] - V[s]))
        V = V_nuevo
        if delta < epsilon:
            break
    return V


def evaluar_politica_matricial(mdp, politica):
    """
    Resuelve (I − γP^π)·V = R^π usando numpy.
    Más precisa pero O(n³).
    """
    n      = len(mdp.estados)
    idx    = {s: i for i, s in enumerate(mdp.estados)}
    P      = np.zeros((n, n))
    R_vec  = np.zeros(n)

    for s in mdp.estados:
        i = idx[s]
        a = politica.get(s)
        if a is None:
            continue
        for p, s2 in mdp.T(s, a):
            j = idx[s2]
            P[i, j] += p
            R_vec[i] += p * mdp.R(s, a, s2)

    A = np.eye(n) - mdp.gamma * P
    V_vec = np.linalg.solve(A, R_vec)
    return {s: float(V_vec[idx[s]]) for s in mdp.estados}


# ──────────────── Fase 2: Mejora de Política ─────────────────────────

def mejorar_politica(mdp, V, politica_actual):
    """Retorna nueva política greedy y si hubo cambios."""
    nueva = {}
    cambio = False
    for s in mdp.estados:
        mejor_a, mejor_q = None, float('-inf')
        for a in mdp.acciones:
            trans = mdp.T(s, a)
            if not trans:
                continue
            q = sum(p * (mdp.R(s, a, s2) + mdp.gamma * V[s2])
                    for p, s2 in trans)
            if q > mejor_q:
                mejor_q, mejor_a = q, a
        nueva[s] = mejor_a
        if mejor_a != politica_actual.get(s):
            cambio = True
    return nueva, cambio


# ───────────────────── Iteración de Políticas ────────────────────────

def iteracion_politicas(mdp: MDP, usar_matricial=True):
    """
    Retorna (V*, π*, número_de_iteraciones).
    """
    # Política inicial: primera acción disponible
    politica = {}
    for s in mdp.estados:
        for a in mdp.acciones:
            if mdp.T(s, a):
                politica[s] = a
                break

    for it in range(1, 1000):
        # Fase 1: Evaluar
        if usar_matricial:
            V = evaluar_politica_matricial(mdp, politica)
        else:
            V = evaluar_politica_iterativa(mdp, politica)

        # Fase 2: Mejorar
        politica_nueva, cambio = mejorar_politica(mdp, V, politica)
        politica = politica_nueva

        if not cambio:
            print(f"  Política estable tras {it} iteración(es).")
            return V, politica, it

    print("  Advertencia: no convergió.")
    return V, politica, -1


# ─────────────── Grid 4×3 (reutilizado del tema 27) ──────────────────

def construir_grid():
    obstruccion = (1, 1)
    estados     = [(c, f) for c in range(4) for f in range(3)
                   if (c, f) != obstruccion]
    terminales  = {(3, 2): +1.0, (3, 1): -1.0}
    acciones    = ['N', 'S', 'E', 'O']
    r_paso      = -0.04
    gamma       = 0.9

    def siguiente(s, a):
        c, f = s
        dc = {'N':(0,1),'S':(0,-1),'E':(1,0),'O':(-1,0)}[a]
        nc, nf = c+dc[0], f+dc[1]
        return (nc,nf) if (nc,nf) in estados else s

    def perp(a):
        return {'N':['E','O'],'S':['E','O'],'E':['N','S'],'O':['N','S']}[a]

    transiciones, recompensas = {}, {}
    for s in estados:
        if s in terminales:
            continue
        for a in acciones:
            tr = [(0.8, siguiente(s,a)),
                  (0.1, siguiente(s,perp(a)[0])),
                  (0.1, siguiente(s,perp(a)[1]))]
            transiciones[(s,a)] = tr
            for p, s2 in tr:
                recompensas[(s,a,s2)] = terminales.get(s2, r_paso)

    return MDP(estados, acciones, transiciones, recompensas, gamma)


SIMBOLO = {'N':'↑','S':'↓','E':'→','O':'←',None:'T'}

def imprimir_grid(V, politica, terminales):
    for f in range(2,-1,-1):
        rv = re = ""
        for c in range(4):
            s = (c,f)
            if s == (1,1):
                rv += f"{'###':>9s} "; re += f"{'###':>5s} "
            elif s in terminales:
                rv += f"{terminales[s]:>+9.3f} "; re += f"{'T':>5s} "
            elif s in V:
                rv += f"{V[s]:>9.4f} "
                re += f"{SIMBOLO.get(politica.get(s),'?'):>5s} "
            else:
                rv += f"{'---':>9s} "; re += f"{'---':>5s} "
        print(f"  {rv}    {re}")
    print()


def demo():
    print("=" * 65)
    print("  Iteración de Políticas — Grid 4×3")
    print("=" * 65)

    mdp        = construir_grid()
    terminales = {(3,2): +1.0, (3,1): -1.0}

    print("\n  [Método matricial]")
    V, pi, iters = iteracion_politicas(mdp, usar_matricial=True)
    print("  Valores V*(s)  /  Política π*(s):")
    imprimir_grid(V, pi, terminales)

    print("  [Método iterativo]")
    V2, pi2, iters2 = iteracion_politicas(mdp, usar_matricial=False)
    print("  Valores V*(s)  /  Política π*(s):")
    imprimir_grid(V2, pi2, terminales)

    print("  Comparación de métodos:")
    print(f"  {'Estado':10s}  {'V(matricial)':>14s}  {'V(iterativo)':>14s}  {'Dif':>10s}")
    print("  " + "-" * 55)
    for s in sorted(mdp.estados):
        d = abs(V[s] - V2[s])
        print(f"  {str(s):10s}  {V[s]:>14.6f}  {V2[s]:>14.6f}  {d:>10.2e}")
    print()


if __name__ == "__main__":
    demo()

