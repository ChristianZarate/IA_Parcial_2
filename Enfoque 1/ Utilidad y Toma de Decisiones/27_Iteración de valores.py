"""
27 - Iteración de Valores (Value Iteration)
===========================================
Algoritmo de Programación Dinámica para resolver MDPs.

Ecuación de Bellman (actualización):
  V_{k+1}(s) = max_a  Σ_{s'} T(s, a, s') · [R(s, a, s') + γ · V_k(s')]

Converge cuando ||V_{k+1} − V_k||_∞ < ε (epsilon de convergencia).
La política óptima se extrae al final:
  π*(s) = argmax_a  Σ_{s'} T(s,a,s') · [R(s,a,s') + γ · V*(s')]
"""


# ────────────────────────── clase MDP ────────────────────────────────

class MDP:
    """
    Proceso de Decisión de Markov (finito, tabular).
    estados   : lista de estados
    acciones  : lista de acciones
    T(s,a,s') : probabilidad de transición
    R(s,a,s') : recompensa de transición
    gamma      : factor de descuento
    """

    def __init__(self, estados, acciones, transiciones, recompensas, gamma=0.9):
        self.estados       = estados
        self.acciones      = acciones
        self.transiciones  = transiciones   # dict {(s,a): [(prob, s')]}
        self.recompensas   = recompensas    # dict {(s,a,s'): r}
        self.gamma         = gamma

    def T(self, s, a):
        """Lista de (probabilidad, estado_siguiente)."""
        return self.transiciones.get((s, a), [])

    def R(self, s, a, s_prima):
        return self.recompensas.get((s, a, s_prima), 0.0)


# ─────────────────── Iteración de Valores ────────────────────────────

def iteracion_valores(mdp: MDP, epsilon=1e-6, max_iter=1000):
    """
    Retorna (V, política, iteraciones).
    V       : dict {estado: valor}
    política: dict {estado: acción}
    """
    # Inicializar V en cero
    V = {s: 0.0 for s in mdp.estados}

    for it in range(1, max_iter + 1):
        V_nuevo = {}
        delta   = 0.0

        for s in mdp.estados:
            # Calcular Q(s, a) para cada acción
            valores_acciones = []
            for a in mdp.acciones:
                trans = mdp.T(s, a)
                if not trans:
                    continue
                q_sa = sum(p * (mdp.R(s, a, s2) + mdp.gamma * V[s2])
                           for p, s2 in trans)
                valores_acciones.append(q_sa)

            V_nuevo[s] = max(valores_acciones) if valores_acciones else 0.0
            delta = max(delta, abs(V_nuevo[s] - V[s]))

        V = V_nuevo
        if delta < epsilon:
            print(f"  Convergió en {it} iteraciones (delta={delta:.2e})")
            break
    else:
        print(f"  No convergió en {max_iter} iteraciones.")

    # Extraer política
    politica = {}
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
        politica[s] = mejor_a

    return V, politica


# ─────────────── Mundo Grid 4×3 (Russell & Norvig) ───────────────────

def construir_grid():
    """
    Grid 4×3 clásico:
      Columnas 0-3, Filas 0-2 (0=abajo)
      Obstrucción en (1,1).
      Terminal en (3,2) recompensa +1, (3,1) recompensa -1.
      Recompensa paso = -0.04
    """
    # Definir estados (evitar obstrucción)
    obstruccion = (1, 1)
    estados = [(c, f) for c in range(4) for f in range(3)
               if (c, f) != obstruccion]

    terminales = {(3, 2): +1.0, (3, 1): -1.0}
    acciones   = ['N', 'S', 'E', 'O']
    r_paso     = -0.04
    gamma      = 0.9

    def siguiente(s, a):
        """Estado resultante de mover en dirección a desde s."""
        c, f = s
        dc = {'N': (0,1), 'S': (0,-1), 'E': (1,0), 'O': (-1,0)}[a]
        nc, nf = c + dc[0], f + dc[1]
        if (nc, nf) in estados:
            return (nc, nf)
        return s   # choca con pared → permanece

    def perpendiculares(a):
        """Acciones perpendiculares (para el movimiento estocástico)."""
        perp = {'N': ['E','O'], 'S': ['E','O'],
                'E': ['N','S'], 'O': ['N','S']}
        return perp[a]

    transiciones = {}
    recompensas  = {}

    for s in estados:
        if s in terminales:
            continue
        for a in acciones:
            perp = perpendiculares(a)
            # 0.8 prob de ir hacia donde se intenta, 0.1 a cada perpendicular
            trans_s = [
                (0.8, siguiente(s, a)),
                (0.1, siguiente(s, perp[0])),
                (0.1, siguiente(s, perp[1])),
            ]
            transiciones[(s, a)] = trans_s
            for p, s2 in trans_s:
                r = terminales.get(s2, r_paso)
                recompensas[(s, a, s2)] = r

    return MDP(estados, acciones, transiciones, recompensas, gamma)


# ─────────────────────────── demo ────────────────────────────────────

SIMBOLO_ACCION = {'N': '↑', 'S': '↓', 'E': '→', 'O': '←', None: 'T'}

def imprimir_grid(valores, politica=None, estados_terminales=None):
    terminales = estados_terminales or {}
    print()
    for f in range(2, -1, -1):
        fila_v = ""
        fila_p = ""
        for c in range(4):
            s = (c, f)
            if s == (1, 1):
                fila_v += f"{'###':>9s} "
                fila_p += f"{'###':>5s} "
            elif s in terminales:
                fila_v += f"{terminales[s]:>+9.3f} "
                fila_p += f"{'T':>5s} "
            elif s in valores:
                fila_v += f"{valores[s]:>9.4f} "
                a = politica.get(s) if politica else None
                fila_p += f"{SIMBOLO_ACCION.get(a,'?'):>5s} "
            else:
                fila_v += f"{'---':>9s} "
                fila_p += f"{'---':>5s} "
        print(f"  {fila_v}    {fila_p}")
    print()


def demo():
    print("=" * 65)
    print("  Iteración de Valores — Grid 4×3 (Russell & Norvig)")
    print("=" * 65)

    mdp = construir_grid()
    V, politica = iteracion_valores(mdp, epsilon=1e-8)

    terminales = {(3, 2): +1.0, (3, 1): -1.0}

    print("\n  Valores V*(s)  /  Política π*(s):")
    print(f"  {'Valores':>42s}    {'Política':>25s}")
    imprimir_grid(V, politica, terminales)

    print("  Valores seleccionados:")
    for s in [(0,0),(1,0),(2,0),(3,0),(0,2),(2,2),(3,2),(3,1)]:
        if s in V:
            a = politica.get(s, 'T')
            print(f"    {str(s):8s}  V={V[s]:+.4f}  π={SIMBOLO_ACCION.get(a,'T')}")
    print()


if __name__ == "__main__":
    demo()

