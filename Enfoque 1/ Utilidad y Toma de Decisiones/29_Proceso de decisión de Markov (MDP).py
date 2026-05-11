"""
29 - Proceso de Decisión de Markov (MDP)
=========================================
Un MDP se define formalmente como la tupla (S, A, T, R, γ):
  S  : conjunto de estados
  A  : conjunto de acciones
  T  : función de transición  T(s,a,s') = P(s'|s,a)
  R  : función de recompensa  R(s,a,s')
  γ  : factor de descuento  ∈ [0,1)

Propiedades Markovianas:
  - El siguiente estado depende SOLO del estado actual y la acción.
  - El pasado es irrelevante dado el estado presente.

Este archivo define la estructura completa del MDP y muestra cómo
simularlo con diferentes políticas, además de resolver el óptimo con
Iteración de Valores (integrada aquí para ser auto-contenido).
"""

import random


# ─────────────────────── definición del MDP ──────────────────────────

class MDP:
    """
    Proceso de Decisión de Markov tabular finito.
    """
    def __init__(self, estados, acciones, transiciones, recompensas,
                 gamma=0.9, estados_terminales=None):
        self.estados            = estados
        self.acciones           = acciones
        self.transiciones       = transiciones   # {(s,a): [(prob, s')]}
        self.recompensas        = recompensas    # {(s,a,s'): r}
        self.gamma              = gamma
        self.estados_terminales = estados_terminales or set()

    def T(self, s, a):
        return self.transiciones.get((s, a), [])

    def R(self, s, a, s2):
        return self.recompensas.get((s, a, s2), 0.0)

    def es_terminal(self, s):
        return s in self.estados_terminales

    def acciones_disponibles(self, s):
        return [a for a in self.acciones if self.T(s, a)]

    def siguiente_estado(self, s, a):
        """Muestreo estocástico del siguiente estado."""
        trans = self.T(s, a)
        if not trans:
            return s, 0.0
        r = random.random()
        acum = 0.0
        for p, s2 in trans:
            acum += p
            if r <= acum:
                r_val = self.R(s, a, s2)
                return s2, r_val
        return trans[-1][1], self.R(s, a, trans[-1][1])

    def info(self):
        print(f"  Estados        : {len(self.estados)}")
        print(f"  Acciones       : {self.acciones}")
        print(f"  Terminales     : {self.estados_terminales}")
        print(f"  γ (descuento)  : {self.gamma}")
        print(f"  Transiciones   : {len(self.transiciones)} pares (s,a)")


# ─────────── Resolución: Iteración de Valores (integrada) ────────────

def resolver(mdp: MDP, epsilon=1e-8):
    V = {s: 0.0 for s in mdp.estados}
    for _ in range(10_000):
        delta = 0.0
        V_n   = {}
        for s in mdp.estados:
            if mdp.es_terminal(s):
                V_n[s] = 0.0
                continue
            qs = []
            for a in mdp.acciones:
                tr = mdp.T(s, a)
                if tr:
                    qs.append(sum(p*(mdp.R(s,a,s2)+mdp.gamma*V[s2]) for p,s2 in tr))
            V_n[s] = max(qs) if qs else 0.0
            delta  = max(delta, abs(V_n[s]-V[s]))
        V = V_n
        if delta < epsilon:
            break

    politica = {}
    for s in mdp.estados:
        if mdp.es_terminal(s):
            continue
        best_a, best_q = None, float('-inf')
        for a in mdp.acciones:
            tr = mdp.T(s, a)
            if tr:
                q = sum(p*(mdp.R(s,a,s2)+mdp.gamma*V[s2]) for p,s2 in tr)
                if q > best_q:
                    best_q, best_a = q, a
        politica[s] = best_a
    return V, politica


# ──────────────────── Simulación de un episodio ───────────────────────

def simular_episodio(mdp: MDP, politica, estado_inicial,
                     max_pasos=100):
    """
    Simula un episodio siguiendo la política dada.
    Retorna (historial, recompensa_total_descontada).
    """
    s        = estado_inicial
    historial = [(s, None, None)]
    G         = 0.0
    t         = 0

    while not mdp.es_terminal(s) and t < max_pasos:
        a = politica.get(s)
        if a is None:
            break
        s2, r = mdp.siguiente_estado(s, a)
        G += (mdp.gamma ** t) * r
        historial.append((s2, a, r))
        s  = s2
        t += 1

    return historial, G


# ──────────── MDP de ejemplo: mundo lineal de 5 estados ──────────────

def construir_mdp_lineal():
    """
    Estados: 0,1,2,3,4   (4 es terminal con R=+10, 0 es trampa R=-1)
    Acciones: derecha, izquierda
    Transiciones estocásticas: 0.9 prob de ir donde se quiere,
                                0.1 prob de quedarse.
    """
    estados    = [0, 1, 2, 3, 4]
    acciones   = ['der', 'izq']
    terminales = {0, 4}

    transiciones = {}
    recompensas  = {}

    for s in [1, 2, 3]:
        for a in acciones:
            intento = s + 1 if a == 'der' else s - 1
            intento = max(0, min(4, intento))

            trans = [(0.9, intento), (0.1, s)]
            transiciones[(s, a)] = trans
            for p, s2 in trans:
                if s2 == 4:
                    recompensas[(s, a, s2)] = +10.0
                elif s2 == 0:
                    recompensas[(s, a, s2)] = -1.0
                else:
                    recompensas[(s, a, s2)] = -0.1   # costo de paso

    return MDP(estados, acciones, transiciones, recompensas,
               gamma=0.95, estados_terminales=terminales)


# ─────────────────────────── demo ────────────────────────────────────

def demo():
    print("=" * 60)
    print("  Proceso de Decisión de Markov (MDP) — Mundo Lineal")
    print("=" * 60)

    mdp = construir_mdp_lineal()
    print("\n  Definición del MDP:")
    mdp.info()

    print("\n  Resolución con Iteración de Valores...")
    V, pi = resolver(mdp)

    print("\n  Resultados:")
    print(f"  {'Estado':>8}  {'V*(s)':>10}  {'π*(s)':>8}")
    print("  " + "-" * 32)
    for s in mdp.estados:
        etiqueta = "(terminal)" if mdp.es_terminal(s) else ""
        a = pi.get(s, "—")
        print(f"  {s:>8}  {V[s]:>10.4f}  {str(a):>8}  {etiqueta}")

    random.seed(7)
    print("\n  Simulación de 5 episodios desde estado 2:")
    print(f"  {'Episodio':>10}  {'Pasos':>7}  {'G (total)':>12}  Trayectoria")
    print("  " + "-" * 65)
    for ep in range(1, 6):
        hist, G = simular_episodio(mdp, pi, 2, max_pasos=50)
        tray = " → ".join(str(h[0]) for h in hist)
        print(f"  {ep:>10}  {len(hist)-1:>7}  {G:>12.4f}  {tray}")

    # Comparar política óptima vs política aleatoria
    print("\n  Comparación de políticas (10 episodios desde estado 2):")
    pi_aleatorio = {s: random.choice(['der','izq']) for s in [1,2,3]}
    G_opt  = [simular_episodio(mdp, pi, 2, 100)[1] for _ in range(10)]
    G_rand = [simular_episodio(mdp, pi_aleatorio, 2, 100)[1] for _ in range(10)]
    print(f"  G promedio óptima  : {sum(G_opt)/len(G_opt):.4f}")
    print(f"  G promedio aleatoria: {sum(G_rand)/len(G_rand):.4f}")
    print()


if __name__ == "__main__":
    demo()

