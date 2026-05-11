"""
30 - MDP Parcialmente Observable (POMDP)
=========================================
En un POMDP el agente no observa el estado directamente.
Tupla: (S, A, T, R, Ω, O, γ)
  S  : estados
  A  : acciones
  T  : transiciones  P(s'|s,a)
  R  : recompensas   R(s,a)
  Ω  : observaciones
  O  : función de observación  P(o|s',a)
  γ  : descuento

El agente mantiene una *distribución de creencias* b(s) = P(S=s | historial)
y la actualiza con la regla de Bayes tras cada acción-observación.

Política basada en creencias:  π : Δ(S) → A

Aquí implementamos:
  1. Actualización de creencias (Belief Update)
  2. Política QMDP (aproximación: resuelve el MDP subyacente y evalúa
     la acción como el promedio ponderado por las creencias)
"""

import random


# ─────────────── Estructura del POMDP ────────────────────────────────

class POMDP:
    def __init__(self, estados, acciones, observaciones,
                 transiciones, recompensas, obs_func, gamma=0.9):
        """
        transiciones: {(s,a): [(prob, s')]}
        recompensas : {(s,a): r}
        obs_func    : {(s2,a): [(prob, o)]}  — P(o|s',a)
        """
        self.estados       = estados
        self.acciones      = acciones
        self.observaciones = observaciones
        self.transiciones  = transiciones
        self.recompensas   = recompensas
        self.obs_func      = obs_func
        self.gamma         = gamma

    def T(self, s, a):
        return self.transiciones.get((s, a), [])

    def R(self, s, a):
        return self.recompensas.get((s, a), 0.0)

    def O(self, s2, a):
        return self.obs_func.get((s2, a), [])

    def prob_obs(self, o, s2, a):
        for p, obs in self.O(s2, a):
            if obs == o:
                return p
        return 0.0

    def siguiente_estado(self, s, a):
        r = random.random(); acum = 0.0
        for p, s2 in self.T(s, a):
            acum += p
            if r <= acum:
                return s2
        return self.T(s, a)[-1][1]

    def obtener_observacion(self, s2, a):
        r = random.random(); acum = 0.0
        for p, o in self.O(s2, a):
            acum += p
            if r <= acum:
                return o
        return self.O(s2, a)[-1][1]


# ─────────────── Actualización de Creencias ──────────────────────────

def actualizar_creencias(b, a, o, pomdp: POMDP):
    """
    b'(s') ∝ P(o|s',a) · Σ_s P(s'|s,a) · b(s)
    """
    b_nuevo = {}
    for s2 in pomdp.estados:
        prob_o_dado_s2 = pomdp.prob_obs(o, s2, a)
        suma = sum(prob_t * b.get(s, 0.0)
                   for s in pomdp.estados
                   for prob_t, st in pomdp.T(s, a) if st == s2)
        b_nuevo[s2] = prob_o_dado_s2 * suma

    total = sum(b_nuevo.values())
    if total > 0:
        b_nuevo = {s: v/total for s, v in b_nuevo.items()}
    else:
        # Si prob=0, volver a prior uniforme
        n = len(pomdp.estados)
        b_nuevo = {s: 1/n for s in pomdp.estados}
    return b_nuevo


# ────────────── Política QMDP (aproximación eficiente) ────────────────

def iteracion_valores_mdp(pomdp: POMDP, epsilon=1e-8):
    """Resuelve el MDP subyacente (observabilidad completa)."""
    V = {s: 0.0 for s in pomdp.estados}
    for _ in range(10_000):
        delta = 0.0
        Vn    = {}
        for s in pomdp.estados:
            qs = []
            for a in pomdp.acciones:
                tr = pomdp.T(s, a)
                if tr:
                    qs.append(pomdp.R(s, a) +
                              pomdp.gamma * sum(p*V[s2] for p,s2 in tr))
            Vn[s] = max(qs) if qs else 0.0
            delta = max(delta, abs(Vn[s]-V[s]))
        V = Vn
        if delta < epsilon:
            break
    return V


def politica_qmdp(b, V_mdp, pomdp: POMDP):
    """
    π_QMDP(b) = argmax_a Σ_s b(s) · Q*(s,a)
    donde Q*(s,a) = R(s,a) + γ · Σ_{s'} T(s,a,s') · V*(s')
    """
    mejor_a, mejor_val = None, float('-inf')
    for a in pomdp.acciones:
        val = 0.0
        for s in pomdp.estados:
            q_sa = pomdp.R(s, a) + pomdp.gamma * sum(
                p * V_mdp[s2] for p, s2 in pomdp.T(s, a))
            val += b[s] * q_sa
        if val > mejor_val:
            mejor_val, mejor_a = val, a
    return mejor_a


# ────────────── Demo: Tiger Problem (clásico en POMDPs) ───────────────
#
# Dos puertas: izquierda y derecha.
# Detrás de una hay un tigre (S=TI → tigre a la izq, S=TD → tigre a la der).
# Acciones: escuchar (L), abrir izquierda (AI), abrir derecha (AD).
# Si abres la puerta con el tigre → -100; si abres la correcta → +10.
# Escuchar da una pista con 85% de precisión.

def construir_tiger_problem():
    estados      = ['TI', 'TD']  # tigre Izq, tigre Der
    acciones     = ['L', 'AI', 'AD']
    observaciones= ['oye_izq', 'oye_der']
    gamma        = 0.95

    # Transiciones: abrir puerta reinicia; escuchar no cambia estado
    trans = {}
    for s in estados:
        trans[(s, 'L')]  = [(1.0, s)]
        # Tras abrir → 50/50
        trans[(s, 'AI')] = [(0.5,'TI'), (0.5,'TD')]
        trans[(s, 'AD')] = [(0.5,'TI'), (0.5,'TD')]

    # Recompensas
    rew = {}
    for s in estados:
        rew[(s, 'L')]  = -1.0
        rew[('TI','AI')] = -100.0  # tigre a izq, abre izq
        rew[('TD','AI')] = +10.0   # tigre a der, abre izq
        rew[('TI','AD')] = +10.0
        rew[('TD','AD')] = -100.0

    # Función de observación: P(o | s', a)
    obs = {}
    for s2 in estados:
        # Escuchar: 85% de prob correcta
        if s2 == 'TI':
            obs[(s2,'L')]  = [(0.85,'oye_izq'), (0.15,'oye_der')]
        else:
            obs[(s2,'L')]  = [(0.15,'oye_izq'), (0.85,'oye_der')]
        # Tras abrir puerta: 50/50
        for a in ['AI','AD']:
            obs[(s2,a)] = [(0.5,'oye_izq'), (0.5,'oye_der')]

    return POMDP(estados, acciones, observaciones, trans, rew, obs, gamma)


def demo():
    print("=" * 60)
    print("  POMDP — Tiger Problem")
    print("=" * 60)

    pomdp = construir_tiger_problem()
    random.seed(0)

    # Resolver MDP base para QMDP
    V_mdp = iteracion_valores_mdp(pomdp)
    print(f"\n  V*(TI) = {V_mdp['TI']:.4f}  |  V*(TD) = {V_mdp['TD']:.4f}")

    # Simular con política QMDP
    estado_real = random.choice(pomdp.estados)
    print(f"\n  Estado real (oculto): {estado_real}")
    print(f"  Creencias iniciales : {{TI: 0.50, TD: 0.50}}")
    b = {'TI': 0.5, 'TD': 0.5}

    print(f"\n  {'Paso':>5}  {'Acción':>8}  {'Obs':>10}  "
          f"{'b(TI)':>8}  {'b(TD)':>8}  {'Recompensa':>12}")
    print("  " + "-" * 60)

    recompensa_total = 0.0
    for t in range(1, 11):
        a = politica_qmdp(b, V_mdp, pomdp)
        s2 = pomdp.siguiente_estado(estado_real, a)
        o  = pomdp.obtener_observacion(s2, a)
        r  = pomdp.R(estado_real, a)
        recompensa_total += (pomdp.gamma ** (t-1)) * r
        b  = actualizar_creencias(b, a, o, pomdp)
        estado_real = s2

        print(f"  {t:>5}  {a:>8}  {o:>10}  "
              f"{b['TI']:>8.4f}  {b['TD']:>8.4f}  {r:>12.2f}")

    print(f"\n  Recompensa total descontada: {recompensa_total:.4f}")
    print()


if __name__ == "__main__":
    demo()

