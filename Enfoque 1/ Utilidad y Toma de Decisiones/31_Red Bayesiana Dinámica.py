"""
31 - Red Bayesiana Dinámica (Dynamic Bayesian Network — DBN)
=============================================================
Una DBN es una Red Bayesiana que modela distribuciones de probabilidad
sobre secuencias temporales de variables.

Componentes principales:
  - Modelo de transición: P(X_t | X_{t-1})   — cómo evoluciona el estado
  - Modelo de observación: P(E_t | X_t)       — cómo se observa el estado
  - Distribución inicial: P(X_0)

Tareas de inferencia:
  1. Filtrado  : P(X_t | e_{1:t})             — estado actual
  2. Predicción: P(X_{t+k} | e_{1:t})         — estado futuro
  3. Suavizado : P(X_k | e_{1:t})  k < t      — estado pasado
  4. Decodificación (Viterbi): arg max P(x_{1:t} | e_{1:t})
"""


# ────────────────── DBN Tabular (2 nodos por paso) ───────────────────

class DBN:
    """
    DBN con un único nodo de estado X y un nodo de evidencia E por paso.
    Generalizable a múltiples variables, pero este esquema ilustra
    los algoritmos fundamentales de forma clara.
    """

    def __init__(self, estados, evidencias,
                 prob_inicial, trans, obs):
        """
        estados      : lista de estados
        evidencias   : lista de posibles observaciones
        prob_inicial : dict {estado: prob}
        trans        : dict {(s, s'): P(s'|s)}
        obs          : dict {(s, e): P(e|s)}
        """
        self.estados      = estados
        self.evidencias   = evidencias
        self.prior        = prob_inicial
        self.trans        = trans     # P(s'|s)
        self.obs          = obs       # P(e|s)

    def P_trans(self, s, s2):
        return self.trans.get((s, s2), 0.0)

    def P_obs(self, s, e):
        return self.obs.get((s, e), 0.0)


# ──────────────────── 1. Filtrado (Forward algorithm) ────────────────

def filtrado(dbn: DBN, evidencias_seq):
    """
    Calcula P(X_t | e_{1:t}) para cada t.
    Retorna lista de distribuciones.
    """
    creencia = dbn.prior.copy()
    historial = [creencia.copy()]

    for e in evidencias_seq:
        # Predicción: P(X_t | e_{1:t-1}) = Σ_{x_{t-1}} P(X_t|x_{t-1})·P(x_{t-1}|e)
        prediccion = {}
        for s2 in dbn.estados:
            prediccion[s2] = sum(dbn.P_trans(s, s2) * creencia[s]
                                 for s in dbn.estados)

        # Actualización: multiplicar por verosimilitud P(e_t | X_t)
        for s in dbn.estados:
            prediccion[s] *= dbn.P_obs(s, e)

        # Normalizar
        total = sum(prediccion.values())
        creencia = {s: v/total for s, v in prediccion.items()} if total > 0 else prediccion
        historial.append(creencia.copy())

    return historial


# ──────────────────── 2. Predicción ──────────────────────────────────

def prediccion(dbn: DBN, creencia_actual, pasos):
    """
    Proyecta la creencia k pasos hacia el futuro sin nuevas evidencias.
    """
    b = creencia_actual.copy()
    for _ in range(pasos):
        b_nueva = {}
        for s2 in dbn.estados:
            b_nueva[s2] = sum(dbn.P_trans(s, s2) * b[s] for s in dbn.estados)
        b = b_nueva
    return b


# ──────────────────── 3. Suavizado (Forward-Backward) ────────────────

def suavizado(dbn: DBN, evidencias_seq):
    """
    Calcula P(X_k | e_{1:T}) para todo k usando Forward-Backward.
    """
    T = len(evidencias_seq)

    # Forward: f[t] = P(X_t | e_{1:t})
    f = [dbn.prior.copy()]
    for e in evidencias_seq:
        pred = {s2: sum(dbn.P_trans(s,s2)*f[-1][s] for s in dbn.estados)
                for s2 in dbn.estados}
        upd  = {s: pred[s]*dbn.P_obs(s,e) for s in dbn.estados}
        total = sum(upd.values())
        f.append({s: v/total for s,v in upd.items()} if total>0 else upd)

    # Backward: b[t] = P(e_{t+1:T} | X_t)  (mensajes hacia atrás)
    b = [{s: 1.0 for s in dbn.estados}]   # b[T] = 1
    for e in reversed(evidencias_seq):
        b_new = {}
        for s in dbn.estados:
            b_new[s] = sum(dbn.P_trans(s,s2) * dbn.P_obs(s2,e) * b[0][s2]
                           for s2 in dbn.estados)
        total = sum(b_new.values())
        b.insert(0, {s: v/total for s,v in b_new.items()} if total>0 else b_new)

    # Combinar: P(X_k | e_{1:T}) ∝ f[k] · b[k]
    resultado = []
    for k in range(T+1):
        smooth = {s: f[k][s]*b[k][s] for s in dbn.estados}
        total  = sum(smooth.values())
        smooth = {s: v/total for s,v in smooth.items()} if total>0 else smooth
        resultado.append(smooth)
    return resultado


# ──────────────────── 4. Viterbi (Decodificación) ────────────────────

def viterbi(dbn: DBN, evidencias_seq):
    """
    Secuencia más probable de estados: arg max P(x_{1:T} | e_{1:T}).
    """
    # Inicializar
    delta = {s: dbn.prior[s] * dbn.P_obs(s, evidencias_seq[0])
             for s in dbn.estados}
    psi   = [{s: None for s in dbn.estados}]

    for e in evidencias_seq[1:]:
        delta_n, psi_n = {}, {}
        for s2 in dbn.estados:
            opciones   = {s: delta[s] * dbn.P_trans(s, s2) for s in dbn.estados}
            mejor_prev = max(opciones, key=opciones.get)
            delta_n[s2] = opciones[mejor_prev] * dbn.P_obs(s2, e)
            psi_n[s2]   = mejor_prev
        delta = delta_n
        psi.append(psi_n)

    # Retroceder
    seq = [max(delta, key=delta.get)]
    for paso in reversed(psi[1:]):
        seq.insert(0, paso[seq[0]])

    return seq, max(delta.values())


# ─────────────────── Demo: Lluvia / Paraguas ──────────────────────────

def demo():
    """
    El agente observa si el guardián lleva paraguas (P=True/False).
    El estado oculto es si llueve (R=True/False).
    P(R_t | R_{t-1}): persiste 70%, cambia 30%.
    P(P_t | R_t):
      Si llueve:     paraguas con 90%
      Si no llueve:  paraguas con 20%
    """
    print("=" * 60)
    print("  Red Bayesiana Dinámica — Lluvia/Paraguas")
    print("=" * 60)

    estados   = ['lluvia', 'sol']
    evidencias= ['paraguas', 'sin_paraguas']

    prior = {'lluvia': 0.5, 'sol': 0.5}

    trans = {
        ('lluvia','lluvia'): 0.7, ('lluvia','sol'): 0.3,
        ('sol',   'sol')  : 0.7, ('sol',   'lluvia'): 0.3,
    }

    obs = {
        ('lluvia','paraguas')    : 0.9,
        ('lluvia','sin_paraguas'): 0.1,
        ('sol',   'paraguas')   : 0.2,
        ('sol',   'sin_paraguas'): 0.8,
    }

    dbn = DBN(estados, evidencias, prior, trans, obs)

    # Secuencia de observaciones
    secuencia = ['paraguas','paraguas','sin_paraguas','paraguas','paraguas']
    print(f"\n  Secuencia de observaciones: {secuencia}\n")

    # Filtrado
    hist_f = filtrado(dbn, secuencia)
    print("  === Filtrado P(X_t | e_{1:t}) ===")
    print(f"  {'t':>3}  {'Lluvia':>10}  {'Sol':>10}  {'Obs'}")
    for t, b in enumerate(hist_f):
        obs_t = secuencia[t-1] if t > 0 else "—"
        print(f"  {t:>3}  {b['lluvia']:>10.4f}  {b['sol']:>10.4f}  {obs_t}")

    # Predicción 3 pasos adelante desde el estado filtrado final
    b_final = hist_f[-1]
    print(f"\n  === Predicción 3 pasos desde t={len(secuencia)} ===")
    for k in range(1, 4):
        b_pred = prediccion(dbn, b_final, k)
        print(f"  t+{k}  lluvia={b_pred['lluvia']:.4f}  sol={b_pred['sol']:.4f}")

    # Suavizado
    hist_s = suavizado(dbn, secuencia)
    print(f"\n  === Suavizado P(X_k | e_{{1:{len(secuencia)}}}) ===")
    for t, b in enumerate(hist_s):
        print(f"  k={t}  lluvia={b['lluvia']:.4f}  sol={b['sol']:.4f}")

    # Viterbi
    seq, prob = viterbi(dbn, secuencia)
    print(f"\n  === Viterbi (secuencia más probable) ===")
    print(f"  {seq}")
    print(f"  Probabilidad de la secuencia (escala): {prob:.6f}")
    print()


if __name__ == "__main__":
    demo()

