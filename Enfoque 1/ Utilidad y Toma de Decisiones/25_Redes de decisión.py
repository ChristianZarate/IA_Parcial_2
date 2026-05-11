"""
25 - Redes de Decisión (Diagramas de Influencia)
=================================================
Una Red de Decisión (Influence Diagram) extiende las Redes Bayesianas
con dos tipos de nodos adicionales:
  - Nodos de Decisión (□): variables que controla el agente.
  - Nodos de Utilidad   (◇): función de utilidad sobre estados y decisiones.

El objetivo es encontrar la política que maximiza la utilidad esperada.

Ejemplo: Decisión de llevar paraguas según el pronóstico del tiempo.
"""

from itertools import product


# ─────────────── Tabla de Probabilidad Condicional ───────────────────

class TPC:
    """Tabla de Probabilidad Condicional P(var | padres)."""

    def __init__(self, variable, valores, padres, tabla):
        """
        variable: nombre
        valores : lista de posibles valores
        padres  : lista de nombres de variables padre
        tabla   : dict { (vals_padres,) -> { valor: prob } }
                  Si no hay padres: { (): { valor: prob } }
        """
        self.variable = variable
        self.valores  = valores
        self.padres   = padres
        self.tabla    = tabla

    def probabilidad(self, valor, evidencia={}):
        clave = tuple(evidencia.get(p) for p in self.padres)
        return self.tabla[clave][valor]


# ─────────────────── Nodo de Utilidad ────────────────────────────────

class NodoUtilidad:
    """Función de utilidad U(padres)."""

    def __init__(self, padres, tabla):
        self.padres = padres
        self.tabla  = tabla   # dict { (vals_padres,) -> utilidad }

    def utilidad(self, evidencia):
        clave = tuple(evidencia.get(p) for p in self.padres)
        return self.tabla.get(clave, 0)


# ─────────────────── Red de Decisión simplificada ────────────────────

class RedDecision:
    """
    Red de Decisión con:
    - Variables de azar (TPC)
    - Una variable de decisión con dominio
    - Un nodo de utilidad
    """

    def __init__(self):
        self.nodos_azar    = {}   # nombre -> TPC
        self.decision_var  = None
        self.decision_vals = []
        self.utilidad      = None  # NodoUtilidad

    def agregar_nodo_azar(self, tpc: TPC):
        self.nodos_azar[tpc.variable] = tpc

    def agregar_decision(self, nombre, valores):
        self.decision_var  = nombre
        self.decision_vals = valores

    def agregar_utilidad(self, nodo: NodoUtilidad):
        self.utilidad = nodo

    # ── Inferencia: utilidad esperada de una decisión ──

    def _orden_topologico(self):
        """Orden simple: nodos sin padres primero."""
        orden = []
        restantes = list(self.nodos_azar.keys())
        asignados = set()
        while restantes:
            for v in list(restantes):
                tpc = self.nodos_azar[v]
                if all(p in asignados or p == self.decision_var
                       for p in tpc.padres):
                    orden.append(v)
                    asignados.add(v)
                    restantes.remove(v)
        return orden

    def utilidad_esperada(self, decision, evidencia={}):
        """E[U | decision, evidencia]. Suma sobre todos los mundos posibles."""
        orden = self._orden_topologico()
        # Construir todas las combinaciones de valores de azar
        azar_vars  = [v for v in orden]
        azar_vals  = [self.nodos_azar[v].valores for v in azar_vars]

        ue_total = 0.0
        for combo in product(*azar_vals):
            mundo = {**evidencia, self.decision_var: decision}
            for var, val in zip(azar_vars, combo):
                mundo[var] = val

            # Probabilidad del mundo
            prob = 1.0
            for var, val in zip(azar_vars, combo):
                tpc = self.nodos_azar[var]
                prob *= tpc.probabilidad(val, mundo)

            # Utilidad del mundo
            u = self.utilidad.utilidad(mundo)
            ue_total += prob * u

        return ue_total

    def decision_optima(self, evidencia={}):
        """Retorna (decisión_óptima, utilidad_esperada_máxima)."""
        mejor_d, mejor_ue = None, float('-inf')
        for d in self.decision_vals:
            ue = self.utilidad_esperada(d, evidencia)
            if ue > mejor_ue:
                mejor_ue = ue
                mejor_d  = d
        return mejor_d, mejor_ue


# ─────────────────────────── demo ────────────────────────────────────

def demo():
    """
    Escenario: ¿Llevar paraguas?
    Variables:
      Lluvia   (azar)  : sí / no        P(Lluvia=sí) = 0.3
      Pronóstico (azar): bueno / malo   P(Pronostico | Lluvia)
      Paraguas (decisión): llevar / no llevar
      Utilidad: depende de (Lluvia, Paraguas)
    """
    print("=" * 55)
    print("  Red de Decisión — ¿Llevar Paraguas?")
    print("=" * 55)

    red = RedDecision()

    # Nodo Lluvia: sin padres
    red.agregar_nodo_azar(TPC(
        variable = 'Lluvia',
        valores  = ['sí', 'no'],
        padres   = [],
        tabla    = { (): {'sí': 0.3, 'no': 0.7} }
    ))

    # Nodo Pronóstico: depende de Lluvia
    red.agregar_nodo_azar(TPC(
        variable = 'Pronostico',
        valores  = ['malo', 'bueno'],
        padres   = ['Lluvia'],
        tabla    = {
            ('sí',): {'malo': 0.8, 'bueno': 0.2},
            ('no',): {'malo': 0.1, 'bueno': 0.9},
        }
    ))

    # Decisión: llevar o no el paraguas
    red.agregar_decision('Paraguas', ['llevar', 'no_llevar'])

    # Utilidad: depende de Lluvia y Paraguas
    #   Llueva y llevo paraguas  →  70   (cómodo pero cargado)
    #   Llueva y no llevo         → -20   (me mojo)
    #   No llueva y llevo         →  20   (innecesario)
    #   No llueva y no llevo      → 100   (perfecta libertad)
    red.agregar_utilidad(NodoUtilidad(
        padres = ['Lluvia', 'Paraguas'],
        tabla  = {
            ('sí', 'llevar')    :  70,
            ('sí', 'no_llevar') : -20,
            ('no', 'llevar')    :  20,
            ('no', 'no_llevar') : 100,
        }
    ))

    # ── Sin evidencia ──
    print("\n  Sin evidencia del pronóstico:")
    for d in ['llevar', 'no_llevar']:
        ue = red.utilidad_esperada(d)
        print(f"    U[{d:12s}] = {ue:6.2f}")
    d_opt, ue_opt = red.decision_optima()
    print(f"  → Decisión óptima: {d_opt}  (UE={ue_opt:.2f})")

    # ── Con evidencia: pronóstico malo ──
    print("\n  Con evidencia: Pronóstico = malo")
    for d in ['llevar', 'no_llevar']:
        ue = red.utilidad_esperada(d, evidencia={'Pronostico': 'malo'})
        print(f"    U[{d:12s}] = {ue:6.2f}")
    d_opt, ue_opt = red.decision_optima(evidencia={'Pronostico': 'malo'})
    print(f"  → Decisión óptima: {d_opt}  (UE={ue_opt:.2f})")

    # ── Con evidencia: pronóstico bueno ──
    print("\n  Con evidencia: Pronóstico = bueno")
    for d in ['llevar', 'no_llevar']:
        ue = red.utilidad_esperada(d, evidencia={'Pronostico': 'bueno'})
        print(f"    U[{d:12s}] = {ue:6.2f}")
    d_opt, ue_opt = red.decision_optima(evidencia={'Pronostico': 'bueno'})
    print(f"  → Decisión óptima: {d_opt}  (UE={ue_opt:.2f})")
    print()


if __name__ == "__main__":
    demo()

