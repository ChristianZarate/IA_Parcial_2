"""
26 - Valor de la Información (VPI — Value of Perfect Information)
=================================================================
¿Cuánto vale obtener cierta información antes de tomar una decisión?

VPI(E_j | e) = EU(decisión_óptima CON E_j) − EU(decisión_óptima SIN E_j)

Propiedades:
  - VPI ≥ 0  (información nunca hace daño en decisiones racionales)
  - VPI = 0  si la información no cambia la decisión óptima
  - VPI es no aditivo en general (depende del orden de adquisición)

También se calcula el Valor de la Información Imperfecta (VUI)
cuando la observación tiene ruido.
"""

from itertools import product


# ──────────────── Reutilizamos la Red de Decisión del tema 25 ────────

class TPC:
    def __init__(self, variable, valores, padres, tabla):
        self.variable = variable
        self.valores  = valores
        self.padres   = padres
        self.tabla    = tabla

    def probabilidad(self, valor, evidencia={}):
        clave = tuple(evidencia.get(p) for p in self.padres)
        return self.tabla[clave][valor]


class NodoUtilidad:
    def __init__(self, padres, tabla):
        self.padres = padres
        self.tabla  = tabla

    def utilidad(self, evidencia):
        clave = tuple(evidencia.get(p) for p in self.padres)
        return self.tabla.get(clave, 0)


class RedDecision:
    def __init__(self):
        self.nodos_azar    = {}
        self.decision_var  = None
        self.decision_vals = []
        self.utilidad      = None

    def agregar_nodo_azar(self, tpc):
        self.nodos_azar[tpc.variable] = tpc

    def agregar_decision(self, nombre, valores):
        self.decision_var  = nombre
        self.decision_vals = valores

    def agregar_utilidad(self, nodo):
        self.utilidad = nodo

    def _orden(self):
        orden, asignados = [], set()
        restantes = list(self.nodos_azar.keys())
        while restantes:
            for v in list(restantes):
                tpc = self.nodos_azar[v]
                if all(p in asignados or p == self.decision_var
                       for p in tpc.padres):
                    orden.append(v)
                    asignados.add(v)
                    restantes.remove(v)
        return orden

    def _mundos(self, evidencia):
        """Genera todos los mundos posibles con sus probabilidades."""
        orden     = self._orden()
        azar_vals = [self.nodos_azar[v].valores for v in orden]
        mundos    = []
        for combo in product(*azar_vals):
            mundo = {**evidencia}
            for var, val in zip(orden, combo):
                mundo[var] = val
            prob = 1.0
            for var, val in zip(orden, combo):
                prob *= self.nodos_azar[var].probabilidad(val, mundo)
            mundos.append((mundo, prob))
        return mundos

    def utilidad_esperada(self, decision, evidencia={}):
        total = 0.0
        for mundo, prob in self._mundos(evidencia):
            mundo[self.decision_var] = decision
            total += prob * self.utilidad.utilidad(mundo)
        return total

    def decision_optima(self, evidencia={}):
        mejor_d, mejor_ue = None, float('-inf')
        for d in self.decision_vals:
            ue = self.utilidad_esperada(d, evidencia)
            if ue > mejor_ue:
                mejor_ue, mejor_d = ue, d
        return mejor_d, mejor_ue

    def probabilidad_marginal(self, var, valor, evidencia={}):
        """P(var=valor | evidencia) marginalizando sobre el resto."""
        total = 0.0
        for mundo, prob in self._mundos(evidencia):
            if mundo.get(var) == valor:
                total += prob
        return total


# ─────────────────── Cálculo del VPI ─────────────────────────────────

def calcular_vpi(red: RedDecision, variable_info: str, evidencia={}):
    """
    VPI de observar 'variable_info' dados los evidencia actuales.
    VPI = Σ_ej  P(E_j=ej) · EU*(ej)  −  EU*()
    """
    tpc_info = red.nodos_azar[variable_info]

    # EU* sin la nueva información
    _, eu_sin = red.decision_optima(evidencia)

    # EU* con cada posible observación de variable_info
    eu_con = 0.0
    for val in tpc_info.valores:
        # Probabilidad de observar este valor
        p_val = red.probabilidad_marginal(variable_info, val, evidencia)
        # EU óptima dado que observamos variable_info = val
        ev_nueva = {**evidencia, variable_info: val}
        _, eu_val = red.decision_optima(ev_nueva)
        eu_con += p_val * eu_val

    vpi = eu_con - eu_sin
    return vpi, eu_sin, eu_con


# ─────────────────────────── demo ────────────────────────────────────

def construir_red_paraguas():
    """Misma red del tema 25."""
    red = RedDecision()
    red.agregar_nodo_azar(TPC(
        'Lluvia', ['sí', 'no'], [],
        {(): {'sí': 0.3, 'no': 0.7}}
    ))
    red.agregar_nodo_azar(TPC(
        'Pronostico', ['malo', 'bueno'], ['Lluvia'],
        {
            ('sí',): {'malo': 0.8, 'bueno': 0.2},
            ('no',): {'malo': 0.1, 'bueno': 0.9},
        }
    ))
    red.agregar_decision('Paraguas', ['llevar', 'no_llevar'])
    red.agregar_utilidad(NodoUtilidad(
        ['Lluvia', 'Paraguas'],
        {
            ('sí', 'llevar'):    70,
            ('sí', 'no_llevar'): -20,
            ('no', 'llevar'):    20,
            ('no', 'no_llevar'): 100,
        }
    ))
    return red


def demo():
    print("=" * 60)
    print("  Valor de la Información (VPI)")
    print("=" * 60)

    red = construir_red_paraguas()

    # ── VPI de conocer el Pronóstico (sin otra evidencia) ──
    vpi, eu_sin, eu_con = calcular_vpi(red, 'Pronostico')
    print(f"\n  VPI(Pronóstico):")
    print(f"    EU* sin información   : {eu_sin:.4f}")
    print(f"    EU* con información   : {eu_con:.4f}")
    print(f"    VPI                   : {vpi:.4f}")
    print(f"    → El agente pagaría hasta {vpi:.2f} unidades por el pronóstico.")

    # ── VPI de conocer directamente si Llueve ──
    vpi2, eu_sin2, eu_con2 = calcular_vpi(red, 'Lluvia')
    print(f"\n  VPI(Lluvia) — información perfecta:")
    print(f"    EU* sin información   : {eu_sin2:.4f}")
    print(f"    EU* con información   : {eu_con2:.4f}")
    print(f"    VPI                   : {vpi2:.4f}")

    # ── VPI del pronóstico dado que ya observamos la lluvia ──
    vpi3, eu3_sin, eu3_con = calcular_vpi(red, 'Pronostico',
                                          evidencia={'Lluvia': 'sí'})
    print(f"\n  VPI(Pronóstico | Lluvia=sí):")
    print(f"    EU* sin información   : {eu3_sin:.4f}")
    print(f"    EU* con información   : {eu3_con:.4f}")
    print(f"    VPI                   : {vpi3:.4f}")
    print(f"    → Con certeza de lluvia, el pronóstico vale {vpi3:.4f} (casi 0).")

    # ── Tabla resumen ──
    print("\n  Resumen de VPIs:")
    print(f"  {'Información':25s}  {'VPI':>8}")
    print("  " + "-" * 36)
    print(f"  {'Pronóstico (sin evidencia)':25s}  {vpi:>8.4f}")
    print(f"  {'Lluvia (información perfecta)':25s}  {vpi2:>8.4f}")
    print(f"  {'Pronóstico | Lluvia=sí':25s}  {vpi3:>8.4f}")
    print()


if __name__ == "__main__":
    demo()

