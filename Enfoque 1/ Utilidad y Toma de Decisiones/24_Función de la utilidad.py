"""
24 - Teoría de la Utilidad: Función de Utilidad
================================================
La teoría de la utilidad permite modelar preferencias de un agente
racional bajo incertidumbre. Se basa en los axiomas de von Neumann–
Morgenstern: los agentes maximizan la *utilidad esperada*.

Conceptos clave:
  - Lotería: distribución de probabilidad sobre resultados.
  - Utilidad esperada: E[U] = Σ p_i · U(x_i)
  - Actitud ante el riesgo: neutral, averso, amante del riesgo.
  - Función de utilidad monetaria: lineal, logarítmica, exponencial.
"""

import math
import random


# ─────────────────── funciones de utilidad comunes ───────────────────

def utilidad_lineal(x):
    """Agente neutral al riesgo."""
    return x

def utilidad_logaritmica(x, base=math.e):
    """Agente averso al riesgo (cóncava)."""
    if x <= 0:
        return float('-inf')
    return math.log(x, base)

def utilidad_exponencial(x, r=0.001):
    """Utilidad con aversión al riesgo constante (CARA). r>0 → averso."""
    return 1 - math.exp(-r * x)

def utilidad_cuadratica(x, b=0.001):
    """Cóncava para valores pequeños (averso al riesgo)."""
    return x - b * x**2


# ─────────────────────── clase Lotería ───────────────────────────────

class Loteria:
    """Distribución de probabilidad sobre resultados numéricos."""

    def __init__(self, resultados, probabilidades):
        """
        resultados    : lista de valores monetarios
        probabilidades: lista de probs (deben sumar 1)
        """
        assert abs(sum(probabilidades) - 1.0) < 1e-6, "Las probs deben sumar 1."
        self.resultados      = resultados
        self.probabilidades  = probabilidades

    def valor_esperado(self):
        return sum(p * x for p, x in zip(self.probabilidades, self.resultados))

    def utilidad_esperada(self, funcion_utilidad):
        return sum(p * funcion_utilidad(x)
                   for p, x in zip(self.probabilidades, self.resultados))

    def equivalente_certeza(self, funcion_utilidad, funcion_inversa,
                             rango=(0, 10_000), pasos=10_000):
        """
        El valor cierto CE tal que U(CE) = E[U(Lotería)].
        Se busca numéricamente en el rango dado.
        """
        ue = self.utilidad_esperada(funcion_utilidad)
        for i in range(pasos + 1):
            x = rango[0] + i * (rango[1] - rango[0]) / pasos
            if funcion_utilidad(x) >= ue:
                return x
        return rango[1]

    def simular(self, n=10_000):
        """Simulación Monte Carlo del valor promedio."""
        total = 0
        for _ in range(n):
            r = random.random()
            acum = 0
            for p, x in zip(self.probabilidades, self.resultados):
                acum += p
                if r <= acum:
                    total += x
                    break
        return total / n

    def __repr__(self):
        items = ", ".join(f"${x:.0f}@{p:.0%}"
                          for x, p in zip(self.resultados, self.probabilidades))
        return f"Lotería({items})"


# ──────────────────── comparación de actitudes ───────────────────────

def comparar_actitudes(loteria: Loteria):
    print(f"\n  Lotería: {loteria}")
    print(f"  Valor Esperado: ${loteria.valor_esperado():,.2f}")

    funciones = {
        "Neutral (lineal)":        utilidad_lineal,
        "Averso (logarítmica)":    utilidad_logaritmica,
        "Amante (exponencial inv)": lambda x: -utilidad_exponencial(x, r=-0.001),
    }

    print()
    for nombre, f in funciones.items():
        ue = loteria.utilidad_esperada(f)
        print(f"  [{nombre}]")
        print(f"    Utilidad Esperada : {ue:.4f}")


# ──────────────────────── prima de riesgo ────────────────────────────

def prima_riesgo(loteria: Loteria, funcion_utilidad, funcion_inv_aprox):
    """Prima de riesgo = VE - Equivalente de Certeza."""
    ve = loteria.valor_esperado()
    ce = loteria.equivalente_certeza(funcion_utilidad, funcion_inv_aprox)
    prima = ve - ce
    return ve, ce, prima


# ─────────────────────────── demo ────────────────────────────────────

def demo():
    print("=" * 55)
    print("  Teoría de la Utilidad: Función de Utilidad")
    print("=" * 55)

    random.seed(0)

    # Lotería 1: Ganar $10,000 con 50% o $0 con 50%
    l1 = Loteria([10_000, 0], [0.5, 0.5])
    comparar_actitudes(l1)

    # Prima de riesgo con utilidad logarítmica
    ve, ce, prima = prima_riesgo(
        l1,
        lambda x: utilidad_logaritmica(x + 1),   # +1 para evitar log(0)
        None
    )
    print(f"\n  Prima de Riesgo (averso logarítmico):")
    print(f"    Valor Esperado       : ${ve:,.2f}")
    print(f"    Equivalente Certeza  : ${ce:,.2f}")
    print(f"    Prima de Riesgo      : ${prima:,.2f}")

    # Lotería 2: Multipremiada
    print()
    l2 = Loteria([50_000, 10_000, 1_000, 0], [0.05, 0.20, 0.35, 0.40])
    comparar_actitudes(l2)

    # Simulación Monte Carlo
    sim = l2.simular(100_000)
    print(f"\n  Simulación Monte Carlo (100k muestras): ${sim:,.2f}")
    print(f"  Valor Esperado real                  : ${l2.valor_esperado():,.2f}")

    # Tabla de utilidades
    print("\n  Tabla comparativa de funciones de utilidad:")
    print(f"  {'Valor':>8}  {'Lineal':>10}  {'Log':>10}  {'Exp(r=0.001)':>14}")
    print("  " + "-" * 47)
    for x in [0, 500, 1000, 5000, 10000]:
        ul = utilidad_lineal(x)
        ulo = utilidad_logaritmica(x + 1)
        ue = utilidad_exponencial(x)
        print(f"  {x:>8,}  {ul:>10,.2f}  {ulo:>10.4f}  {ue:>14.6f}")
    print()


if __name__ == "__main__":
    demo()

