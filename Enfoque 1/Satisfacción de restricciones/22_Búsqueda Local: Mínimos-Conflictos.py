"""
22 - Búsqueda Local: Mínimos-Conflictos (Min-Conflicts)
=========================================================
Algoritmo de búsqueda local para CSPs.
- Parte de una asignación completa (puede tener conflictos).
- En cada paso escoge una variable conflictiva al azar y le
  asigna el valor que minimice el número de conflictos.
- Muy eficiente en problemas grandes (e.g. N-Reinas).
"""

import random


# ─────────────────────────── utilidades ────────────────────────────

def contar_conflictos(asignacion, var, valor, restricciones):
    """Cuenta cuántas restricciones viola 'var=valor' con la asignación actual."""
    conflictos = 0
    for (v1, v2), restriccion in restricciones.items():
        if v1 == var and v2 in asignacion:
            if not restriccion(valor, asignacion[v2]):
                conflictos += 1
        elif v2 == var and v1 in asignacion:
            if not restriccion(asignacion[v1], valor):
                conflictos += 1
    return conflictos


def variables_conflictivas(asignacion, restricciones):
    """Devuelve lista de variables que tienen al menos un conflicto."""
    conflictivas = []
    for var in asignacion:
        if contar_conflictos(asignacion, var, asignacion[var], restricciones) > 0:
            conflictivas.append(var)
    return conflictivas


# ──────────────────────── algoritmo principal ────────────────────────

def min_conflicts(variables, dominios, restricciones, max_pasos=1000):
    """
    Búsqueda Local por Mínimos-Conflictos.
    Retorna asignación solución o None si no la encontró.
    """
    # 1. Asignación inicial aleatoria completa
    asignacion = {v: random.choice(dominios[v]) for v in variables}

    for paso in range(max_pasos):
        # 2. Comprobar si es solución
        conflictivas = variables_conflictivas(asignacion, restricciones)
        if not conflictivas:
            print(f"  ✓ Solución encontrada en el paso {paso}.")
            return asignacion

        # 3. Elegir variable conflictiva al azar
        var = random.choice(conflictivas)

        # 4. Asignar valor con menos conflictos
        min_conf = float('inf')
        mejores_valores = []
        for valor in dominios[var]:
            c = contar_conflictos(asignacion, var, valor, restricciones)
            if c < min_conf:
                min_conf = c
                mejores_valores = [valor]
            elif c == min_conf:
                mejores_valores.append(valor)

        asignacion[var] = random.choice(mejores_valores)

    print("  ✗ No se encontró solución dentro del límite de pasos.")
    return None


# ──────────────────────── N-Reinas (demo) ────────────────────────────

def demo_n_reinas(n=8):
    """
    Problema de las N-Reinas usando Min-Conflicts.
    Variables: columnas (0..n-1); valor = fila donde está la reina.
    Restricción: ninguna reina comparte fila ni diagonal.
    """
    print(f"\n{'='*50}")
    print(f"  Problema de las {n}-Reinas")
    print(f"{'='*50}")

    variables = list(range(n))
    dominios  = {v: list(range(n)) for v in variables}

    # Restricción: para columnas c1 y c2, las reinas no deben atacarse
    def no_ataca(fila1, fila2, col1=None, col2=None):
        # Se llamará con el índice de columna capturado en el closure
        return True  # placeholder; usamos versión con closure abajo

    restricciones = {}
    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            def hacer_restriccion(col1, col2):
                def restriccion(f1, f2):
                    return f1 != f2 and abs(f1 - f2) != abs(col1 - col2)
                return restriccion
            restricciones[(c1, c2)] = hacer_restriccion(c1, c2)

    solucion = min_conflicts(variables, dominios, restricciones, max_pasos=10_000)

    if solucion:
        # Imprimir tablero
        tablero = [['.' for _ in range(n)] for _ in range(n)]
        for col, fila in solucion.items():
            tablero[fila][col] = 'Q'
        print()
        for fila in tablero:
            print("  " + " ".join(fila))
        print()


# ──────────────────────── demo coloreado de mapa ─────────────────────

def demo_coloreo_mapa():
    """
    Coloreo del mapa de Australia (WA, NT, SA, Q, NSW, V, T).
    Restricción: territorios adyacentes deben tener diferente color.
    """
    print(f"\n{'='*50}")
    print("  Coloreo de Mapa — Australia")
    print(f"{'='*50}")

    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    colores = ['Rojo', 'Verde', 'Azul']
    dominios = {v: colores[:] for v in variables}

    adyacencias = [
        ('WA', 'NT'), ('WA', 'SA'),
        ('NT', 'SA'), ('NT', 'Q'),
        ('SA', 'Q'),  ('SA', 'NSW'), ('SA', 'V'),
        ('Q', 'NSW'), ('NSW', 'V'),
    ]

    restricciones = {}
    for v1, v2 in adyacencias:
        restricciones[(v1, v2)] = lambda c1, c2: c1 != c2

    solucion = min_conflicts(variables, dominios, restricciones, max_pasos=1000)

    if solucion:
        print()
        for var, color in solucion.items():
            print(f"  {var:4s} → {color}")
        print()


# ─────────────────────────────── main ────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    demo_n_reinas(8)
    demo_coloreo_mapa()

