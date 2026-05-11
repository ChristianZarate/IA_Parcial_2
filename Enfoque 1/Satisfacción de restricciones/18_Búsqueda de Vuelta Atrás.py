# Búsqueda de Vuelta Atrás (Backtracking Search)
# Algoritmo sistemático para CSPs: asigna variables una a una y retrocede
# cuando detecta inconsistencia. Incluye heurísticas de ordenamiento.

class CSP:
    def __init__(self, variables, dominios, restricciones):
        self.variables = variables
        self.dominios = {v: list(d) for v, d in dominios.items()}
        self.restricciones = restricciones
        self.nodos_expandidos = 0

    def es_consistente(self, var, valor, asignacion):
        for v1, v2, fn in self.restricciones:
            if v1 == var and v2 in asignacion:
                if not fn(valor, asignacion[v2]):
                    return False
            if v2 == var and v1 in asignacion:
                if not fn(asignacion[v1], valor):
                    return False
        return True

    def vecinos_de(self, var):
        vecinos = set()
        for v1, v2, _ in self.restricciones:
            if v1 == var: vecinos.add(v2)
            if v2 == var: vecinos.add(v1)
        return vecinos


# ============================================================
# Heurísticas de selección de variable y valor
# ============================================================

def seleccion_simple(csp, asignacion):
    """Elige la primera variable sin asignar."""
    return next(v for v in csp.variables if v not in asignacion)

def mrv(csp, asignacion):
    """MRV (Minimum Remaining Values): elige la variable con menos valores legales."""
    sin_asignar = [v for v in csp.variables if v not in asignacion]
    return min(sin_asignar, key=lambda v: len([
        val for val in csp.dominios[v]
        if csp.es_consistente(v, val, asignacion)
    ]))

def lcv(csp, var, asignacion):
    """LCV (Least Constraining Value): ordena valores que menos restringen a vecinos."""
    def conteo_eliminados(valor):
        total = 0
        for vecino in csp.vecinos_de(var):
            if vecino not in asignacion:
                for v_val in csp.dominios[vecino]:
                    prueba = {**asignacion, var: valor}
                    if not csp.es_consistente(vecino, v_val, prueba):
                        total += 1
        return total
    return sorted(csp.dominios[var], key=conteo_eliminados)


# ============================================================
# Backtracking con heurísticas
# ============================================================

def backtracking(csp, asignacion=None, usar_mrv=True, usar_lcv=True):
    if asignacion is None:
        asignacion = {}

    if len(asignacion) == len(csp.variables):
        return asignacion

    csp.nodos_expandidos += 1

    var = mrv(csp, asignacion) if usar_mrv else seleccion_simple(csp, asignacion)
    valores = lcv(csp, var, asignacion) if usar_lcv else csp.dominios[var]

    print(f"  Asignando {var}, opciones: {valores}")

    for valor in valores:
        if csp.es_consistente(var, valor, asignacion):
            asignacion[var] = valor
            resultado = backtracking(csp, asignacion, usar_mrv, usar_lcv)
            if resultado is not None:
                return resultado
            del asignacion[var]
            print(f"  Backtrack en {var}={valor}")

    return None


if __name__ == "__main__":
    # Problema de las N-Reinas (N=6)
    N = 6
    variables = list(range(N))
    dominios = {i: list(range(N)) for i in range(N)}

    def no_atacan(q1, q2, col1, col2):
        return q1 != q2 and abs(q1 - q2) != abs(col1 - col2)

    restricciones = []
    for i in range(N):
        for j in range(i + 1, N):
            restricciones.append((i, j, lambda a, b, ci=i, cj=j: no_atacan(a, b, ci, cj)))

    print(f"=== Backtracking con MRV y LCV ===")
    print(f"Problema: {N}-Reinas\n")

    csp = CSP(variables, dominios, restricciones)
    solucion = backtracking(csp, usar_mrv=True, usar_lcv=True)

    if solucion:
        print(f"\nSolución (columna de cada reina por fila):")
        for fila, col in sorted(solucion.items()):
            tablero = ['.' if c != col else 'Q' for c in range(N)]
            print(f"  Fila {fila}: {' '.join(tablero)}")
        print(f"\nNodos expandidos: {csp.nodos_expandidos}")
    else:
        print("Sin solución.")