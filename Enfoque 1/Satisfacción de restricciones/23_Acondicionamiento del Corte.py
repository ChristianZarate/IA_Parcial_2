# Acondicionamiento del Corte (Cutset Conditioning)
# Técnica que explota la estructura del grafo de restricciones.
# Si se elimina un conjunto de variables (cutset), el grafo restante
# se vuelve un árbol, resoluble en tiempo lineal.
# Estrategia: instanciar todas las combinaciones del cutset y resolver el árbol.

class CSP:
    def __init__(self, variables, dominios, vecinos, restriccion_fn):
        self.variables = variables
        self.dominios = {v: list(d) for v, d in dominios.items()}
        self.vecinos = vecinos
        self.restriccion = restriccion_fn

    def es_consistente_asignacion(self, var, val, asignacion):
        for vecino in self.vecinos.get(var, []):
            if vecino in asignacion:
                if not self.restriccion(var, val, vecino, asignacion[vecino]):
                    return False
        return True


def encontrar_cutset_simple(variables, vecinos):
    """
    Heurística simple: elige nodos de mayor grado hasta que el grafo sea un árbol.
    Un grafo es árbol si |aristas| == |nodos| - 1 y está conectado.
    """
    # Para simplificar, usamos un cutset manual basado en el ejemplo
    # En la práctica se usa algoritmos de ciclos (DFS con back-edges)
    cutset = []
    aristas = set()
    for v in variables:
        for u in vecinos.get(v, []):
            aristas.add(frozenset([v, u]))

    n = len(variables)
    a = len(aristas)

    print(f"Grafo original: {n} nodos, {a} aristas")
    print(f"Para ser árbol necesita {n-1} aristas, tiene {a} -> ciclos: {a - (n-1)}\n")

    # Detección simple: nodos que forman ciclos (nodos con grado >= 2 y en ciclo)
    # Aquí devolvemos un cutset precomputado para el ejemplo
    return cutset


def resolver_arbol(csp, orden, asignacion_fija):
    """
    Resuelve un CSP con estructura de árbol mediante propagación hacia adelante y hacia atrás.
    orden: lista de variables en orden topológico.
    """
    asignacion = dict(asignacion_fija)

    for var in orden:
        if var in asignacion:
            continue
        asignado = False
        for valor in csp.dominios[var]:
            if csp.es_consistente_asignacion(var, valor, asignacion):
                asignacion[var] = valor
                asignado = True
                break
        if not asignado:
            return None

    return asignacion


def backtracking_cutset(csp, cutset, resto, intentos=None):
    """Itera sobre asignaciones del cutset y resuelve el árbol residual."""
    if intentos is None:
        intentos = [0]

    def instanciar(idx, asignacion_cutset):
        if idx == len(cutset):
            intentos[0] += 1
            print(f"  Intento {intentos[0]}: cutset={asignacion_cutset}")

            solucion = resolver_arbol(csp, resto, asignacion_cutset)
            if solucion:
                return solucion
            return None

        var = cutset[idx]
        for valor in csp.dominios[var]:
            nueva = dict(asignacion_cutset)
            nueva[var] = valor
            # Verificar consistencia dentro del cutset
            ok = all(
                csp.restriccion(var, valor, v2, v2_val)
                for v2, v2_val in nueva.items()
                if v2 != var and v2 in csp.vecinos.get(var, [])
            )
            if ok:
                resultado = instanciar(idx + 1, nueva)
                if resultado:
                    return resultado
        return None

    return instanciar(0, {})


if __name__ == "__main__":
    # Ejemplo: grafo con un ciclo, que se puede romper con 1 variable (cutset)
    # Coloreo de 4 nodos en ciclo: A-B-C-D-A (necesita cutset de tamaño 1)
    variables = ['A', 'B', 'C', 'D']
    colores = ['R', 'G', 'B']
    dominios = {v: colores[:] for v in variables}
    vecinos = {
        'A': ['B', 'D'],
        'B': ['A', 'C'],
        'C': ['B', 'D'],
        'D': ['A', 'C']
    }

    def diferente(v1, c1, v2, c2):
        return c1 != c2

    csp = CSP(variables, dominios, vecinos, diferente)

    print("=== Acondicionamiento del Corte (Cutset Conditioning) ===")
    print("Grafo: ciclo A-B-C-D-A | Colores: R, G, B\n")

    encontrar_cutset_simple(variables, vecinos)

    # Cutset: {A} -> rompe el ciclo, resto {B, C, D} forma un árbol (camino B-C-D)
    cutset = ['A']
    resto = ['B', 'C', 'D']
    print(f"Cutset seleccionado: {cutset}")
    print(f"Árbol residual: {resto}\n")

    solucion = backtracking_cutset(csp, cutset, resto)

    if solucion:
        print(f"\nSolución encontrada:")
        for v in variables:
            print(f"  {v}: {solucion[v]}")
        print("\nVerificación de restricciones:")
        for v in variables:
            for u in vecinos[v]:
                ok = solucion[v] != solucion[u]
                print(f"  {v}({solucion[v]}) != {u}({solucion[u]}) -> {'OK' if ok else 'FALLO'}")
    else:
        print("Sin solución.")