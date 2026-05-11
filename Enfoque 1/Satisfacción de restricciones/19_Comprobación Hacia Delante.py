# Comprobación Hacia Delante (Forward Checking)
# Al asignar un valor a una variable, elimina valores inconsistentes
# de los dominios de las variables vecinas no asignadas.
# Detecta fallos más pronto que el backtracking simple.

class CSP:
    def __init__(self, variables, dominios, vecinos, restriccion_fn):
        self.variables = variables
        self.dominios = {v: list(d) for v, d in dominios.items()}
        self.vecinos = vecinos       # dict: var -> [vars relacionadas]
        self.restriccion = restriccion_fn  # f(var1, val1, var2, val2) -> bool
        self.nodos = 0
        self.podas = 0

    def eliminar(self, var, val, eliminados):
        self.dominios[var].remove(val)
        self.podas += 1
        eliminados.setdefault(var, []).append(val)

    def restaurar(self, eliminados):
        for var, vals in eliminados.items():
            self.dominios[var].extend(vals)


def forward_checking(csp, var, val, asignacion):
    """Elimina valores inconsistentes de los vecinos no asignados."""
    eliminados = {}
    for vecino in csp.vecinos.get(var, []):
        if vecino not in asignacion:
            for v_val in csp.dominios[vecino][:]:
                if not csp.restriccion(var, val, vecino, v_val):
                    csp.eliminar(vecino, v_val, eliminados)
            if not csp.dominios[vecino]:  # dominio vacío = fallo
                return False, eliminados
    return True, eliminados


def backtracking_fc(csp, asignacion=None):
    if asignacion is None:
        asignacion = {}

    if len(asignacion) == len(csp.variables):
        return asignacion

    # MRV
    sin_asignar = [v for v in csp.variables if v not in asignacion]
    var = min(sin_asignar, key=lambda v: len(csp.dominios[v]))

    for valor in list(csp.dominios[var]):
        csp.nodos += 1
        print(f"  Probando {var}={valor} | dominios: { {v: csp.dominios[v] for v in sin_asignar} }")

        asignacion[var] = valor
        ok, eliminados = forward_checking(csp, var, valor, asignacion)

        if ok:
            resultado = backtracking_fc(csp, asignacion)
            if resultado is not None:
                return resultado

        csp.restaurar(eliminados)
        del asignacion[var]

    return None


if __name__ == "__main__":
    # Coloreo de mapa: Australia
    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    colores = ['R', 'G', 'B']
    dominios = {v: colores[:] for v in variables}

    adyacencias = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'Q'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
        'Q':  ['NT', 'SA', 'NSW'],
        'NSW': ['Q', 'SA', 'V'],
        'V':  ['SA', 'NSW'],
        'T':  []
    }

    def no_igual(v1, val1, v2, val2):
        return val1 != val2

    csp = CSP(variables, dominios, adyacencias, no_igual)

    print("=== Comprobación Hacia Delante (Forward Checking) ===")
    print("Coloreo del mapa de Australia\n")

    solucion = backtracking_fc(csp)

    if solucion:
        print(f"\nSolución:")
        for v, c in solucion.items():
            print(f"  {v}: {c}")
        print(f"\nNodos visitados: {csp.nodos}")
        print(f"Valores podados: {csp.podas}")
    else:
        print("Sin solución.")