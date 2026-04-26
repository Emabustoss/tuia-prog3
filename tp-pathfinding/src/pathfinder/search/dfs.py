from ..models.grid import Grid
from ..models.frontier import StackFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node


class DepthFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """
        Encuentra un camino utilizando Búsqueda en Profundidad (DFS).
        
        Explora un camino hasta llegar a un callejón sin salida antes de retroceder 
        (backtracking). Utiliza una estructura de datos tipo Pila (LIFO) para la frontera. 
        Es eficiente en memoria, pero NO garantiza encontrar el camino más corto (no es óptimo).
        """
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)
        # Frontera tipo Pila
        frontera = StackFrontier()
        frontera.add(root)
        # Diccionario para registrar los estados que ya fueron visitados
        expanded = dict()

        if grid.objective_test(root.state):
            return Solution(root, expanded)

        while True:
            # Si la frontera se vacía y no encontramos la meta, no hay solución
            if frontera.is_empty():   # o sin () si en tu clase es atributo
                return NoSolution(expanded)
            # Extraemos el nodo más reciente
            n = frontera.remove()
            # Si el estado ya fue expandido por otra ruta, lo ignoramos
            if n.state in expanded:
                continue
            # Marcamos el estado como expandido al sacarlo de la pila
            expanded[n.state] = True
            # Expansión de vecinos
            for a in grid.actions(n.state):
                s = grid.result(n.state, a)
                # Solo procesamos el estado si no fue expandido previamente
                if s not in expanded:
                    son = Node(
                        "",
                        state=s,
                        cost=n.cost + grid.individual_cost(n.state, a),
                        parent=n,
                        action=a,
                    )
                    # Test de objetivo temprano:verificamos si el hijo es la meta
                    if grid.objective_test(s):
                        return Solution(son, expanded)
                    # Lo agregamos a la pila para explorarlo en profundidad
                    frontera.add(son)


        return NoSolution(expanded)