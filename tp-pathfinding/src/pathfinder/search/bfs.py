from ..models.grid import Grid
from ..models.frontier import QueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node

class BreadthFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """
        Encuentra el camino más corto utilizando Búsqueda en Anchura (BFS).
        
        Explora el grafo nivel por nivel, garantizando encontrar la solución óptima 
        (el camino con menos pasos). Utiliza una estructura de datos tipo Cola (FIFO) 
        para gestionar la frontera de exploración.
        """
        # Creamos el nodo raíz con estado inicial y costo 0
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)
        
        # Diccionario de Alcanzados
        reached = {}
        
        # Frontera tipo Cola
        frontera = QueueFrontier()
        
        # Agregamos la raíz a la frontera y marcamos el estado como alcanzado
        frontera.add(root)
        reached[root.state] = True
        
        #Comprobamos si el estado inicial es el Optimo, en caso contrario continua
        if grid.objective_test(root.state):
            return Solution(root, reached)

        while True:
            # Si la frontera se vacía y no encontramos la meta, no hay solución
            if frontera.is_empty():
                return NoSolution(reached)
            # Extraemos el nodo más antiguo de la frontera
            n = frontera.remove()
            # Expandimos el nodo explorando todas sus acciones posibles
            for a in grid.actions(n.state):
                s = grid.result(n.state, a)
                # Solo procesamos el estado si no fue visitado previamente
                if s not in reached:
                    son = Node(
                        "",
                        state=s,
                        cost=n.cost + grid.individual_cost(n.state, a),
                        parent=n,
                        action=a,
                    )
                    # Test de objetivo temprano:verificamos si el hijo es la meta
                    if grid.objective_test(s):
                        return Solution(son, reached)
                    # Lo marcamos como visitado y lo agregamos a la cola para explorarlo
                    reached[s] = True
                    frontera.add(son)

        return NoSolution(reached)