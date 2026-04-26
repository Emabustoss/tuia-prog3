from ..models.grid import Grid
from ..models.frontier import PriorityQueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node

class GreedyBestFirstSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """
        Encuentra un camino utilizando GBFS.
        
        Es un algoritmo informado que utiliza una Cola de Prioridad. Para decidir 
        que nodo explorar, se basa unicamente en la heurística h(n) (la distancia 
        estimada a la meta). Es muy rapido, pero no garantiza encontrar el camino 
        más corto, ya que ignora el costo real acumulado g(n) y puede dejarse engañar 
        por obstaculos.
        """
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)
        reached = {}
        # Frontera ordenada por prioridad (menor h(n))
        frontera = PriorityQueueFrontier()
        # La prioridad es solo la heurística 
        frontera.add(root, grid.heuristic(root.state)) 

        while not frontera.is_empty():
            # Extraemos el nodo que parece estar mas cerca de la meta
            n = frontera.pop()
            # Test de objetivo al extraer de la frontera
            if grid.objective_test(n.state):
                return Solution(n, reached)
            # Si el estado ya fue visitado, lo ignoramos para evitar bucles
            if n.state in reached:
                continue
            reached[n.state] = True
            # Expansion de vecinos
            for a in grid.actions(n.state):
                s = grid.result(n.state, a)
                if s not in reached:
                    son = Node(
                        "",
                        state=s,
                        # Calculamos el costo real (g) por si hace falta para el Solution
                        # pero no lo usamos para decidir la prioridad en esta busqueda
                        cost=n.cost + grid.individual_cost(n.state, a),
                        parent=n,
                        action=a,
                    )
                    # Prioridad = solo estimación (h). Ignoramos el costo
                    frontera.add(son, grid.heuristic(s)) 

        return NoSolution(reached)