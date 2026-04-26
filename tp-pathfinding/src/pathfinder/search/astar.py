from ..models.grid import Grid
from ..models.frontier import PriorityQueueFrontier
from ..models.solution import NoSolution, Solution
from ..models.node import Node

class AStarSearch:
    @staticmethod
    def search(grid: Grid) -> Solution:
        """
        Encuentra el camino óptimo usando Búsqueda A* (A-Star).
        Evalúa f(n) = g(n) + h(n), donde g(n) es el costo acumulado 
        y h(n) es la heurística estimada hasta la meta.
        """
        root = Node("", state=grid.initial, cost=0, parent=None, action=None)
        reached = {}
        # Frontera ordenada por prioridad (menor f(n)
        frontera = PriorityQueueFrontier()
        frontera.add(root, root.cost + grid.heuristic(root.state)) 

        while not frontera.is_empty():
            n = frontera.pop()
            
            
            if grid.objective_test(n.state):
                return Solution(n, reached)

            if n.state in reached:
                continue
            reached[n.state] = True

            # Expansión de vecinos
            for a in grid.actions(n.state):
                s = grid.result(n.state, a)
                
                if s not in reached:
                    son = Node(
                        "",
                        state=s,
                        cost=n.cost + grid.individual_cost(n.state, a), # g(n)
                        parent=n,
                        action=a,
                    )
                    
                    # Prioridad = costo real (g) + estimación (h)
                    f_n = son.cost + grid.heuristic(s)
                    frontera.add(son, f_n) 

        return NoSolution(reached)
    