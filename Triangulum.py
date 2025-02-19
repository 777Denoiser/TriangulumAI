import math
import random
import time
import networkx as nx
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from kivy.app import App
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# here i imported all the necessary libraries above

#below is defined the monochromatic triangle definition,
#i initialized the graphics, dictionary for vertices,
# andvarables for storing best solution and fitness and history of best solution and best path
class MonochromaticTriangle:
    def __init__(self, mt_data):
        self.graph = nx.Graph()
        self.vertices = {}
        self.best_solution = None
        self.best_fitness = np.inf
        self.best_solution_history = []
        self.best_path_history = []
        self.initialize(mt_data)

# below is initialized the dataset columns numberrange x and y
# so the column parser through the Genetic Algorithm
# assigning a # to each vertex, and adding the vertex to the graph
# also generated a population of coloring for the vertices.

    def initialize(self, mt_data):
        self.graph = nx.Graph()
        self.vertices = {}
        self.best_solution = None
        self.best_fitness = np.inf
        for i, row in mt_data.iterrows():
            try:
                x = float(row['numberrange_x'])
                y = float(row['numberrange_y'])
            except ValueError:
                print(f"Error parsing vertex: {row}")
                continue
            vertex = i + 1
            self.vertices[vertex] = (x, y)
            self.graph.add_node(vertex)
        num_colors = len(mt_data)
        self.population = [
            [random.randint(1, num_colors) for _ in range(len(mt_data))]
            for _ in range(len(mt_data))
        ]

#below the fitness definition calculates the total length of edges in the graph

    def fitness(self, solution):
        triangles_count = sum(nx.algorithms.triangles(self.graph, solution).values())
        length = sum(self.graph.edges[e]['length'] for e in self.graph.edges(solution))
        fitness = triangles_count + 0.01 * length
        return fitness

# below definition iterates over population attribute of the self object for each individual in terms of i, j, k to detect monochromatic triangles
    def generate_monochromatic_triangles(self):
        monochromatic_triangles = []
        for individual in self.population:
            triangles = []
            for i in range(len(individual)):
                for j in range(i + 1, len(individual)):
                    for k in range(j + 1, len(individual)):
                        if individual[i] == individual[j] == individual[k]:
                            vertex_i = list(self.vertices.keys())[i]
                            vertex_j = list(self.vertices.keys())[j]
                            vertex_k = list(self.vertices.keys())[k]
                            triangle = [self.vertices[vertex_i], self.vertices[vertex_j], self.vertices[vertex_k]]
                            triangles.append(triangle)
            monochromatic_triangles.append(triangles)
        return monochromatic_triangles

#below it create the genetics algorithm that runs for all specific number of generations
# this maintains the population of solutions in each generation.
# crossover and mutation and two-opt local search are applied
    def run_ga(self, generations=100, pop_size=40, mutation_rate=0.05):
        num_colors = len(self.vertices)
        self.best_solution = self.population[0].copy()  # initial best solution
        self.best_fitness = self.fitness(self.best_solution)
        self.best_solution_history.append(self.best_solution.copy())  # Append initial best solution to history
        self.best_path_history.append(
            self.two_opt_local_search([self.best_solution])[0])  # Append initial best path to history
        for _ in range(generations):
            parents = [
                min(random.choices(self.population, k=2), key=self.fitness)
                for _ in range(len(self.population))
            ]
            offspring = [
                p1.copy() if m == 0 else p2.copy()
                for p1, p2, m in zip(parents[::2], parents[1::2],
                                     [[random.randint(0, 1) for _ in range(len(p1))] for p1 in parents[::2]])
            ]
            mutated_offspring = [
                [random.randint(0, num_colors - 1) if random.random() < mutation_rate else gene for gene in individual]
                for individual in offspring
            ]
            improved_offspring = [
                self.two_opt_local_search([solution])[0]
                for solution in mutated_offspring
            ]
            self.population.extend(improved_offspring)
            self.population = self.population[:pop_size]
            best = min(self.population, key=self.fitness)
            if self.fitness(best) < self.best_fitness:
                self.best_solution = best
                self.best_fitness = self.fitness(best)
            self.best_solution_history.append(self.best_solution.copy())  # Append best solution to history
            self.best_path_history.append(
                self.two_opt_local_search([self.best_solution])[0])  # Append best path to history
        return self.best_solution

# here i take all lists of solutions and input and,
    # iteratively improve ones each solution by swapping pairs of elements in the solution
    def two_opt_local_search(self, solutions):
        improved_solutions = []
        for solution in solutions:
            improved = solution.copy()
            fitness_before = self.fitness(improved)
            while True:
                made_improvement = False
                for i in range(len(improved) - 1):
                    for j in range(i + 1, len(improved)):
                        new_solution = improved.copy()
                        new_solution[i:j] = reversed(new_solution[i:j])
                        if self.fitness(new_solution) < fitness_before:
                            improved = new_solution
                            fitness_before = self.fitness(improved)
                            made_improvement = True
                if not made_improvement:
                    break
            improved_solutions.append(improved)
        return improved_solutions

# this is where it initializes the graph and dictionary of vertices and stores the best solution
class MonochromaticTriangleVisualization:
    def __init__(self, mt_data, mt):
        self.graph = nx.Graph()
        self.vertices = {}
        self.best_solution = None
        self.mt = mt
        self.best_solution_text = None
        self.best_fitness_text = None

        for i, row in mt_data.iterrows():
            try:
                x = float(row['numberrange_x'])
                y = float(row['numberrange_y'])
            except ValueError:
                print(f"Error parsing vertex: {row}")
                continue
            vertex = i  # Use index as the vertex key
            self.vertices[vertex] = (x, y)
            self.graph.add_node(vertex)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)

#this is where it plotted coordinates of certain vertices only then graph checking if it is in the best solution set
    def draw_vertices(self):
        if self.best_solution:
            best_solution_set = set(self.best_solution)
            for vertex in self.graph.nodes:
                if vertex in best_solution_set:
                    x, y = self.vertices[vertex]  # Retrieve vertex coordinates
                    self.ax.plot(x, y, 'ro', markersize=3)


# this is where i update the labels for the console output and best solution and best path
    def update_text_labels(self):
        if self.best_solution_text:
            self.best_solution_text.set_text(f"Best Solution: {self.best_solution}")
        if self.best_fitness_text:
            self.best_fitness_text.set_text(f"Best Fitness: {self.mt.best_fitness:.2f}")

# this si wheres i draw the best path in blue arrows between vertices iterating over the best solution list
    def draw_best_path(self):
        if self.best_solution:
            for i in range(len(self.best_solution) - 1):
                start_vertex = self.best_solution[i]
                end_vertex = self.best_solution[i + 1]
                if start_vertex in self.vertices and end_vertex in self.vertices:
                    start_coords = self.vertices[start_vertex]
                    end_coords = self.vertices[end_vertex]
                    self.ax.annotate("", xy=start_coords, xytext=end_coords,
                                     arrowprops=dict(arrowstyle="->", color='blue', lw=2,
                                                     connectionstyle="arc3,rad=0.1"))
            # Connect the last vertex with the first vertex to complete the cycle
            start_vertex = self.best_solution[-1]
            end_vertex = self.best_solution[0]
            if start_vertex in self.vertices and end_vertex in self.vertices:
                start_coords = self.vertices[start_vertex]
                end_coords = self.vertices[end_vertex]
                self.ax.annotate("", xy=start_coords, xytext=end_coords,
                                 arrowprops=dict(arrowstyle="->", color='blue', lw=2,
                                                 connectionstyle="arc3,rad=0.1"))

# this is where i update the plot and draw the nodes and edges and the labels and best path
# this is also where i drew the dotted circles to contain the blue arrows and red edges between them
    def update_plot(self):
        self.ax.clear()
        nx.draw_networkx_nodes(self.graph, self.vertices, node_color='black', ax=self.ax)
        if self.best_solution:
            nx.draw_networkx_edges(self.graph, self.vertices,
                                   edgelist=[(self.best_solution[i], self.best_solution[i + 1]) for i in
                                             range(len(self.best_solution) - 1)], edge_color='red', ax=self.ax)
            nx.draw_networkx_labels(self.graph, self.vertices, labels={v: str(v) for v in self.graph.nodes},
                                    font_color='white', font_size=10, ax=self.ax)
            self.draw_best_path()  # Draw the best path
        self.draw_circle()
        self.canvas.draw()

# this is where it calculates the center point of the graph. It then calculates the radius of the circle
    # here it calculates the x and y coordinates for each node on the circle using trig
    def draw_nodes_in_circle(self):
        num_nodes = len(self.vertices)
        center = (
            sum(x for x, _ in self.vertices.values()) / num_nodes,
            sum(y for _, y in self.vertices.values()) / num_nodes)
        radius = max(math.dist(center, vertex) for vertex in self.vertices.values())
        angle_increment = 2 * math.pi / num_nodes
        for i, vertex in enumerate(self.vertices):
            angle = i * angle_increment
            x = center[0] + (radius + 0.1) * math.cos(angle)
            y = center[1] + (radius + 0.1) * math.sin(angle)
            self.ax.plot(x, y, 'ro', markersize=3)
            self.ax.text(x, y + 0.15, str(vertex), ha='center', va='center', color='white', fontsize=10)
            self.vertices[vertex] = (x, y)
            next_vertex = (vertex + 1) % num_nodes
            if next_vertex in self.vertices:
                self.ax.plot([x, self.vertices[next_vertex][0]], [y, self.vertices[next_vertex][1]], 'r-', linewidth=2)


# this is where i draw the monochromatic triangles and the graph with nodes and edges
    # and the plotting the best solution
    def visualize(self):
        top = Tk()
        top.title("Genetic Algorithm Visualization")
        self.canvas.draw()
        tkagg = FigureCanvasTkAgg(self.fig, master=top)
        tkagg.draw()
        tkagg.get_tk_widget().pack()
        monochromatic_triangles = self.mt.generate_monochromatic_triangles()
        for i, triangles in enumerate(monochromatic_triangles):
            for triangle in triangles:
                if all(vertex in self.vertices for vertex in triangle):
                    x = [self.vertices[vertex][0] for vertex in triangle]
                    y = [self.vertices[vertex][1] for vertex in triangle]
                    self.ax.plot(x + [x[0]], y + [y[0]], 'r-', linewidth=2)
            self.ax.clear()
            nx.draw_networkx_nodes(self.graph, self.vertices, node_color='black', ax=self.ax)
            if i < len(self.mt.best_solution_history):
                best_solution = self.mt.best_solution_history[i]
                best_path = self.mt.best_path_history[i]
                nx.draw_networkx_edges(self.graph, best_solution, edge_color='red', ax=self.ax)
                nx.draw_networkx_edges(self.graph, best_path, edge_color='blue', ax=self.ax)
                nx.draw_networkx_labels(self.graph, self.vertices, font_color='white', font_size=10, ax=self.ax)
            self.draw_vertices()
            self.draw_circle()
            self.draw_best_path()  # Draw the best path
            self.canvas.draw()
            top.update()
            time.sleep(1)  # Add a delay to visualize each iteration
        top.mainloop()

# this is where i draw the circle around the graph
    def draw_circle(self):
        num_nodes = len(self.vertices)
        center = (
            sum(x for x, _ in self.vertices.values()) / num_nodes,
            sum(y for _, y in self.vertices.values()) / num_nodes)
        radius = max(math.dist(center, vertex) for vertex in self.vertices.values())
        circle = plt.Circle(center, radius, edgecolor='black', facecolor='none', linestyle='dotted')
        self.ax.add_patch(circle)
        angle_increment = 2 * math.pi / num_nodes
        for vertex in self.vertices:
            angle = vertex * angle_increment
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            self.ax.text(x, y, str(vertex), ha='center', va='center', color='white', fontsize=10)
            self.vertices[vertex] = (x, y)

# this is where it creates and configures most of the main GUI
class MonochromaticTriangleGUI(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mt = None
        self.mt_data = None
        self.generations = 100  # Default value for generation size
        self.mutation_rate = 0.05  # Default value for mutation rate
        self.pop_size = 40  # Default value for population size
        with self.canvas.before:
            Color(0, 0.2, 1, 0.1)
            self.rect = RoundedRectangle(size=self.size, pos=self.pos, radius=[20, ])
        self.bind(size=self._update_rect, pos=self._update_rect)
        # Browse button
        self.browse_button = Button(text="Browse", size_hint=(0.2, 0.1), pos_hint={'x': 0.4, 'y': 0.8},
                                    background_color=(2, 0.5, 1, 1))  # Set desired button color
        self.browse_button.bind(on_release=self.browse_file)
        self.add_widget(self.browse_button)
        # Number of generations
        self.num_generations_label = Label(text="Number of Generations:", font_name="Verdana", font_size=30,
                                           size_hint=(0.2, 0.1), pos_hint={'x': 0.6, 'y': 0.6})
        self.add_widget(self.num_generations_label)
        self.num_generations_entry = TextInput(size_hint=(0.2, 0.1), pos_hint={'x': 0.8, 'y': 0.6})
        self.add_widget(self.num_generations_entry)
        # Number of mutations
        self.num_mutations_label = Label(text="Number of Mutations:", font_name="Verdana", font_size=30,
                                         size_hint=(0.2, 0.1), pos_hint={'x': 0.6, 'y': 0.5})
        self.add_widget(self.num_mutations_label)
        self.num_mutations_entry = TextInput(size_hint=(0.2, 0.1), pos_hint={'x': 0.8, 'y': 0.5})
        self.add_widget(self.num_mutations_entry)
        # Number of populations
        self.num_populations_label = Label(text="Number of Populations:", font_name="Verdana", font_size=30,
                                           size_hint=(0.2, 0.1), pos_hint={'x': 0.6, 'y': 0.4})
        self.add_widget(self.num_populations_label)
        self.num_populations_entry = TextInput(size_hint=(0.2, 0.1), pos_hint={'x': 0.8, 'y': 0.4})
        self.add_widget(self.num_populations_entry)
        # Run button
        self.run_button = Button(text="(Auto Run)", size_hint=(0.2, 0.1), pos_hint={'x': 0.4, 'y': 0.2},
                                 background_color=(2, 0.5, 1, 1))  # Set desired button color
        self.run_button.bind(on_release=self.run)
        self.add_widget(self.run_button)
        self.csv_files = []

# this is where updates the position and size of the rectangle objects window
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

# this definition opens a file dialog to select one or more CSV files and,
    # loads the CSV dataset by adding the dataset to a list and assigning columns names
    def browse_file(self, *args):
        Tk().withdraw()
        file_paths = askopenfilenames()  # Allow selecting multiple files
        if not file_paths:
            print("No files selected")
            return
        try:
            for file_path in file_paths:
                print("Loading CSV:", file_path)
                mt_data = pd.read_csv(file_path)
                mt_data.columns = ['numberrange_x', 'numberrange_y']
                self.csv_files.append((file_path, mt_data))  # Append (file_path, mt_data) tuple to the list
        except Exception as e:
            print("Error loading CSV:", e)
            return
        self.visualize_genetic_algorithm()

 # This is where we can visualize the GA w/ WOC over a list of CSV files or just one
    # the instance passed through the GUI Generations, mutations, population if not then revert to default
    def visualize_genetic_algorithm(self):
        for file_path, mt_data in self.csv_files:
            mt = MonochromaticTriangle(mt_data)
            mt_visualization = MonochromaticTriangleVisualization(mt_data, mt)

            start_time = time.time()
            best_solution = mt.run_ga(generations=self.generations, pop_size=self.pop_size,
                                      mutation_rate=self.mutation_rate)
            end_time = time.time()

            best_fitness = mt.fitness(best_solution)
            print("Best Solution (Best Path):", best_solution)
            print("Best Fitness Score:", best_fitness)
            print("Time taken:", end_time - start_time, "seconds")

            mt_visualization.best_solution = best_solution
            mt_visualization.visualize()

#This is where i run the csv filesystem and retrieve data entry from GUI and calls the previous definition
    def run(self, instance):
        if not self.csv_files:
            print("No data loaded")
            return
        num_generations_text = self.num_generations_entry.text
        if num_generations_text:
            self.generations = int(num_generations_text)
        mutation_rate_text = self.num_mutations_entry.text
        if mutation_rate_text:
            self.mutation_rate = float(mutation_rate_text)
        pop_size_text = self.num_populations_entry.text
        if pop_size_text:
            self.pop_size = int(pop_size_text)
        self.visualize_genetic_algorithm()

#this is where i return the GUI
class MonochromaticTriangleApp(App):
    def build(self):
        return MonochromaticTriangleGUI()

#this is where i run the GUI
if __name__ == "__main__":
    MonochromaticTriangleApp().run()

