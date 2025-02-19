# TriangulumAI

## Overview

This repository implements an AI-driven approach to solving the Monochromatic Triangle Problem. It leverages a genetic algorithm (GA), combined with a two-opt local search, to optimize vertex coloring in a complete graph with the goal of minimizing monochromatic triangles. A visualization module provides real-time feedback on the GA's progress, depicting the graph, vertex coloring, and the best solution found so far.

## Features

*   **Genetic Algorithm (GA) Implementation:** Employs a complete GA pipeline, including:
    *   **Initialization:** Randomly generates a population of vertex coloring solutions based on input data.
    *   **Fitness Evaluation:** Quantifies the quality of each solution by considering the number of monochromatic triangles and edge lengths.
    *   **Selection:** Uses tournament selection to choose parent solutions for reproduction.
    *   **Crossover:** Combines genetic material from parent solutions to create offspring.
    *   **Mutation:** Introduces random variations to offspring to maintain diversity within the population.
    *   **Elitism:** Tracks and maintains the best-performing solution across generations.
*   **Two-Opt Local Search:** Optimizes solutions generated by the GA by iteratively swapping pairs of vertices in the solution.
*   **Monochromatic Triangle Detection:** Implements an efficient algorithm to identify and count monochromatic triangles within the graph, used as part of the fitness evaluation.
*   **CSV Data Input:** Reads vertex coordinates from CSV files using the `pandas` library, allowing for easy integration with custom datasets.
*   **Real-Time Visualization:** Provides a visual representation of the graph and the GA's optimization process using `matplotlib` and `tkinter`. Key visual elements include:
    *   Vertex representation with color-coding based on assigned colors
    *   Highlighting of monochromatic triangles
    *   Visualization of the best solution's path through the graph
*   **GUI Parameter Control:** Kivy framework that enables easy parameter adjustments for the GA.

## Technologies Used

*   **Python 3.x:** Core programming language
*   **networkx:** Used for graph representation and manipulation. Facilitates the creation of complete graphs and calculation of triangles.
*   **numpy:** Provides numerical computation capabilities for fitness calculations and data handling.
*   **pandas:** Provides efficient data structure called dataframes and tools for handling CSV data.
*   **tkinter:** Enables basic GUI elements for file browsing and visualization.
*   **kivy:** Enables GUI and parameterization.
*   **matplotlib:** Used to render graphs and visualize.

## Installation

1.  **Clone the repository:**

    ```
    git clone [repository URL]
    cd [repository name]
    ```

2.  **Install the required packages using pip:**

    ```
    pip install networkx numpy pandas matplotlib kivy
    ```

    *   It's highly recommended to use a virtual environment (e.g., `venv` or `conda`) to isolate project dependencies and avoid conflicts with other Python packages.

## Usage

1.  **Prepare your data:**

    *   Create a CSV file containing vertex coordinate data. The file must have two columns, named `numberrange_x` and `numberrange_y`, representing the x and y coordinates of each vertex.
    *   Example `data.csv` format:

        ```
        numberrange_x,numberrange_y
        1.0,2.0
        3.0,4.0
        5.0,6.0
        ...
        ```

2.  **Run the main script:**

    ```
    python main.py
    ```

3.  **Interact with the GUI:**

    *   Use the "Browse" button to select your data file.
    *   Adjust the following GA parameters:
        *   **Generations:** The number of iterations the GA will run for.
        *   **Population Size:** The number of solutions in each generation.
        *   **Mutation Rate:** The probability of a gene (vertex color) mutating in an offspring.
    *   Click "Start" to initiate the algorithm and visualization.

## Project Structure

├── README.md # This file
├── main.py # Main application script with Kivy GUI
├── monochromatic_triangle.py # Core logic for the GA and problem
├── visualization.py # Visualization components using matplotlib and tkinter
├── data/ # Sample datasets
│ └── sample_data.csv
└── ...**


## Code Explanation

### `monochromatic_triangle.py`

This module implements the core logic for solving the monochromatic triangle problem using a genetic algorithm.

#### `MonochromaticTriangle` Class

*   **`\_\_init\_\_(self, mt_data)`**
    *   Initializes the graph (`networkx.Graph`), a dictionary to store vertices (`self.vertices`), and variables to store the best solution, its fitness, and the history of best solutions and paths.
    *   `mt_data`: A `pandas.DataFrame` containing vertex coordinates.
    *   Calls `self.initialize(mt_data)` to initialize the graph and generate an initial population of solutions.
*   **`initialize(self, mt_data)`**
    *   Reads vertex coordinates from the input `mt_data` (a `pandas.DataFrame`).
    *   Adds each vertex to the `networkx.Graph` object.
    *   Generates an initial population of random coloring solutions, where each solution is a list of integers representing the colors assigned to each vertex.
*   **`fitness(self, solution)`**
    *   Calculates the fitness of a given vertex coloring solution. The fitness function is designed to *minimize* the number of monochromatic triangles and edge lengths.
    *   It uses `nx.algorithms.triangles` to count the number of triangles formed by the given solution.
    *   The total length of edges is calculated by summing the lengths of edges connecting the vertices.
    *   The final fitness value is a weighted sum of the triangle count and the total edge length.
*   **`generate_monochromatic_triangles(self)`**
    *   Iterates over the population and identifies monochromatic triangles in each solution.
    *   Returns a list of monochromatic triangles for each individual in the population.
*   **`run_ga(self, generations=100, pop_size=40, mutation_rate=0.05)`**
    *   Implements the genetic algorithm.
    *   The GA runs for a specified number of generations.
    *   The population size is maintained at a specified value.
    *   Uses tournament selection to pick a sample from the population.
    *   Crossover, mutation, and two-opt local search are applied to evolve the population.

        *   **Crossover:** Randomly select genes from both parents to create offspring.
        *   **Mutation:**

            ```
            mutated_offspring = [
                [random.randint(0, num_colors - 1) if random.random() < mutation_rate else gene for gene in individual]
                for individual in offspring
            ]
            ```
    *   The best solution (i.e., the coloring with the lowest number of monochromatic triangles) is tracked and stored.
    *   The history of best solutions and paths is also maintained for visualization purposes.
*   **`two_opt_local_search(self, solutions)`**
    *   Implements the two-opt local search heuristic to refine the solutions generated by the GA.
    *   Iteratively swaps pairs of elements (vertex colors) in the solution to improve the fitness.

### `visualization.py`

This module implements the visualization aspects of the Monochromatic Triangle Problem solver.

#### `MonochromaticTriangleVisualization` Class

*   **`\_\_init\_\_(self, mt_data, mt)`**
    *   Initializes the visualization environment, including the `networkx.Graph`, vertex coordinates (`self.vertices`), `matplotlib` figure and axes (`self.fig`, `self.ax`), and `tkinter` canvas.
    *   Stores a reference to the `MonochromaticTriangle` object (`self.mt`) to access the GA's state.
*   **`draw_vertices(self)`**
    *   Plots the vertices on the graph using `matplotlib`.
    *   Vertices that are part of the best solution are highlighted in red.
*   **`update_text_labels(self)`**
    *   Updates the text labels in the visualization to display the current best solution and best fitness value.
*   **`draw_best_path(self)`**
    *   Draws the best path in blue arrows between vertices, iterating over the best solution list.
*   **`update_plot(self)`**
    *   Clears the axes and redraws the graph with the current vertex coloring.
    *   Calls `draw_best_path()` to draw the best path.
    *   Draws a circle around the graph.
    *   Updates the `matplotlib` canvas.
*   **`draw_nodes_in_circle(self)`**
    *   Calculates the center point of the graph and draws a circle around the graph.
    *   It then calculates the x and y coordinates for each node on the circle using trigonometric functions.
*   **`visualize(self)`**
    *   Initializes a `tkinter` window to display the visualization.
    *   Iterates through the `best_solution_history` and `best_path_history` to visualize the algorithm's progress.
    *   For each iteration, it draws the monochromatic triangles, nodes, edges, and the best path.
    *   Uses `time.sleep(1)` to add a delay between iterations for better visualization.
    *   Starts the `tkinter` main loop.
*   **`draw_circle(self)`**
    *   Draws a circle around the graph.
    *   Calculates the center and radius of the circle based on the vertex coordinates.
    *   Adds the circle as a patch to the `matplotlib` axes.
    *   Draws labels for each vertex around the circle.

### `main.py`

This module contains the main GUI logic for the application, built using `kivy`.

#### `MonochromaticTriangleGUI` Class

*   **`\_\_init\_\_(self, \*\*kwargs)`**
    *   Initializes the GUI elements using the Kivy framework.
    *   Creates buttons for browsing data files and running the GA.
    *   Sets default values for GA parameters (generations, mutation rate, population size).
    *   Creates a background color and rounded rectangle for the GUI.

## Contributing

Feel free to contribute to this project by submitting pull requests, reporting issues, or suggesting improvements.

## License

[Choose a license, e.g., MIT License]

## Future Work

*   Implement more advanced crossover and mutation operators.
*   Explore different local search heuristics (e.g., simulated annealing).
*   Add support for different graph types (e.g., sparse graphs).
*   Develop a more interactive and customizable visualization interface.
*   Add documentation
*   AND MOREE

