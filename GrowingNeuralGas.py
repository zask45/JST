"""
This code implements the Growing Neural Gas algorithm that creates a graph 
that learns the topologies in the given input data. 

See e.g. followning documents references: 
https://papers.nips.cc/paper/893-a-growing-neural-gas-network-learns-topologies.pdf
http://www.booru.net/download/MasterThesisProj.pdf
"""

from FeatureGraph.graph import Graph
import numpy as np



class GrowingNeuralGas:
    def __init__(self,
                feature_dim : int): 
        
        self.feature_dim = feature_dim
        
        # GNG vertex counter 
        self.n_vertex = 0 
        
        # Unique vertex naming counter
        self.name_counter = 0
        
        # Init stuff
        self.__init_graph()
        
    
    def __get_unique_name(self, prefix : str = 'N') -> str:
        """
        Generates unique names for GNG graph elements
        """
        name = prefix + str(self.name_counter)
        self.name_counter += 1
        return name
        
            
    def __init_graph(self) -> None:
        """
        Initialize the GNG graph database
        """
        self.graph = Graph()

        
    def __add_vertex(self, feature_vector : np.ndarray) -> str:
        """
        Add new vertex to the GNG graph. 
        Returns the vertex key.
        """
        key = self.__get_unique_name()
        self.graph.add_vertex(key=key)
        self.graph.set_vertex_param(key=key, feature_vector=feature_vector, error=0)
        self.n_vertex += 1
        return key
    
    
    def __add_edge(self, key_a : str, key_b : str) -> None:
        """
        Add new edge to the GNG graph and initialize the GNG specific parameters
        """
        self.graph.add_edge(key_a=key_a, key_b=key_b, bidirectional=True)
        self.graph.set_edge_param(key_a=key_a, key_b=key_b, age=0)
        

    def __delete_vertex(self, key : str) -> None: 
        """
        Delete vertex and all edges associated to it. 
        """
        self.graph.delete_vertex(key)
        self.n_vertex -= 1


    def __delete_edge(self, key_a : str, key_b : str) -> None:
        self.graph.delete_edge(key_a, key_b, bidirectional=True)
    
        
    def __get_vertex_param(self, key, param_key) -> any:
        value = self.graph.get_vertex_param(key=key, param_key=param_key)
        return value
    
    
    def __get_edge_param(self, key_a, key_b, param_key) -> any:
        value = self.graph.get_edge_param(key_a=key_a, key_b=key_b, param_key=param_key)
        return value

        
    def __set_vertex_param(self, key, **kwargs) -> None:
        self.graph.set_vertex_param(key=key, **kwargs)
    
    
    def __set_edge_param(self, key_a, key_b, **kwargs) -> None:
        self.graph.set_edge_param(key_a=key_a, key_b=key_b, **kwargs)
        
    
    def __find_nearest_vertices(self, ref_vect : np.ndarray, 
                                n_vertex : int = 2) -> list: 
        # Get all vertex keys
        vertices = self.graph.get_vertices()
        
        # Calculate all distances
        distances = []
        for vertex in vertices: 
            vertex_vect = self.__get_vertex_param(key=vertex, param_key='feature_vector')
            dist = np.linalg.norm(vertex_vect - ref_vect)
            distances.append([vertex, dist])
        
        distances = sorted(distances, key=lambda x: x[1])
        return distances[:n_vertex]
    
    
    def __get_neighbor_vertices(self, key) -> list:
        neighbors = self.graph.get_edges(key)['out']
        return neighbors
    
    
    def __delete_expired_edges(self, age_limit : int) -> None:
        vertices = self.graph.get_vertices()

        for key_a in vertices: 
            neighbors = self.graph.get_edges(key_a)['out']
            for key_b in neighbors:
                edge_age = self.__get_edge_param(key_a, key_b, 'age')
                if edge_age > age_limit:
                    self.graph.delete_edge(key_a, key_b, bidirectional=True)
    
    
    def __delete_unconnected_vertices(self) -> None:
        vertices = self.graph.get_vertices()
        for key in vertices: 
            neighbors = self.graph.get_edges(key)['out']
            if len(neighbors) == 0:
                self.graph.delete_vertex(key) 
                self.n_vertex -= 1

                
    def __find_largest_error_vertex(self) -> str: 
        vertices = self.graph.get_vertices()        
        max_err_val = 0
        max_err_key = ''
        for key in vertices: 
            error = self.__get_vertex_param(key=key, param_key='error')
            if error > max_err_val:
                max_err_val = error
                max_err_key = key
        return [max_err_key, max_err_val]

    
    def __find_largest_error_neighbor(self, key) -> str: 
        neighbors = self.graph.get_edges(key)['out']
        max_err_val = 0
        max_err_key = ''
        for key in neighbors: 
            error = self.__get_vertex_param(key=key, param_key='error')
            if error > max_err_val:
                max_err_val = error
                max_err_key = key
        return [max_err_key, max_err_val]


    def __scale_vertex_error_values(self, attenuation : float) -> None:
        vertices = self.graph.get_vertices()       
        for key in vertices: 
            error = self.__get_vertex_param(key, 'error')
            error -= attenuation * error
            self.__set_vertex_param(key, error=error)
        
        
    def get_graph(self): 
        return self.graph
    
    
    def fit(self, dataset : np.ndarray,
            iterations : int,
            max_vertex : int,
            winner_upd_coeff : float = 0.05,
            neighbor_upd_coeff : float = 0.0005,
            edge_age_limit : int = 100,
            vertex_insert_interval : int = 100,
            vertex_insert_error_scaling : float = 0.5,
            error_attenuation : float = 0.0005, 
            plot_interval : int = 100,
            plot_function : any = None) -> None : 
        
        n_pts = dataset.shape[0]
        
        for iteration in range(1, iterations + 1): 
            if self.n_vertex < 2: # Initialize graph
                # Add the two starting vertices and edge between them
                key_a = self.__add_vertex(feature_vector=np.random.rand(self.feature_dim))
                key_b = self.__add_vertex(feature_vector=np.random.rand(self.feature_dim))
                self.__add_edge(key_a, key_b)

            # Plot stuff
            if (plot_interval is not None) and \
               (iteration % plot_interval == 0) and \
               (plot_function is not None):
                plot_function(dataset, self.graph, iteration)
           
            # Get random data point to be used in GNG graph fitting. 
            idx = np.random.randint(n_pts)
            data_vect = dataset[idx]

            # Find two nearest vertices from the GNG graph
            nearest = self.__find_nearest_vertices(data_vect, n_vertex=2)
            
            # Update the winner vertex error value
            vertex_a_key = nearest[0][0]
            vertex_a_dist = nearest[0][1]
            error = self.__get_vertex_param(key=vertex_a_key, param_key='error')
            error += vertex_a_dist ** 2
            self.__set_vertex_param(key=vertex_a_key, error=error)
            
            # Update winner vertex feature vector
            vertex_vect = self.__get_vertex_param(key=vertex_a_key, param_key='feature_vector')
            vertex_vect = vertex_vect + winner_upd_coeff * (data_vect - vertex_vect)
            self.__set_vertex_param(key=vertex_a_key, feature_vector=vertex_vect)
            
            # Update winner's neighbor vertices
            neighbor_vertices = self.__get_neighbor_vertices(vertex_a_key)
            for vertex_b_key in neighbor_vertices: 
                # Update vectors
                vertex_vect = self.__get_vertex_param(key=vertex_b_key, param_key='feature_vector')
                vertex_vect = vertex_vect + neighbor_upd_coeff * (data_vect - vertex_vect)
                self.__set_vertex_param(key=vertex_b_key, feature_vector=vertex_vect)
                
                # Update edge ages
                edge_age = self.__get_edge_param(vertex_a_key, vertex_b_key, 'age')
                edge_age += 1
                self.__set_edge_param(vertex_a_key, vertex_b_key, age=edge_age)
                
            # Update the second nearest vertex 
            vertex_b_key = nearest[1][0]
            vertex_b_dist = nearest[1][1]
            if vertex_b_key not in neighbor_vertices: 
                # Add edge if it doesn't exist yet
                self.__add_edge(vertex_a_key, vertex_b_key)
            else: 
                # Set edge age to zero
                self.__set_edge_param(vertex_a_key, vertex_b_key, age=0)
                #pass

            # Delete too old edges
            self.__delete_expired_edges(edge_age_limit)
            # Delete vertices with no edges
            self.__delete_unconnected_vertices()
            
            # Add new vertex 
            if (iteration % vertex_insert_interval == 0) and (self.n_vertex < max_vertex):
                # Get vertex with largest error value
                search_result = self.__find_largest_error_vertex() 
                vertex_a_key = search_result[0]
                vertex_a_error = search_result[1]
                vertex_a_vect = self.__get_vertex_param(vertex_a_key, 'feature_vector')

                # Get the neighbor with largest error
                neighbor_result = self.__find_largest_error_neighbor(vertex_a_key)
                vertex_b_key = neighbor_result[0]
                vertex_b_error = neighbor_result[1]
                vertex_b_vect = self.__get_vertex_param(vertex_b_key, 'feature_vector')

                # Calculate new error values
                new_a_error = vertex_a_error * vertex_insert_error_scaling
                self.__set_vertex_param(vertex_a_key, error=new_a_error)
                new_b_error = vertex_b_error * vertex_insert_error_scaling
                self.__set_vertex_param(vertex_b_key, error=new_b_error)

                # Setup new vertex
                new_vertex_vector = (vertex_a_vect + vertex_b_vect) / 2
                new_vertex_error = new_a_error
                new_vertex_key = self.__add_vertex(new_vertex_vector)
                self.__set_vertex_param(new_vertex_key, error=new_vertex_error)
                
                # Rearrange the edges 
                self.__delete_edge(vertex_a_key, vertex_b_key)
                self.__add_edge(new_vertex_key, vertex_a_key)
                self.__add_edge(new_vertex_key, vertex_b_key)

            # Reduce all error values
            self.__scale_vertex_error_values(error_attenuation)
                