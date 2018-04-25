import math, random

"""Note to reader: this is a work in progress as of 4/24/18. The code will not do anything very impressive, but it does build all the necessary data structures for a working neural network. It also lays out the key methods of the neural_net class, which will be essential to its later functioning. Thank you for your interest! - David DeLuca
"""

class neural_net:
	def __init__(self,LAYERS=3, NEURONS=3):
		#initializes the arrays of the neural network with LEVELS levels and NEURONS neurons in each level
		self.n_input = [[0 for i in range(0,NEURONS)] for j in range(0,LAYERS)]
		self.n_output = [[0 for i in range(0,NEURONS)] for j in range(0,LAYERS)]
		self.n_dte_no = [[0 for i in range(0,NEURONS)] for j in range(0,LAYERS)]
		self.c_weight = [[[random.randint(0,100)/100 for i in range(0,NEURONS)] for j in range(0,NEURONS)] for k in range(0,LAYERS-1)]
		self.c_new_weight = [[[0 for i in range(0,NEURONS)] for j in range(0,NEURONS)] for k in range(0,LAYERS-1)]
		#initialize a list of neural networks to which this one will spit its output
		self.d_nets = []
		#store the value of the length of the arrays, so we don't have to use 'len()' all the time
		self.num_neurons = NEURONS
		self.num_layers = LAYERS
		#set some other important constants
		self.LIST_FORMAT = 0
	
	def grow(self,N):
		"""Appends N new layers to the end of the neural network."""
		for i in range(0,N):
			self.n_input.append([0 for i in range(0,self.num_neurons)])
			self.n_output.append([0 for i in range(0,self.num_neurons)])
			self.n_dte_no.append([0 for i in range(0,self.num_neurons)])
			self.c_weight.append([[0 for i in range(0,self.num_neurons)] for j in range(0,self.num_neurons)])
			self.c_new_weight.append([[0 for i in range(0,self.num_neurons)] for j in range(0,self.num_neurons)])
			self.num_layers += 1

	def chop(self,INDEX=-1):
		"""Removes layer INDEX from the neural network.
		
		Takes one optional argument, INDEX (integer - the index of the layer to remove), and returns nothing. If INDEX is omitted, the method removes the final layer from the neural network. Prints an error message if the layer does not exist or if there is some other problem while popping the layer from the list."""
		try:
			if INDEX == -1:
				self.n_input.pop()
				self.n_output.pop()
				self.n_dte_no.pop()
				self.c_weight.pop()
				self.c_new_weight.pop()
			else:
				self.n_input.pop(INDEX)
				self.n_output.pop(INDEX)
				self.n_dte_no.pop(INDEX)
				self.c_weight.pop(INDEX)
				self.c_new_weight.pop(INDEX)
		except:
			print("Error while attempting removal of layer "+str(INDEX)+".")
		else:
			self.num_layers -= 1

	def seed(self,INPUT_ARRAY,DIRECT=1):
		"""Sets n_output for Layer 0, according to a maximum slice of INPUT_ARRAY."""
		numSeeds = min(len(INPUT_ARRAY),self.num_neurons)
		self.n_output[0] = INPUT_ARRAY[0:numSeeds]

	def flow(self):
		"""Computes all n_outputs as function of respective n_inputs, then accumulates n_output in n_input of Layer n+1, modulated by c_weights. Cascades forward layer by layer.
		
		Takes no arguments, returns nothing."""
		for i in range(0,self.num_layers):
			for j in range(0,self.num_neurons):
				#skip first layer because outputs thereof already decided by seed
				if i > 0:
					self.n_output[i][j] = self.squish(self.n_input[i][j])
				#note: don't need if statement checking for layer number because we already limit it in first for statement
				if i < self.num_layers-1:
					for k in range(0,self.num_neurons):
						#increase the input sum of each of the neurons in the layer i+1 by the output of the current neuron in layer i, multiplied by the weight of the connection to that L(i+1) neuron
						#? possible to use sum function somehow? not necessary?
						self.n_input[i+1][k] += self.c_weight[i][j][k] * self.n_output[i][j]  
				else:
					#we are on the final layer, and outputs have been calculated. now we need to feed our output to all destination networks connected to this one
					for each_destination in self.d_nets:
						self.feed(each_destination)

	def shave(self,N):
		"""Removes N neurons from each layer, returns nothing"""
		for i in range(0,self.num_layers):
			self.n_input[i].pop()
			self.n_output[i].pop()
			self.n_dte_no[i].pop()
			self.c_weight[i].pop()
			self.c_new_weight[i].pop()
			self.num_neurons -= 1

	def unhitch(self,DESTINATION):
		"""Removes DESTINATION (a neural network) from the network's list of networks to send output"""
		try:
			self.d_nets.remove(DESTINATION)
		except:
			print("The specified neural network was not hitched in the first place.")
	
	def hitch(self,DESTINATION):
		"""Adds DESTINATION (a neural network) to the network's list of networks to send output."""
		try:
			self.d_nets.append(DESTINATION)
		except:
			print("There was an error while hitching the given neural network.")

	def fatten(self,N):
		"""Adds N neurons to each layer, returns nothing"""
		for i in range(0,self.num_layers):
			#add N connections to each neuron, but not from last layer
			if i < self.num_layers-1:
				for j in range(0,self.num_neurons):
					for k in range(0,N):
						self.c_weight[i][j].append(0)
						self.c_new_weight[i][j].append(0)
			for j in range(0,N):
				self.n_input[i].append(0)
				self.n_output[i].append(0)
				self.n_dte_no[i].append(0)
				if i < self.num_layers-1:
					self.c_weight[i].append([0 for k in range(0,self.num_neurons+N)])
					self.c_new_weight[i].append([0 for k in range(0,self.num_neurons+N)])
		self.num_neurons += 1
				
	def spit(self,FORMAT=0):
		"""Returns an array representing the final layer's output values.
		
		Takes one optional argument, FORMAT, which is used to determine the format of the output. If FORMAT is 0, then the function will return a list of numerical values corresponding to n_output on the final layer."""
		if FORMAT == self.LIST_FORMAT:
			return self.n_output[self.num_layers-1]
	
	def feed(self,DESTINATION):
		"""'Feeds' the final layer's output to DESTINATION, a neural network object."""
		DESTINATION.seed(self.spit())

	def print(self):
		"""Prints each of the principal arrays to the console. Takes no arguments."""
		print(self.n_input,self.n_output,self.n_dte_no,self.c_weight,self.c_new_weight,sep="\n\n")

	def wipe(self):
		"""Resets neural net inputs, outputs, and connection deltas. Does *not* reset connection weights. Takes no arguments."""
		for i in range(0,self.num_layers):
			for j in range(0,self.num_neurons):
				self.n_input[i][j] = 0
				self.n_output[i][j] = 0
				self.n_dte_no[i][j] = 0
				for k in range(0,self.num_neurons):
					self.c_new_weight[i][j][k] = 0

	def transform(self,LIST,METHOD):
		"""Performs the FUNCTION on every member and submember of list, then returns the transformed list. Takes two arguments, the LIST and the FUNCTION."""
		for each_subitem in LIST:
			if type(each_subitem) is list:
				return_list.append(self.transform(each_subitem),METHOD)
			else:
				try:
					return FUNCTION(each_subitem)
				except:
					print("ERROR: Couldn't perform the operation on the subitem.")
					return each_subitem

	def squish(self,N):
		"""Runs the logistic function on numerical input N. Returns result"""
		return 1 / ( 1 + math.e**(-N) )

	def squerr(self,OUTPUT,TARGET):
		"""Returns the squared error, given the actual OUTPUT (arg1) and the TARGET (arg2) output."""
		return 0.5 * (TARGET - OUTPUT) ** 2 

	def scale(self, DEST_NEURON_OUTPUT,
		   ORIGIN_NEURON_OUTPUT,
		   D_TOTAL_ERROR_WRT_OUTPUT,
		   CONNECTION_WEIGHT,
		   LEARNING_RATE):
		"""Returns a connection weight based on several variables:
			- DEST_NEURON_OUTPUT = the output of the destination neuron of the connection
			- ORIGIN_NEURON_OUTPUT = the output of the origin neuron of the connection
			- D_TOTAL_ERROR_WRT_OUTPUT = the derivative of the total error with respect to the destination neuron output
			- CONNECTION_WEIGHT = the weight of the connection in question
			- LEARNING_RATE = the learning rate """
		return CONNECTION_WEIGHT + LEARNING_RATE * (D_TOTAL_ERROR_WRT_OUTPUT) * (DEST_NEURON_OUTPUT * (1 - DEST_NEURON_OUTPUT)) * ORIGIN_NEURON_OUTPUT

	def fresh(self):
		"""Sets the weight of every connection to its new weight, as calculated by the 'revise' method. Takes no arguments."""
		for i in range(0,self.num_layers-1):
			for j in range(0,self.num_neurons):
				for k in range(0,self.num_neurons):
					self.c_weight[i][j][k] = self.c_new_weight[i][j][k]
					self.c_new_weight[i][j][k] = 0

	def revise(self,CORRECT_OUTPUT):
		"""Updates the 'new weight' 3-dimensional array pre-update.
			Takes one argument, CORRECT_OUTPUT, a one-dimensional numerical array that represents the correct outputs, 0 to n, of the final layer of the array."""
		#starts at layer total-1 because there are no connections going from last layer
		for i in reversed(range(0,self.num_layers-1)):
			for j in range(0,self.num_neurons):
				for k in range(0,self.num_neurons):
					
					#this is the first iteration, where i = one less than the total number of countable layers
					if i == self.num_layers-2:
						temp_d_te_out = self.squerr(self.n_output[i][k],CORRECT_OUTPUT[k])
						self.n_dte_no[i+1][k] = temp_d_te_out
					else:
						#if we're not on the final layer, we are calculating the derivative of total error with respect to the destination neuron output. We calculate that by summing the derivatives of the destination output error with respect to the current neuron output, which we calculated in previous steps
						temp_d_te_out = sum( self.n_dte_no[i+1][k] * self.c_weight[i][k][m] for m in range(0,self.num_neurons) )
						self.n_dte_no[i][j] = temp_d_te_out
		
					self.c_new_weight[i][j][k] = self.scale(self.n_output[i][k],
													self.n_output[i][j],
													temp_d_te_out,
													self.c_weight[i][j][k],
													0.5)			

	def train(self,INPUT_ARRAY,ANSWER_ARRAY):
		"""Runs arg1 INPUT_ARRAY through the neural network, evaluates the result next to ANSWER_ARRAY, then adjusts the connection weights in the neural network to reduce the total error on the next go."""
		#munch INPUT_ARRAY
		#compare result to ANSWER_ARRAY
		#calculate delta for each connection
		#update connection weights with delta
		#print a report of what happened, inc. total error
		pass

	def munch(self,INPUT_ARRAY):
		"""Seeds INPUT_ARRAY to the top layer of the neural network, pushes it through subsequent layers, and then returns the output of the final layer. This is the "black box." """
		self.seed(INPUT_ARRAY)
		self.flow()
		return self.spit()

neural_net1 = neural_net(5,2)
neural_net1.print()




