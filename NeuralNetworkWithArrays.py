import math, random, time, datetime
from PIL import Image
import array

def all_same(LIST):
	first_item = LIST[0]
	for item in LIST:
		if item != first_item:
			return False
	return True

def log(FILENAME,DATA):
	dtext = time.strftime("%m/%d/%Y %H:%M:%S")
	try:
		with open(FILENAME,'a+') as log_file:
			if type(DATA) == list:
				print(*DATA,dtext,sep='\t',file=log_file)
			else:
				print(DATA,dtext,sep='\t',file=log_file)
			log_file.close()
			pass
	except:
		print("Could you close the log file please?")
	pass

def slope(LIST):
	first = LIST[0]
	last = LIST[len(LIST)-1]
	rise = last-first
	run = 1
	return rise/run

def ct():
	"""Shorthand function for returning the current time."""
	return time.clock()

def average(LIST,d=5):
	"""Returns the average of the given dataset (LIST), rounded to d decimal points."""
	start_time = ct()
	try:
		_temp_average = round(sum(LIST)/len(LIST),d)
		total_time = ct() - start_time
		log("log.txt",["Average",total_time])
		return _temp_average
	except:
		log("log.txt",["Average","FAILED"])
		return 0

def std_deviation(LIST,d=5):
	"""Returns the standard deviation of a dataset (LIST), rounded to d decimal points."""
	try:
		start_time = ct()
		avg = average(LIST)
		dev = [(LIST[i]-avg)**2 for i in range(len(LIST))]
		log("log.txt",["Std Deviation",ct()-start_time])
		_temp_std_dev = average(dev,d)
	except:
		log("log.txt",["Std Deviation","FAILED"])
		_temp_std_dev = -1	
	finally:
		return _temp_std_dev

def remove_outliers(LIST):
	"""Removes numerical data points from a LIST that are more than 2 standard deviations from the mean."""
	try:	
		start_time = ct()
		avg = average(LIST)
		std = std_deviation(LIST)
		return_list = []
		count = 0
		for i in range(0,len(LIST)):
			if abs(LIST[i]-avg) < std*2:
				return_list.append(LIST[i])
			else:
				count += 1
		print("Outliers removed from dataset ("+str(count)+").")
		log("log.txt",["Remove Outliers",ct()-start_time,count])
	except:
		log("log.txt",["Remove Outliers","FAILED"])
		return_list = LIST
	return return_list

def transform(LIST,METHOD):
	"""Performs the FUNCTION on every member and submember of list, then returns the transformed list. Takes two arguments, the LIST and the FUNCTION."""
	start_time = ct()
	return_list = []
	for each_subitem in LIST:
		if type(each_subitem) is list:
			return_list.append(transform(each_subitem,METHOD))
		else:
			try:
				return_list.append(METHOD(each_subitem))
			except:
				#print("ERROR: Couldn't perform the operation on the subitem.")
				return_list.append(each_subitem)
	log("log.txt",["Transform",ct()-start_time])
	return return_list

def r0to1():
	return random.randint(0,10)/10

class neural_net:
	def __init__(self,LAYERS=3, NEURONS=3,LR=13.1):
		#initializes the arrays of the neural network with LEVELS levels and NEURONS neurons in each level
		start_time = ct()
		print("Building neural net.")
		self.n_input = [[0 for i in range(0,NEURONS)] for j in range(0,LAYERS)]
		self.n_output = [[0 for i in range(0,NEURONS)] for j in range(0,LAYERS)]
		self.n_dte_no = [[0 for i in range(0,NEURONS)] for j in range(0,LAYERS)]
		self.c_weight = [[[r0to1() for i in range(0,NEURONS)] for j in range(0,NEURONS)] for k in range(0,LAYERS-1)]
		#self.c_weight = [[[0 for i in range(0,NEURONS)] for j in range(0,NEURONS)] for k in range(0,LAYERS-1)]
		self.c_new_weight = [[[0 for i in range(0,NEURONS)] for j in range(0,NEURONS)] for k in range(0,LAYERS-1)]
		print("Neural net built.")
		#initialize a list of neural networks to which this one will spit its output
		self.d_nets = []
		#store the value of the length of the arrays, so we don't have to use 'len()' all the time
		#can just use len(self.n_input[0])
		self.num_neurons_in_layer = [NEURONS for i in range(0,LAYERS)] #maybe already obsolete
		self.num_neurons = NEURONS #soon to be obsolete
		self.num_layers = LAYERS
		#set some other important constants
		self.LIST_FORMAT = 0
		#and a container for the arrays
		self.numerical_arrays_container = [
			["Neuron Input Values (Sum)",self.n_input],
			["Neuron Output Values (Logistic)",self.n_output],
			["Derivative of Total Error wrt Neuron Output (Logistic)",self.n_dte_no],
			["Weight of Connection (Float)",self.c_weight]
			#["Calc. New Weight of Connection (Float)",self.c_new_weight]
						   ]
		self.print_setting = 0
		self.learning_rate = LR
		log("log.txt",["NN Init",ct()-start_time])

	def grow(self,N,M):
		"""Appends N new layers, each with M neurons, to the end of the neural network."""
		start_time = ct()
		for i in range(0,N):
			self.n_input.append([0 for i in range(0,M)])
			self.n_output.append([0 for i in range(0,M)])
			self.n_dte_no.append([0 for i in range(0,M)])
			self.c_weight.append([[r0to1() for i in range(0,M)] for j in range(0,len(self.n_input[len(self.n_input)-2]))])
			self.c_new_weight.append([[0 for i in range(0,M)] for j in range(0,len(self.n_input[len(self.n_input)-2]))])
			self.num_layers += 1
		log("log.txt",["NN Grow",ct()-start_time])

	def chop(self,INDEX=-1):
		"""Removes layer INDEX from the neural network.
		
		Takes one optional argument, INDEX (integer - the index of the layer to remove), and returns nothing. If INDEX is omitted, the method removes the final layer from the neural network. Prints an error message if the layer does not exist or if there is some other problem while popping the layer from the list."""
		start_time = ct()
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
			log("log.txt",["NN Chop","FAILED"])
		else:
			self.num_layers -= 1
			log("log.txt",["NN Chop",ct()-start_time])

	def seed(self,INPUT_ARRAY,DIRECT=1):
		"""Sets n_output for Layer 0, according to a maximum slice of INPUT_ARRAY."""
		#numSeeds = min(len(INPUT_ARRAY),len(self.n_output[0]))
		for i in range(0,len(self.n_output[0])):
			self.n_output[0][i] = INPUT_ARRAY[i%len(INPUT_ARRAY)]
		#self.n_output[0] = INPUT_ARRAY[0:numSeeds]

	def flow(self):
		"""Computes all n_outputs as function of respective n_inputs, then accumulates n_output in n_input of Layer n+1, modulated by c_weights. Cascades forward layer by layer.
		
		Takes no arguments, returns nothing."""
		for i in range(0,len(self.n_output)):
			for j in range(0,len(self.n_input[i])):
				#skip first layer because outputs thereof already decided by seed
				if i > 0:
					self.n_output[i][j] = self.squish(self.n_input[i][j])
				#note: don't need if statement checking for layer number because we already limit it in first for statement
				if i < len(self.n_input)-1:
					for k in range(0,len(self.c_weight[i][j])):
						#increase the input sum of each of the neurons in the layer i+1 by the output of the current neuron in layer i, multiplied by the weight of the connection to that L(i+1) neuron
						#? possible to use sum function somehow? not necessary?
						#print("layer",i)
						#print("neuron",j)
						#print("connection",k)						
						#print(self.n_input[i+1][k])
						#print(self.c_weight[i][j][k])
						#print(self.n_output[i][j])
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
			#self.num_neurons -= 1

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
		if FORMAT == 0:
			return self.n_output[self.num_layers-1]
	
	def feed(self,DESTINATION):
		"""'Feeds' the final layer's output to DESTINATION, a neural network object."""
		DESTINATION.seed(self.spit())

	def error(self,TARGET_ARRAY,d=3):
		return round(sum(self.squerr(self.n_output[self.num_layers-1][i],TARGET_ARRAY[i]) for i in range(0,len(self.n_output[len(self.n_output)-1]))),d)

	def print(self,d=2):
		"""Prints each of the principal arrays to the console. Takes one optional argument, 'd,' which is the number of decimal places to round the output values."""
		roundx = lambda x : round(x,d)
		print_arrays = transform(self.numerical_arrays_container,roundx)
		for each_subarray in print_arrays:
			print(each_subarray[0])
			for each_layer in each_subarray[1]:
				print(each_layer)

	def wipe(self):
		"""Resets neural net inputs, outputs, and connection deltas. Does *not* reset connection weights. Takes no arguments."""
		for i in range(0,len(self.n_output)):
			for j in range(0,len(self.n_output[i])):
				self.n_input[i][j] = 0
				self.n_output[i][j] = 0
				self.n_dte_no[i][j] = 0

	def squish(self,N):
		"""Runs the logistic function on numerical input N. Returns result"""
		return 1 / ( 1 + math.e**(-N) )
		

	def squerr(self,ACTUAL_OUTPUT,TARGET_OUTPUT):
		"""Returns the squared error, given:
			- the ACTUAL_OUTPUT
			- and the TARGET_OUTPUT."""
		return 0.5 * (TARGET_OUTPUT - ACTUAL_OUTPUT) ** 2 

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
		for i in range(0,len(self.c_weight)):
			for j in range(0,len(self.c_weight[i])):
				for k in range(0,len(self.c_weight[i][j])):
					self.c_weight[i][j][k] = self.c_new_weight[i][j][k]
					self.c_new_weight[i][j][k] = 0

	def revise(self,CORRECT_OUTPUT):
		"""Updates the 'new weight' 3-dimensional array pre-update.
			Takes one argument, CORRECT_OUTPUT, a one-dimensional numerical array that represents the correct outputs, 0 to n, of the final layer of the array."""
		#starts at layer total-1 because there are no connections going from last layer
		for i in reversed(range(0,len(self.n_output))):
			for j in range(0,len(self.n_output[i])):
				if i == len(self.n_output)-1:
					#for ease of reading, store the derivative of total error with respect to neuron output
					temp_d_te_out = self.n_output[i][j]-CORRECT_OUTPUT[j]
					#now store the derivative of the neuron output with respect to the neuron input, which is the derivative of the logistic function
					temp_d_op_net = ( self.n_output[i][j] * ( 1 - self.n_output[i][j] ) )
					#now store the produce of these two values in the array index that corresponds with that neuron index (e.g., if we're just starting out: the first index of the final list. And since we're starting in the second to last layer, we need to store it in layer i+1 the first time around.)
					self.n_dte_no[i][j] = (-1) * temp_d_te_out * temp_d_op_net
				else:
					#if we're not on the final layer, we are calculating the derivative of total error with respect to the destination neuron output. We calculate that by summing the derivatives of the destination output error with respect to the current neuron output, which we calculated in previous steps
					#As clarification, this is meant to be the sum of partial derivatives
					#each partial derivative is represented by the product of:
					# a connection weight (index m) going from a shared neuron to various neurons in the next layer
					# the derivative of total error with respect to the destination neuron
					# #This sum is generated using a generator."""
					#temp_d_te_out = sum( [ self.n_dte_no[i+1][k] * self.c_weight[i][j][k] for k in range(0,self.num_neurons) ] )
					temp_d_te_out = sum( [ self.n_dte_no[i+1][k] * self.c_weight[i][j][k] for k in range(0,len(self.n_output[i+1])) ] )
					#print(i,j,temp_d_te_out)
					#Now this value is stored in an index of the array of partial derivatives so we can use it during the next backpropagation step. Wow! Let's take a deep breath, maybe call some friends and have a glass of water.
					self.n_dte_no[i][j] = temp_d_te_out
		pass
		for i in range(0,len(self.n_output)-1):
			for j in range(0,len(self.n_output[i])):
				for k in range(0,len(self.n_output[i+1])):
					#record the new weight of each connection
					self.c_new_weight[i][j][k] = self.scale(self.n_output[i+1][k],
													self.n_output[i][j],
													self.n_dte_no[i+1][k],
													self.c_weight[i][j][k],
													self.learning_rate)	
		pass

	def train(self,INPUT_ARRAY,ANSWER_ARRAY,it=1):
		"""Runs arg1 INPUT_ARRAY through the neural network, evaluates the result next to ANSWER_ARRAY, then adjusts the connection weights in the neural network to reduce the total error on the next go."""
		#munch INPUT_ARRAY
		#compare result to ANSWER_ARRAY
		#calculate delta for each connection
		#update connection weights with delta
		#print a report of what happened, inc. total error
		first_time = ct()
		error_initial = 0
		error_final = 0
		for i in range(0,it):
			#print("Iterations of training example:",it,end='\r')
			self.munch(INPUT_ARRAY)
			self.revise(ANSWER_ARRAY)
			self.fresh()
			if i == 0:
				error_initial = self.error(ANSWER_ARRAY,3)
			if i == it-1:
				if self.print_setting == 1:
					self.print()
				error_final = self.error(ANSWER_ARRAY,3)
			self.wipe()
			second_time = ct()
			#print(str(round(i/it*100,4))+"% done",end='\r')
		#self.print()
		if self.print_setting == 1:
			print("Iterated over training example "+str(it)+" times in "+str(round((second_time-first_time)*1000,3))+" ms.\nTotal error is now "+str(round(error_final,7))+"\nError decreased by "+str(round(error_initial-error_final,7))+" over total iterations.")
		return error_final

	def munch(self,INPUT_ARRAY,DO_PRINT=0):
		"""Seeds INPUT_ARRAY to the top layer of the neural network, pushes it through subsequent layers, and then returns the output of the final layer. This is the "black box." """
		
		self.seed(INPUT_ARRAY)
		self.flow()
		if DO_PRINT:
			print("\nReceived input: "+str(INPUT_ARRAY))
			print("Performed a calculation.")
		output = self.spit()
		return output

	def set(self,TUPLES,THRESHOLD):
		"""Executes a set of training examples (TUPLES), stopping the training program once the total error over the set drops below THRESHOLD"""
		
		start_time = ct()
		error_occurred = 0
		
		if TUPLES == []:
			print("Can't execute an empty training set.")
			error_occurred = 1
		elif THRESHOLD <= 0:
			print("Error threshold must be greater than zero.")
			error_occurred = 1
		if error_occurred:
			print("Aborted training set.")
			log("log.txt",["NN Set","FAILED DURING INIT"])
			return 0

		last_error = 0
		set_error = 0
		error_list = [100 for i in range(0,20)]
		dataset = []
		start_time = ct()
		iterations_record = 0
		print('\n')

		while True:

			for i in range(0,len(TUPLES)):
				set_error += self.train(*TUPLES[i])
			
			print("Executing training set.\tWhole set iterations:",str(iterations_record)+". Error per neuron = "+str(set_error/len(self.n_output[len(self.n_output)-1])),end="\r")
			log("set-errors.txt",[iterations_record,round(set_error,5)])
			
			error_list = error_list[1:]
			error_list.append(set_error)
			iterations_record += 1

			if slope(error_list[-10:]) >= 0.1:
				print("\nError stopped decreasing ("+str(set_error)+")")
				print(slope(error_list[-10:]))
				self.learning_rate *= .95
				log("log.txt",[
					"Log Type",		"Error",		"Layers",			"Inputs",				"Outputs",									"Iterations",
					"\nSET FAIL",		set_error,		len(self.n_input),	len(self.n_input[0]),	len(self.n_output[len(self.n_input)-1]),	iterations_record	
					])
			
			
			if set_error < THRESHOLD:
				log("log.txt",[
					"Log Type",		"Error",		"Layers",			"Inputs",				"Outputs",									"Iterations",
					"\nSET SUCCESS",	set_error,		len(self.n_input),	len(self.n_input[0]),	len(self.n_output[len(self.n_input)-1]),	iterations_record	
					])
				break

			set_error = 0
		
		end_time = ct()
		log("log.txt",["NN Set","SUCCESS",end_time-start_time])
		print("Completed training set in "+str(round(end_time-start_time,4))+" seconds after "+str(iterations_record)+" iterations through the set.")
		return set_error

	def screw(self,X):
		"""Choose a random connection and add 100 to the weight."""
		i = random.randint(0,self.num_layers-2)
		j = random.randint(0,len(self.n_input[i])-1)
		k = random.randint(0,len(self.c_weight[i][j])-1)
		self.c_weight[i][j][k] *= X

	def spit_char(self):
		out = self.spit()
		out_char = chr(int("".join([str(int(out[i])) for i in range(0,len(out))]),2))
		return out_char

	def data(self,MIN_LAYERS=3,MAX_LAYERS=3,MIN_NEURONS=2,MAX_NEURONS=2,MIN_LEARNING_RATE=13,MAX_LEARNING_RATE=13,SAMPLESIZE=10):
		
		dataset = [['NWIDTH','LR','AVG1','STD1','AVG2','STD2']]
		
		for clayer in range(MIN_LAYERS,MAX_LAYERS+1):
			for nwidth in range(MIN_NEURONS,MAX_NEURONS+1):
				for lr in range(MIN_LEARNING_RATE,MAX_LEARNING_RATE+1):

					net = [neural_net(clayer,nwidth,lr) for i in range(0,SAMPLESIZE)]

					times = []
					for i in range(0,len(net)):
						start_time = ct()
						net[i].set(training_list,0.1)
						times.append(ct()-start_time)
	
					avg1 = average(times,4)
					std1 = std_deviation(times,4)
					times_wo_outliers = remove_outliers(times)
					avg2 = average(times_wo_outliers,4)
					std2 = std_deviation(times_wo_outliers,4)
		
					print("\nAverage time per set:",avg1,"seconds. Standard deviation:",std1)
					print("Average time per set (w/o outliers):",avg2,"seconds. Standard deviation (w/o Os):",std2)

					_temp_data = [nwidth,lr,avg1,std1,avg2,std2]
					dataset.append(_temp_data)

					log("log.txt",_temp_data)

		with open("data.txt","w+") as file1:

			for i in range(0,len(dataset)):
				print(*dataset[i],sep='\t')
				print(*dataset[i],sep='\t',file=file1)

		file1.close()
