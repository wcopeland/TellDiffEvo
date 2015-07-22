__author__ = 'Wilbert'

from datetime import datetime
import math
import numpy as np
import random
import tellurium as te


class Optimization(object):
    def __init__(self, model, measurements=None, parameter_map=None):
        random.seed()
        self.model = model
        self.measurements = measurements
        self.parameter_map = parameter_map
        self.islands = []

        # Optimizer Settings
        self.MaximumGenerations = 1000
        self.FitnessThreshold = 1e-3
        self.NumberOfIslands = 1
        self.PopulationSize = 12
        self.MigrationFrequency = 0.30
        self.NumberOfMigrants = 1
        self.SelectionGroupSize = 3
        self.ReplacementGroupSize = 3
        self.MigrationTopology = range(self.NumberOfIslands)[1:] + [0]
        self.CPUs = 1
        return

    def LoadMeasurements(self, filename, header=True, decimal_precision=2):
        # Read tab-separated file
        file = open(filename, 'r')
        data = [[y for y in x.replace('\n','').split('\t') if y != ''] for x in file.readlines()]
        if header is True:
            headings = data[0]
            data = data[1:]
        else:
            headings = ['measurement{0}'.format(x) for x in range(len(data[0]))]
        file.close()

        # Store measurements
        self.measurements = {k:[] for k in headings}
        for row in data:
            for index in range(len(row)):
                self.measurements[headings[index]].append(round( float(row[index]),decimal_precision))
        return

    def CreateParameterMap(self):
        self.parameter_map = {}
        for key in self.model.getGlobalParameterIds():
            info = ParameterInfo(index=len(self.parameter_map))
            self.parameter_map.update({key:info})
        return

    def GetValue(self, key):
        try:
            return self.model.__getattr__(key)
        except:
            return LookupError

    def SetValue(self, key, value):
        try:
            self.model.__setattr__(key, value)
        except:
            return LookupError

    def FixParameter(self, key):
        self.parameter_map[key].fixed = True
        return

    def UnfixParameter(self, key):
        self.parameter_map[key].fixed = False
        return

    def AssignVectorToModel(self, vector):
        for key, pinfo in self.parameter_map.items():
            self.SetValue(key, vector[pinfo.index])
        return

    def Run(self):
        random.seed()
        print('\n\nStarting differential evolution routine...')
        clock = datetime.now()

        self.PopulationSize = max([self.PopulationSize, 4])
        self.islands = []
        for i in range(self.NumberOfIslands):
            self.islands.append(self.CreateIsland())

        generation_count = 0
        while generation_count < self.MaximumGenerations:
            generation_count += 1
            for island in self.islands:
                population_samples = [random.sample(island, k=3) for x in range(len(island))]
                for member_index in range(len(island)):
                    island[member_index] = self.ChallengeMember(island[member_index], population_samples[member_index])
            self.SortIslandsByFitness()
            if generation_count % 10 == 0:
                print('\n')
                for island in self.islands:
                    for member_index in range(len(island)):
                        print('Fitness: {0}\tVector: {1}'.format(round(island[member_index].fitness, 4), island[member_index].vector))
            if min([x[0].fitness for x in self.islands]) < self.FitnessThreshold:
                break

        # Print final report
        for island in self.islands:
            for member_index in range(len(island)):
                print('Fitness: {0}\tVector: {1}'.format(round(island[member_index].fitness, 4), island[member_index].vector))
        print('\nCompleted differential evolution routine. Total time: {0}'.format(datetime.now()-clock))
        return

    def CreateIsland(self):
        island = []
        for i in range(self.PopulationSize):
            island.append(self.CreateRandomMember())
        return island

    def CreateRandomMember(self):
        success = False
        while not success:
            try:
                vector = [None] * len(self.parameter_map)
                for key in self.parameter_map:
                    i = self.parameter_map[key].index
                    if self.parameter_map[key].fixed:
                        vector[i] = self.GetValue(key)
                    else:
                        vector[i] = 10 ** random.uniform(*self.parameter_map[key].log_range)
                assert None not in vector, "Error creating the initial optimization vector. Not all parameters were assigned a value."
                fitness = self.GetFitness(vector)
                success = True
                print('Created a random member.\tFitness: {0}\tVector:{1}'.format(fitness, vector))
            except:
                pass
        return Member(vector, fitness)

    def ChallengeMember(self, original, samples, CR=0.6, F=0.8):
        o = original.vector
        a,b,c = [x.vector for x in samples]

        # Basic mutation
        vector = []
        for i in range(len(o)):
            if random.random() <= CR:
                v = a[i] + F * (b[i]-c[i])
                if v > 0:
                    vector.append(v)
                else:
                    vector.append(o[i]/2.)
            else:
                vector.append(o[i])

        # Check that all vector values conform to the parameter constraints.
        for key, value in self.parameter_map.items():
            # Reset to original if value was supposed to be fixed.
            if value.fixed:
                i = value.index
                vector[i] = o[i]
            # Resample within log_range if value exceeds bounds.
            if vector[i] < value.minimum or vector[i] > value.maximum:
                vector[i] = 10 ** random.uniform(*value.log_range)

        # Evaluate the fitness of the solution.
        try:
            fitness = self.GetFitness(vector)
            if fitness < original.fitness:
                return Member(vector, fitness)
            else:
                return original
        except:
            return original

    def GetFitness(self, vector):
        self.model.reset()
        self.AssignVectorToModel(vector)

        simulated_result = self.model.simulate(0, 15, 300)
        simulated_result = np.asarray([[y for y in x] for x in simulated_result])
        entities = [x.replace('[','').replace(']','') for x in opt.model.selections]
        result = {entities[i]:simulated_result[:,i] for i in range(len(entities))}

        """
        # Set simulation resolution small enough to capture measured time points.
        t_start, t_end = result['time'][0], result['time'][-1]
        steps = t_end - t_start
        timeout = 0
        while True and timeout < 4:
            timeout += 1
            if int(steps) == float(steps):
                break
            else:
                steps *= 10.0
        steps = int(steps) + 1
        """

        # Fitness is calculated as sum of differences squared.
        fitness = 0.
        for key, data in self.measurements.items():
            if key.lower() is not "time":
                observed = data
                expected = result[key]
                assert len(observed) == len (expected), "Error. Observed and expected arrays are not the same size."
                fitness += sum((expected-observed)**2)
        return fitness

    def SortIslandsByFitness(self):
        for i in range(len(self.islands)):
            self.islands[i] = sorted(self.islands[i], key=lambda o: o.fitness)
        return

class Member(object):
    def __init__(self, vector, fitness):
        self.vector = vector
        self.fitness = fitness
        return

class ParameterInfo(object):
    def __init__(self, index, fixed=False, minimum=None, maximum=None, log_range=None):
        # Index of this parameter within the optimization vector.
        self.index = index
        # Indicates whether the parameter value is fixed within the optimization vector. (ie. Do not optimize!)
        self.fixed = fixed
        # If optimization routine produce a value outside of the minimum and maximum, a new value is randomly sampled
        # within the specified log10 range.
        self.minimum = minimum if minimum is not None else 0.
        self.maximum = maximum if maximum is not None else 10.
        self.log_range = log_range if log_range is not None else (0.,1.)
        return

soe = """
-> A; vin
A + E -> AE; k1*A*E
AE -> A + E; k2*AE
AE -> E + B; k3*AE
B -> ; vout

vin = 0.62
vout = 0.43
k1 = 0.35
k2 = 0.018
k3 = 1.6

A = 5.
AE = 0.
E = 3.
B = 0.
"""

"""
# Create exact results
result = opt.model.simulate(0,15,300)
result = np.asarray([[y for y in x] for x in result])
entities = [x.replace('[','').replace(']','') for x in opt.model.selections]
opt.model.plot()
print('{}\t{}\t{}\t{}\t'.format(*entities))
for i in range(len(result)):
    print('{}\t{}\t{}\t{}\t'.format(*result[i]))
"""

opt = Optimization(model=te.loadAntimonyModel(soe))
opt.LoadMeasurements('data2.txt')
opt.CreateParameterMap()
#opt.FixParameter('k1')
#m = opt.CreateRandomMember()

opt.Run()

print('\n')
for k in opt.parameter_map:
    print('{}={}'.format(k,opt.model.__getattr__(k)))