__author__ = 'Wilbert'

import random
import tellurium as te


class Optimization(object):
    def __init__(self, model, measurements=None):
        random.seed()
        self.model = model
        self.measurements = measurements
        self.parameter_map = None
        self.islands = []

        # Optimizer Settings
        self.NumberOfIslands = 1
        self.PopulationSize = 20
        self.MigrationFrequency = 0.30
        self.NumberOfMigrants = 1
        self.SelectionGroupSize = 3
        self.ReplacementGroupSize = 3
        self.MigrationTopology = range(self.NumberOfIslands)[1:] + [0]
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
                fitness = None #todo
                success = True
                print('Created a random member.\tFitness: {0}\tVector:{1}'.format(fitness, vector))
            except:
                pass
        return Member(vector, fitness)

    def AssignMemberVectorToModel(self, member):
        for key, pinfo in self.parameter_map.items():
            self.SetValue(key, member.Vector[pinfo.index])
        return

class Member(object):
    def __init__(self, vector, fitness):
        self.Vector = vector
        self.Fitness = fitness
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
        self.maximum = maximum if maximum is not None else 1000.
        self.log_range = log_range if log_range is not None else (0.,3.)
        return

soe = """
-> A; vin
A + E -> AE; k1*A*E
AE -> A + E; k2*AE
AE -> E + B; k3*AE
B -> ; vout

vin = 0.2
vout = 0.3
k1 = 1
k2 = 1e-2
k3 = 1

A = 10
E = 2
B = 0
"""

opt = Optimization(model=te.loadAntimonyModel(soe))
opt.LoadMeasurements('data.txt', header=False)
opt.CreateParameterMap()
opt.FixParameter('k1')
m = opt.CreateRandomMember()
opt.AssignMemberVectorToModel(m)