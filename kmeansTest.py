import pymprog
import numpy
import csv
import math
import scipy.cluster.vq
import datetime
import random
import decimal


# import station data to perform clustering
reader=csv.reader(open("altFuelStations.csv","rU"),delimiter=',')
x=list(reader)
result=numpy.array(x)

# remove rows that are not the right fuel station type.
# options are BD, CNG, E85, ELEC, HY, LNG, LPG
result = result[result[:,1] != "BD"]
result = result[result[:,1] != "E85"]
result = result[result[:,1] != "ELEC"]
result = result[result[:,1] != "LPG"]

# separate natural gas sites vs hydrogen sites
#Station Latitude in first column, Longitude in second, and remove headers, convert to float
# compressed natural gas
CNGstationLatLong = []
CNGstationLatLong = numpy.array(CNGstationLatLong)
CNGstationLatLong = result[result[:,1] == "CNG"]
CNGstationLatLong = CNGstationLatLong[:,[20,21]]

# liquid natural gas
LNGstationLatLong = []
LNGstationLatLong = numpy.array(LNGstationLatLong)
LNGstationLatLong = result[result[:,1] == "LNG"]
LNGstationLatLong = LNGstationLatLong[:,[20,21]]

# combining natural gas ?
NGstationLatLong = []
NGstationLatLong = numpy.array(NGstationLatLong)
NGstationLatLong = numpy.append(CNGstationLatLong, LNGstationLatLong, axis = 0)

#hydrogen
HYstationLatLong = []
HYstationLatLong = numpy.array(HYstationLatLong)
HYstationLatLong = result[result[:,1] == "HY"]
HYstationLatLong = HYstationLatLong[:,[20,21]]

#convert to float
NGstationLatLong = NGstationLatLong.astype('float64')
HYstationLatLong = HYstationLatLong.astype('float64')



# import demand file/whatever else
NGdemandArray = numpy.ndarray(shape=(len(NGstationLatLong),1), dtype=float, order='F')
for i in range (0, len(NGstationLatLong)):
    NGdemandArray[i] = random.random()*1000000
    # TODO - convert demand to m^3

HYdemandArray = numpy.ndarray(shape=(len(HYstationLatLong),1), dtype=float, order='F')
for i in range (0, len(HYstationLatLong)):
    HYdemandArray[i] = random.random()*1000000
    # TODO - convert demand to m^3


# function to calculate lat and long distances
def distance_on_unit_sphere(lat1, long1, lat2, long2):
    # Convert latitude and longitude spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates on unit sphere
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    if cos > 1:
        cos = 1
    arc = math.acos( cos )

    # multiply by radius of earth in miles
    arc *= 3963.1676
    return arc


# TODO - add condition for iterative clustering until plant & station make profit
# TODO - initialize with existing plants and fix random initialization. 

kNumber = 5

# perform the clustering
# centroid is the output of kNumber centroids and label is the assigning of stations in the
# lat/long array to each centroid. 
centroid, label = scipy.cluster.vq.kmeans2(HYstationLatLong, kNumber, iter=100, thresh=1e-05, minit='random', missing='warn')
plantArray = numpy.ndarray(shape=(len(label),2), dtype=float, order='F')

#now add centroid lat/longs to result (import file) to calculate distances
# copy labels, then replace going along for i in len(centroid)

for i in range(0, len(label)):
    groupID = label[i] 
    plantCenter = centroid[groupID, :]
    plantArray[i] = plantCenter

#aggregate demand by plant
label = numpy.reshape(label, (len(label), 1))
label = numpy.append(label, HYdemandArray, axis = 1)
plantDemandArray = numpy.ndarray(shape=(len(centroid),2), dtype=float, order='F')

for i in range(0, kNumber):
    demandAgg = label[label[:, 0] == i]
    if len(demandAgg) == 0: 
        plantDemandArray[i, :] = [i, 0] 
    else:
        demandAgg = demandAgg[:, 1]
        plantDemand = numpy.sum(demandAgg)
        plantDemandArray[i, :] = [i, plantDemand]

        
#generate distances:
latLongDistArray = numpy.ndarray(shape=(len(HYstationLatLong),1), dtype=float, order='F')
for i in range(0, len(HYstationLatLong)):
    latLongDistArray[i] = distance_on_unit_sphere(HYstationLatLong[i, 0], HYstationLatLong[i,1], plantArray[i,0], plantArray[i,1])



# now set up decision criteria for distribution
# calculate costs of each distribution method, then pick the lowest and
# assign it to a station. 
def costPipeline(distance, demand, nodeStorage):
    # set up pipeline constraint
    pipeCapacity = 50000 # m^3/ mi
    unitPipeCost = 300 # $/mile
    totalPipeCost = distance * unitPipeCost    
    # Figure out if we can actually store  the right quantity in
    # pipeline setup
    if distance != 0:
        if (demand)/distance > pipeCapacity:
            capacityBool = False
            totalPipeCost = float("inf") # we make this too expensive
        else:
            capacityBool = True
    else: capacityBool = True       
    return totalPipeCost, capacityBool

def costTruck(distance, demand, nodeStorage):
    truckCapacity = 30 # m^3
    unitTruckCost = 200 # $
    # do we need to adjust cost piecewise for truck distances?
    # TODO - figure out how to integrate nodeStorage
    truckNumber = math.ceil((demand)/truckCapacity) # round up
    totalTruckCost = unitTruckCost * truckNumber
    return totalTruckCost

# reformer cost = reformer * number of stations but demand constraint
def costReformer(demand, nodeStorage):
    reformerCapacity = 5000
    reformerCost = 3000
    reformerNumber = math.ceil(demand/reformerCapacity) #round up
    totalReformerCost = reformerNumber * reformerCost
    return totalReformerCost


# node storage/cost section
storageCapacity = 30000

distributionCostArray = numpy.ndarray(shape=(len(HYstationLatLong),3), order='F')
for i in range(0, len(HYstationLatLong)):
    edgePipeCost, edgePipeBool = costPipeline(latLongDistArray[i].astype("float32"), HYdemandArray[i].astype("float32"), storageCapacity)
    edgeTruckCost = costTruck(latLongDistArray[i].astype("float32"), HYdemandArray[i].astype("float32"), storageCapacity)
    distributionCostArray[i] = [edgePipeCost, edgePipeBool, edgeTruckCost]


nodeProductionCostArray = numpy.ndarray(shape=(len(NGstationLatLong), 1), order='F')
for i in range(0, len(NGstationLatLong)):
    nodeReformerCost = costReformer(NGdemandArray[i].astype("float32"), storageCapacity)
    nodeProductionCostArray[i] = nodeReformerCost


# Production cost at plant - assume same costs over all regions

def plantCapitalOperatingCosts(aggDemand):
    for i in range(1, len(plantDemandArray)
        if plantDemandArray[i] < 200 # some capacity       
            plantFixedCost = 5
            plantOpCost = 1* plantDemandArray[i].astype('float32')
        elif plantDemandArray[i] >= 200 #more options available
            plantFixedCost = 10
            plantOpCost = 1* plantDemandArray[i].astype('float32')


    return plantCapitalOperatingCost





# TO FIX
# catch all non-positive inequalities
# make sure last array entries aren't getting dropped
# include nodeStorage cost & capacity
# cost of reforming, but will have to write in preclusion of other network
# and only use NG stations
#SAVE EVERYTHING BACK TO HYresult and NYresult

distributionCostArray = numpy.concatenate(([["Edge Pipe Cost", "Edge Pipe Bool", "EdgeTruckCost"]], distributionCostArray), axis = 0)
nodeProductionCostArray = numpy.concatenate(([["Total Node Reformer Cost"]], nodeProductionCostArray), axis = 0)




#add back column headers and reshape
#label = numpy.concatenate((["ClusterID"], label), axis = 1)
#label = numpy.reshape(label, (len(result), 1))

#plantArray = numpy.concatenate(([["PlantLat", "PlantLong"]], plantArray), axis = 0)
#plantArray = numpy.reshape(plantArray, (len(result), 2))

#latLongDistArray = numpy.concatenate((["Distance (Mi)"], latLongDistArray), axis = 1)
#latLongDistArray = numpy.reshape(latLongDistArray, (len(result), 1))

# add to our results file
#result = numpy.append(result, label, axis =1)
#result = numpy.append(result, plantArray, axis =1)
#result = numpy.append(result, latLongDistArray, axis =1)

#export everything from "result" array
#numpy.savetxt("OptOutput@{0}.csv".format(datetime.datetime.now()), result, delimiter=",", fmt="%s")






    


