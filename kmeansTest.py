import pymprog
import numpy
import csv
import math
import scipy.cluster.vq
import datetime
import random
import decimal
import copy

# import station data to perform clustering
reader=csv.reader(open("altFuelStations_cleaned.csv","rU"),delimiter=',')
x=list(reader)
result=numpy.array(x)

# isolate hydrogen sites  
# Description in Col1, Station Latitude in Col2, Longitude in Col3
HYstationLatLongAll = []
HYstationLatLong = []
HYstationLatLongAll = result[result[:,3] == "HY"]
HYstationLatLong = HYstationLatLongAll[:, [1,2]]
HYstationLatLong = HYstationLatLong.astype(float)

# isolate natural gas sites
NGstationLatLongAll = []
NGstationLatLong = []
NGstationLatLongAll = result[result[:,3] != "HY"]
NGstationLatLong = NGstationLatLongAll[:, [1,2]]
NGstationLatLong = NGstationLatLong.astype(float)

# initialize gas station network
reader = csv.reader(open("gas_agg.csv","rU"),delimiter=',')
x = list(reader)
GasstationLatLong = []
GasstationLatLongAll = numpy.array(x)
GasstationLatLong = GasstationLatLongAll[:, [2,3]]
GasstationLatLong = GasstationLatLong.astype(float)


# import demand file/whatever else
NGdemandArray = numpy.ndarray(shape=(len(NGstationLatLong),1), dtype=float, order='F')
for i in range (0, len(NGstationLatLong)):
    NGdemandArray[i] = random.random()*1000000
    # TODO - convert demand to m^3

HYdemandArray = numpy.ndarray(shape=(len(HYstationLatLong),1), dtype=float, order='F')
for i in range (0, len(HYstationLatLong)):
    HYdemandArray[i] = random.random()*1000000
    # TODO - convert demand to m^3

GasdemandArray = numpy.ndarray(shape=(len(GasstationLatLong), 1), dtype = float, order = 'F')
for i in range (0, len(GasstationLatLong)):
    GasdemandArray[i] = random.random()*1000000
    # TODO - convert demand to m^3

# What are maximum number of plants that the total US demand could support?
# Total US demand divided by the minimum viable H2 production plant capacity
maxPlantDemand = numpy.sum(GasdemandArray)
plantMinCapacity = 10000000000
maxPlantNumber = numpy.ceil(maxPlantDemand/plantMinCapacity)

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

    # TODO - initialize with existing plants and fix random initialization. 

# should be maxPlantNumber
for kNumber in range(1,2, 1000): 
    print datetime.datetime.now()
    print kNumber

    # perform the clustering
    # centroid is the output of kNumber centroids and label is the assigning of stations in the
    # lat/long array to each centroid. 
    centroid, label = scipy.cluster.vq.kmeans2(GasstationLatLong, kNumber, iter=100, thresh=1e-05, minit='random', missing='warn')
    plantArray = numpy.ndarray(shape=(len(label),2), dtype=float, order='F')


    # now add centroid lat/longs to result (import file) to calculate distances
    # copy labels, then replace going along for i in len(centroid)
    for i in range(0, len(label)):
        groupID = label[i] 
        plantCenter = centroid[groupID, :]
        plantArray[i] = plantCenter


    # aggregate gas station demand by plant (NG stations already have supply, so do H2 stations). We need to know total
    # demand for gas stations. Since the other types of stations already have a distribution mechanism, and any new
    # production facilities will have to supply aggregate demand of clustered gas stations. 
    label = numpy.reshape(label, (len(label), 1))
    label = numpy.append(label, GasdemandArray, axis = 1)
    plantDemandArray = numpy.ndarray(shape=(len(centroid),2), dtype=float, order='F')

    for i in range(0, kNumber):
        demandAgg = label[label[:, 0] == i]
        if len(demandAgg) == 0: 
            plantDemandArray[i, :] = [i, 0] 
        else:
            demandAgg = demandAgg[:, 1]
            plantDemand = numpy.sum(demandAgg)
            plantDemandArray[i, :] = [i, plantDemand]

            
    #generate distances between production plant and gas stations in particular cluster
    latLongDistArray = numpy.ndarray(shape=(len(GasstationLatLong),1), dtype=float, order='F')
    for i in range(0, len(GasstationLatLong)):
        latLongDistArray[i] = distance_on_unit_sphere(GasstationLatLong[i, 0], GasstationLatLong[i,1], plantArray[i,0], plantArray[i,1])

    #add distances to label file
    label = numpy.append(label, latLongDistArray, axis = 1)

    # now set up decision criteria for distribution
    # calculate costs of each distribution method, then pick the lowest and
    # assign it to a station. 
    def costPipeline(distance, demand):
        # set up pipeline constraint
        pipeCapacity = 50000000 # m^3/ mi
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
        truckNumber = math.ceil((demand)/truckCapacity) # round up
        totalTruckCost = unitTruckCost * truckNumber
        return totalTruckCost

    # reformer cost = reformer * number of stations but demand constraint
    def costReformer(demand):
        #capital costs
        NGCost = 5 # $ per kg of H2 created
        reformerCapacity = 5000
        reformerCost = 3
        reformerNumber = math.ceil(demand/reformerCapacity) #round up
        totalReformerCost = reformerNumber * reformerCost + NGCost*demand
        return totalReformerCost


    # Plant Production Cost
    def plantCost(aggDemand):
        if aggDemand<5000000000:
            plantCost = 2000000000
        elif aggDemand>10000000000:
            plantCost = 5000000000
        else:
            plantCost = 3000000000
        return plantCost




    def assignProduction(plantTotalCost, aggDemand, clusterNodes):
        # figure out what kind of edge to have
        # prodCost is portion of plant total cost assigned to that node
        prodCost = plantTotalCost/aggDemand
        reformCost = numpy.ndarray((len(clusterNodes),1), dtype = float)
        for i in range(0, len(clusterNodes)):
            reformCost[i] = costReformer(clusterNodes[i, 1].astype(float))    
        edgeDecision = numpy.ndarray(shape=(len(clusterNodes), 1), dtype='S20')
        edgeDecision[:] = "Plant"

        print edgeDecision.shape
        print clusterNodes.shape
        print reformCost.shape
        clusterNodes = numpy.append(clusterNodes, reformCost, axis = 1)
        clusterNodes = numpy.append(clusterNodes, edgeDecision, axis = 1)
        transfer = float('inf')
        while transfer != 0:
            updateArray = copy.copy(clusterNodes)
            subAggDemand = numpy.sum(clusterNodes[:, 1].astype('float'))
            prodCost = plantTotalCost/subAggDemand
            for i in range(0, len(clusterNodes)):
                edge = clusterNodes[i, 2]
                edgePipeCost, edgePipeCap = costPipeline(edge.astype("float32"), subAggDemand)
                if edgePipeCap == True:
                    if edgePipeCost + prodCost > reformCost[i]:
                        clusterNodes[i, 4] = "Reformer"
                    else:
                        clusterNodes[i,4] = "Plant"
                else:
                    clusterNodes[i, 4] = "Plant"
            transfer = 0
            for i in range(0, len(clusterNodes)):
                if updateArray[i,4].astype(str) != clusterNodes[i,4].astype(str):
                    transfer +=1
        return clusterNodes

#TODO - foregone gas profits
    DecisionArray = numpy.ndarray(shape=(len(GasstationLatLong), 5))
    start = 0
    for i in range(0, kNumber):
        clusterNodes = label[label[:, 0] == i]
        row = len(clusterNodes)
        aggDemand = plantDemandArray[i, 1].astype("float32")
        plantTotalCost = plantCost(aggDemand)
        Decision = assignProduction(plantTotalCost, aggDemand, clusterNodes)
        DecisionArray[start:row, :] = DecisionArray
        start = row + 1
        # cost multiplication

                             

    distributionCostArray = numpy.ndarray(shape=(len(HYstationLatLong),3), order='F')
    #for i in range(0, len(HYstationLatLong)):
#        edgePipeCost, edgePipeBool = costPipeline(latLongDistArray[i].astype("float32"), HYdemandArray[i].astype("float32"))
#        edgeTruckCost = costTruck(latLongDistArray[i].astype("float32"), HYdemandArray[i].astype("float32"))
#        distributionCostArray[i] = [edgePipeCost, edgePipeBool, edgeTruckCost]


#    nodeProductionCostArray = numpy.ndarray(shape=(len(NGstationLatLong), 1), order='F')
#    for i in range(0, len(NGstationLatLong)):
#        nodeReformerCost = costReformer(NGdemandArray[i].astype("float32"))
#        nodeProductionCostArray[i] = nodeReformerCost


    # Production cost at plant - assume same costs over all regions

#    def plantCapitalOperatingCosts(aggDemand):
#        for i in range(1, len(plantDemandArray)):
#            if plantDemandArray[i] < 200: # some capacity       
#                plantFixedCost = 5
#                plantOpCost = 1* plantDemandArray[i].astype('float32')
#            elif plantDemandArray[i] >= 200: #more options available
#                plantFixedCost = 10
#                plantOpCost = 1* plantDemandArray[i].astype('float32')

#        return plantCapitalOperatingCost

       

    print label[0:20]
    print datetime.datetime.now()
    print
    print
    print
    



    # TO FIX
    # catch all non-positive inequalities
    # make sure last array entries aren't getting dropped
    # include nodeStorage cost & capacity
    # cost of reforming, but will have to write in preclusion of other network
    # and only use NG stations
    #SAVE EVERYTHING BACK TO HYresult and NYresult

    #distributionCostArray = numpy.concatenate(([["Edge Pipe Cost", "Edge Pipe Bool", "EdgeTruckCost"]], distributionCostArray), axis = 0)
    #nodeProductionCostArray = numpy.concatenate(([["Total Node Reformer Cost"]], nodeProductionCostArray), axis = 0)




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






        


