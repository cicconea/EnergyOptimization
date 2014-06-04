import pymprog
import numpy
import csv
import math
import scipy.cluster.vq
import datetime
import random
import decimal
import copy

print "we are starting at ", datetime.datetime.now()

#TODO - decide demand # ie steady state year or total lifetime
#TODO - input national demand
nationalDemand = 100000000
totalGasNetworkCostSum = float('inf')

# import station data to perform clustering
reader=csv.reader(open("traffic_vol_calc.csv","rU"),delimiter=',')
x=list(reader)
result=numpy.array(x)
resultHeader = result[0,:]
result = result[1:, :]

# isolate hydrogen sites  
# Order is: lat | long | brand | id | type | volume | # roads | weight
HYstationLatLongAll = []
HYstationLatLong = []
HYstationLatLongAll = result[result[:,4] == "HY"]
HYstationLatLong = HYstationLatLongAll[:, [0,1]]
HYstationLatLong = HYstationLatLong.astype(float)

# isolate natural gas sites
NGstationLatLongAll = []
NGstationLatLong = []
NGstationLatLongAll = result[result[:,4] != "Gas"]
NGstationLatLongAll = NGstationLatLongAll[NGstationLatLongAll[:,4] != "HY"]
NGstationLatLong = NGstationLatLongAll[:, [0,1]]
NGstationLatLong = NGstationLatLong.astype(float)

# initialize gas station network
GasstationLatLong = []
GasstationLatLongAll = result[result[:,4] == "Gas"]
GasstationLatLong = GasstationLatLongAll[:, [0,1]]
GasstationLatLong = GasstationLatLong.astype(float)


GasdemandArray = numpy.multiply(GasstationLatLongAll[:,7].astype(float), nationalDemand)
NGdemandArray = numpy.multiply(NGstationLatLongAll[:,7].astype(float), nationalDemand)
HYdemandArray = numpy.multiply(HYstationLatLongAll[:,7].astype(float), nationalDemand)


# What are maximum number of plants that the total US demand could support?
# Total US demand divided by the minimum viable H2 production plant capacity
maxPlantDemand = numpy.sum(GasdemandArray)
plantMinCapacity = 5000000
maxPlantNumber = numpy.ceil(maxPlantDemand/plantMinCapacity)

print maxPlantNumber


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

# TODO - should be maxPlantNumber
for kNumber in range(1, maxPlantNumber, 1 ): 
    print "kNumber ", kNumber, " is starting at ", datetime.datetime.now()

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
    GasdemandArray = numpy.reshape(GasdemandArray, (len(GasdemandArray), 1))

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
    # we use these to calculate trucking costs over time
    meanDist = numpy.mean(latLongDistArray)
    stdDist = numpy.std(latLongDistArray)


    # now set up decision criteria for distribution: reforming, electrolysis, and centralized
    # reformer cost = reformer * number of stations but demand constraint
    def costReformer(demand):
        reformerH2Cost = 10.3 #$/kg high side estimate. Lower bound is $7.7
        #taken from NREL analysis http://www.nrel.gov/hydrogen/pdfs/54475.pdf
        totalReformerCost = reformerH2Cost*demand
        return totalReformerCost

    # electrolysis cost
    def costElectrolysis(demand):
        ElectrolysisH2Cost = 12.9 #$/kg high side estimate. Lower bound is $10.0
        #taken from NREL analysis http://www.nrel.gov/hydrogen/pdfs/54475.pdf
        totalElectrolysisCost = ElectrolysisH2Cost*demand
        return totalElectrolysisCost

    # Plant Production Cost
    def plantCost(aggDemand, distance):
        plantProdCost = 10.33 # $per kg H2 produced
        highPlantCost = plantProdCost * 1.25
        lowPlantCost = plantProdCost * 0.75
        if distance > meanDist + stdDist:
            plantCost = aggDemand*highPlantCost
        elif distance < meanDist - stdDist:
            plantCost = aggDemand * lowPlantCost
        else:
            plantCost = aggDemand * plantProdCost
        return plantCost



    def assignProduction(clusterNodes):
        # figure out what kind of edge to have
        # set up reformer cost calculation - what is setup cost of reformer + H2 Production cost
        reformCost = numpy.ndarray((len(clusterNodes),1), dtype = float)
        for i in range(0, len(clusterNodes)):
            reformCost[i] = costReformer(clusterNodes[i, 1].astype(float))    
        ElectrolysisCost = numpy.ndarray((len(clusterNodes),1), dtype = float)
        for i in range(0, len(clusterNodes)):
            ElectrolysisCost[i] = costElectrolysis(clusterNodes[i, 1].astype(float)) 
        CentralizedCost = numpy.ndarray((len(clusterNodes),1), dtype = float)
        for i in range(0, len(clusterNodes)):
            centralAggDemand = clusterNodes[i, 1].astype(float)
            centralDistance = clusterNodes[i,2].astype(float)
            CentralizedCost[i] = plantCost(centralAggDemand, centralDistance) 


        edgeDecision = numpy.ndarray(shape=(len(clusterNodes), 1), dtype='S20')
        edgeCost = numpy.ndarray(shape=(len(clusterNodes), 1), dtype=float)
        for i in range(0, len(clusterNodes)):
            if reformCost[i]< ElectrolysisCost[i]:
                if reformCost[i] < CentralizedCost[i]:
                    edgeDecision[i] = "Reformer"
                    edgeCost[i] = numpy.multiply(reformCost[i], clusterNodes[i,1])
            elif ElectrolysisCost[i] < reformCost[i]:
                if ElectrolysisCost[i] < CentralizedCost[i]:
                    edgeDecision[i] = "Electrolysis"
                    edgeCost[i] = numpy.multiply(ElectrolysisCost[i], clusterNodes[i,1])
            else:
                edgeDecision[i] = "Centralized"
                edgeCost[i] = numpy.multiply(CentralizedCost[i], clusterNodes[i,1])

        clusterNodes = numpy.append(clusterNodes, edgeDecision, axis = 1)
        clusterNodes = numpy.append(clusterNodes, edgeCost, axis = 1)
        return clusterNodes
    
#TODO - foregone gas profits
    start = 0
    totalDecision = []
    for i in range(0, kNumber):
        clusterNodes = label[label[:, 0] == i]
        row = len(clusterNodes)
        aggDemand = plantDemandArray[i, 1].astype("float32")
        Decision = assignProduction(clusterNodes)
        if i == 0:
            totalDecision = Decision
        else:
            totalDecision = numpy.append(totalDecision, Decision, axis = 0)
   
    print "totalDecision complete at ", datetime.datetime.now()


    print totalDecision[0:50]


    networkCostArray = totalDecision[:, 4].astype(float)
    networkCost = numpy.sum(networkCostArray)
    print "Network Cost for ", kNumber, "plants is ", networkCost

    # total costs and other network details stored in:
    lowCostNetwork = numpy.ndarray(shape=(len(totalDecision), 5), dtype ='S20')

    if networkCost < totalGasNetworkCostSum:
        lowCostNetwork = totalDecision
        lowKNumber = kNumber

    totalGasNetworkCostSum = networkCost        

print 'lowest cost gas network cluster number is ', lowKNumber
numpy.savetxt("low_cost_output.csv", lowCostNetwork, fmt = '%-10s', delimiter=",")


print 'Now calculate low cost NG station'
NGstationCost = numpy.ndarray(shape=(len(NGstationLatLongAll),1), dtype=float)
for i in range(0, len(NGdemandArray)):
    # need foregone NG revenue here too!
    NGstationCost[i] = costReformer(NGdemandArray[i].astype(float))
    NGstationSum = numpy.sum(NGstationCost)
NGstationLatLongAll = numpy.append(NGstationLatLongAll, NGstationCost, axis = 1)

numpy.savetxt("ng_low_cost_output.csv", NGstationLatLongAll, fmt = '%-10s', delimiter=",")

print 'Total NG network cost is: ', NGstationSum
print 'Total Network Cost including NG stations is: ', totalGasNetworkCostSum + NGstationSum
print 'analysis complete at ', datetime.datetime.now()


# totalNetwork: clusterID, demand, distance, node type, total node cost

# to finish after - can station make profit?
#def StationProfit(stationArray, H2price, gasPrice):
#    return null



    # TO FIX
    # all production costs covered
    # need to calculate total station costs per steady-state year
    # foregone revenue from gas/NG sales
    # metric for distance to NG pipeline?






        


