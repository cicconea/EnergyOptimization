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
totalNetworkCostSum = float('inf')

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

# TODO - should be maxPlantNumber
for kNumber in range(1,3, 500): 
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

    # now set up decision criteria for distribution
    
    def costTruck(distance, demand):
        truckCapacity = 30 # m^3
        unitTruckCost = 200 # $
        # do we need to adjust cost piecewise for truck distances?
        truckNumber = math.ceil((demand)/truckCapacity) # round up
        totalTruckCost = unitTruckCost * truckNumber
        return totalTruckCost

    # reformer cost = reformer * number of stations but demand constraint
    def costReformer(demand):
        #capital costs
        # we assume the station already exists, so all we need to add is the H2 generating equipment
        NGCost = 5 # $ per kg of H2 created
        reformerCapacity = 5000
        reformerCost = 3
        reformerNumber = math.ceil(demand/reformerCapacity) #round up
        # reformer cost is total installation costs divided by demand (fixed segment) plus NG cost
        totalReformerCost = (reformerNumber * reformerCost) + NGCost*demand
        return totalReformerCost

    # Plant Production Cost
    def plantCost(aggDemand):
        plantVarCost = 10 # $per kg H2 produced
        if aggDemand<5000000000:
            plantCost = 2000000000
        elif aggDemand>10000000000:
            plantCost = 5000000000
        else:
            plantCost = 3000000000
        # assume constant variable costs across all plants. 
        return plantCost + plantVarCost*aggDemand




    def assignProduction(plantTotalCost, aggDemand, clusterNodes):
        # figure out what kind of edge to have
        # prodCost is portion of plant total cost assigned to that node - will change in while loop later
        prodCost = plantTotalCost/aggDemand

        # set up reformer cost calculation - what is setup cost of reformer + H2 Production cost
        reformCost = numpy.ndarray((len(clusterNodes),1), dtype = float)
        for i in range(0, len(clusterNodes)):
            reformCost[i] = costReformer(clusterNodes[i, 1].astype(float))    
        # set up edge decision array - initialize as plant, then go through adjustment process in
        # while loop
        edgeDecision = numpy.ndarray(shape=(len(clusterNodes), 1), dtype='S20')
        edgeDecision[:] = "Plant"
        # add costs for plant production
        plantNodeCost = numpy.ndarray(shape=(len(clusterNodes), 1), dtype=float)
        # we want to spit out all information from this function, including costs, so we append all
        # arrays together
        clusterNodes = numpy.append(clusterNodes, reformCost, axis = 1)
        clusterNodes = numpy.append(clusterNodes, edgeDecision, axis = 1)
        clusterNodes = numpy.append(clusterNodes, plantNodeCost, axis = 1)
        transfer = float('inf')
        while transfer != 0:
            # to check the change from the beginning of the while loop to the end (when cost changes
            # are taken in to account) we copy the clusterNodes at the beginning of the loop
            updateArray = copy.copy(clusterNodes)
            # figure out new demand (if some nodes transfer from plant to reformer in previous period)
            # and calculate the new plant cost
            subAggDemand = numpy.sum(clusterNodes[:, 1].astype('float'))
            prodCost = plantCost(subAggDemand)/subAggDemand
            # now we actually calculate the lowest-cost edge between plant and station. 
            for i in range(0, len(clusterNodes)):
                edge = clusterNodes[i, 2]
                edgeTruckCost = costTruck(edge.astype("float32"), subAggDemand)
                if edgeTruckCost + prodCost >= reformCost[i]:
                    clusterNodes[i, 4] = "Reformer"
                    clusterNodes[i, 5] = clusterNodes[i,3]
                else:
                    clusterNodes[i,4] = "Plant"
                    clusterNodes[i, 5] = clusterNodes[i,2].astype("float32")*prodCost

            # now figure out what has changed from last iteration to this one
            transfer = 0           
            for i in range(0, len(clusterNodes)):
                if updateArray[i,4].astype(str) != clusterNodes[i,4].astype(str):
                    transfer +=1
            print 'transfer = ', transfer
        return clusterNodes
    
#TODO - NG station reformer costs
#TODO - foregone gas profits
    start = 0
    totalDecision = []
    for i in range(0, kNumber):
        clusterNodes = label[label[:, 0] == i]
        row = len(clusterNodes)
        aggDemand = plantDemandArray[i, 1].astype("float32")
        plantTotalCost = plantCost(aggDemand)
        Decision = assignProduction(plantTotalCost, aggDemand, clusterNodes)
        if i == 0:
            totalDecision = Decision
        else:
            totalDecision = numpy.append(totalDecision, Decision, axis = 0)
   

   
    print "totalDecision complete at ", datetime.datetime.now()

    

    # total costs and other network details stored in:
    lowCostNetwork = numpy.ndarray(shape=(len(totalDecision), 5), dtype ='S20')

    if networkCost < totalNetworkCostSum:
        lowCostNetwork = totalDecision
        lowKNumber = kNumber

    totalNetworkCostSum = networkCost        

print 'lowest cost gas network cluster number is ', lowKNumber
print lowCostNetwork[0:100]

print 'Now calculate low cost NG station'

NGstationCost = numpy.ndarray(shape=(len(NGstationLatLongAll),1), dtype=float)
for i in range(0, len(NGdemandArray)):
    # need foregone NG revenue here too!
    NGstationCost[i] = costReformer(NGdemandArray[i].astype(float)) 
    



# need to calculate total station costs per steady-state year
def StationProfit(stationArray, H2price, gasPrice):
    return null
 
plantNodes = totalDecision[totalDecision[:,4] == "Plant"]
reformerNodes = totalDecision[totalDecision[:,4] =="Reformer"]
plantNodesCost = plantNodes[:,5].astype(float)
reformerNodesCost = reformerNodes[:,5].astype(float)
networkCost = numpy.sum(plantNodesCost) + numpy.sum(reformerNodesCost)
print "Network Cost for ", kNumber, "plants is ", networkCost
print 'analysis complete at ', datetime.datetime.now()




    # Production cost at plant - assume same costs over all regions
      





    # TO FIX
    # all production costs covered
    # catch all non-positive inequalities
    # make sure last array entries aren't getting dropped
    # include nodeStorage cost & capacity
    # cost of reforming, but will have to write in preclusion of other network
    # and only use NG stations

 
    #export everything from "result" array
    #numpy.savetxt("OptOutput@{0}.csv".format(datetime.datetime.now()), result, delimiter=",", fmt="%s")






        


