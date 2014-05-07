import pymprog    # Import the module
import numpy
import csv
import math
import scipy.cluster.vq
import datetime
import random


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


#Station Latitude in first column, Longitude in second, and remove headers, convert to float
stationLatLong = result[1:,[20,21]]
stationLatLong = stationLatLong.astype('float32')

#cluster
kNumber = 100
centroid, label = scipy.cluster.vq.kmeans2(stationLatLong, kNumber, iter=50, thresh=1e-05, minit='random', missing='warn')

plantArray = numpy.ndarray(shape=(len(label),2), dtype=float, order='F')

#now add centroid lat/longs to result (import file) to calculate distances
# copy labels, then replace going along for i in len(centroid)

for i in range(0, len(label)):
    groupID = label[i] 
    plantCenter = centroid[groupID, :]
    plantArray[i] = plantCenter 


def distance_on_unit_sphere(lat1, long1, lat2, long2):
    # Convert latitude and longitude to
    # spherical coordinates in radians.
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
    arc = math.acos( cos )

    # multiply by radius of earth in miles
    arc *= 3963.1676
    return arc

#generate distances:
x = len(stationLatLong)
latLongDistArray = []

#sometimes code doesn't run because lat/longs between station & plants too
#similar - divide by 0 error. Ought to fix, but can be re-run to fix.
for i in range(0, x):
    latLongDist = distance_on_unit_sphere(stationLatLong[i, 0], stationLatLong[i,1], plantArray[i,0], plantArray[i,1])
    latLongDistArray.append(latLongDist)

#adjust array to numpy and reshape
latLongDistArray = numpy.array(latLongDistArray)

#add back column headers and reshape
clusterIDArray = ["ClusterID"]
label = numpy.concatenate((clusterIDArray, label), axis = 1)
label = numpy.reshape(label, (len(result), 1))

plantIDArray = [["PlantLat", "PlantLong"]]
plantIDArray = numpy.array(plantIDArray)
plantArray = numpy.concatenate((plantIDArray, plantArray), axis = 0)
plantArray = numpy.reshape(plantArray, (len(result), 2))

distArray = ["Distance (Mi)"]
latLongDistArray = numpy.concatenate((distArray, latLongDistArray), axis = 1)
latLongDistArray = numpy.reshape(latLongDistArray, (len(result), 1))

result = numpy.append(result, label, axis =1)
result = numpy.append(result, plantArray, axis =1)
result = numpy.append(result, latLongDistArray, axis =1)

numpy.savetxt("OptOutput@{0}.csv".format(datetime.datetime.now()), result, delimiter=",", fmt="%s")


#import demand information
# dummy import with random #s

demandArray = numpy.ndarray(shape=(len(stationLatLong),1), dtype=float, order='F')

for i in range (0, len(stationLatLong)):
    demandArray[i] = random.random()

demandArray *= 1000000

# need to set up station cost function
    # include nodeStorage cost & capacity


# now set up decision criteria for distribution
# calculate costs of each distribution method, then pick the lowest and
# assign it to a station. 

def costPipeline(distance, demand, nodeStorage)
    # set up pipeline constraint
    pipeCapacity = 100 # m^3/ mi
    unitPipeCost = 300 # $/mile

    # TODO - convert demand to m^3

    totalPipeCost = distance * unitPipeCost
    # Figure out if we can actually store  the right quantity in
    # pipeline setup
    if (demand - nodeStorage)/distance > pipeCapacity:
        capacityBool = False
        else capacityBool = True
        
    return totalPipeCost, capacityBool
    
def costTruck(distance, demand, nodeStorage)
    truckCapacity = 30 # m^3
    unitTruckCost = 200 # $



# reformer cost = reformer * number of stations but demand constraint
    


