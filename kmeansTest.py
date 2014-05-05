import pymprog    # Import the module
import numpy
import csv
import math
import scipy.cluster.vq


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
kNumber = 2
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

for i in range(0, x):
    latLongDist = distance_on_unit_sphere(stationLatLong[i, 0], stationLatLong[i,1], plantArray[i,0], plantArray[i,1])
    latLongDistArray.append(latLongDist)

#adjust array to numpy and reshape
latLongDistArray = numpy.array(latLongDistArray)

print latLongDistArray


