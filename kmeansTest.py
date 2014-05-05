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

print "centroid"
print centroid
print "plantArray"
print plantArray

