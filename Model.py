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
    latLongDist = distance_on_unit_sphere(stationLatLong[i, 0], stationLatLong[i,1], stationLatLong[i,2], stationLatLong[i,3])
    latLongDistArray.append(latLongDist)

#adjust array to numpy and reshape
latLongDistArray = numpy.array(latLongDistArray)
numpy.transpose(latLongDistArray)

# Regionalize demand

stationDemand = [[100][400][300][100][250][600][50][200][75]
                 [30][600][400][100][350][125][500][700][10]
                 [40][400][200][75][500]]


# Set up capacity constraints - cost per unit of energy energy delivered by mile - to fix

pipeCapacity = 5
truckCapacity = 2


# Create empty problem instance
p = pymprog.model('FuelNetworkCosts')

#create variables: (set up #legs as binary variables to either be truck or pipeline
binaryLegs = []


# set up minimization as objective function - minimizing cost with either option
p.min(numpy.outer(binaryLegs, latLongDistArray)*pipeCapacity - numpy.outer((1-binaryLegs)*stationDemand/truckCapacity), 'myobj')

#constraints
r=p.st(
    binaryLegs = (0 or 1):
    for each i in range(0, x):
         binaryLegs[x, 0]*pipeCapacity*latLongDistArray[x,0] - (1-binaryLegs[x,0])*stationDemand/truckCapacity >= stationDemand[x,0]                           
    )


#solve and report
p.solve()
print 'Z = %g;' % p.vobj()  # print obj value
# Print struct variable names and primal values
print ';\n'.join('%s = %g {dual: %g}' % (
   x[i].name, x[i].primal, x[i].dual)
                    for i in cid)
print ';\n'.join('%s = %g {dual: %g}' % (
   r[i].name, r[i].primal, r[i].dual)
                    for i in rid)


# Since version 0.3.0
print p.reportKKT()
print "Environment:", pymprog.env
