#include "fstream"
#include "dbscan.h"

using namespace clustering;

int main(int argc, char** argv) {
	if(argc != 5) {
		printf("Usage: filter_dbscan [file] [eps] [minPts] [numThreads]\n");
		return 0;
	}	

	std::string file_name = argv[1];
	double eps = atof(argv[2]);
	size_t minPts = atoi(argv[3]);
	int numThreads = atoi(argv[4]);
	std::ofstream results("results");

	DBSCAN dbs(eps, minPts, numThreads);
	DBSCAN::ClusterData data = dbs.read_file(file_name, ',');
	dbs.dbscan( data );
	results << dbs;
	results.close();

	return 0;
}
