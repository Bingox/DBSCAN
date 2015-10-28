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
	uint32_t max_cluster_id;
	std::ofstream results("results");
	std::ofstream max_cluster("max_cluster");

	DBSCAN dbs(eps, minPts, numThreads);
	DBSCAN::ClusterData data = dbs.read_file(file_name, ',');

	std::vector<uint32_t> v_selected_normalize(data.size2(), 1);
	dbs.normalize( data, v_selected_normalize );

	std::vector<double> weights(data.size2(), 1);
	dbs.dbscan(data, weights);
	DBSCAN::ClusterMap cmap = dbs.gen_cluster_map();
	dbs.print_cluster_stat(cmap);
	max_cluster_id = dbs.get_max_cluster(cmap);
	std::cout<<"max cluster id:"<<max_cluster_id<<std::endl;

	dbs.write_max_cluster(max_cluster_id, file_name, max_cluster);
	results << dbs;

	max_cluster.close();
	results.close();

	return 0;
}
